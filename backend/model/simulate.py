from __future__ import annotations

import argparse
import ast
import json
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.interface.matlab_bridge import matlab_engine_session

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATLAB_DIR = REPO_ROOT / "backend" / "model" / "MATLAB"
SUPPORTED_CONSTRAINT_OPERATORS = frozenset({"==", "<=", ">="})
CONSTRAINT_ABS_TOL = 1e-9
CONSTRAINT_REL_TOL = 1e-9


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file {config_file} must contain a YAML mapping at the top level.")

    return _resolve_model_config(raw_config, config_file)


def simulate_from_config(
    config_path: str | Path,
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    config = load_model_config(config_path)
    return simulate_from_dict(config, engine=engine)


def simulate_from_dict(
    config: Mapping[str, Any],
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    resolved_config = dict(config)
    matlab_dir = Path(resolved_config["model"]["matlab_dir"]).resolve()
    entrypoint = str(resolved_config["model"]["entrypoint"])
    payload = json.dumps(resolved_config, ensure_ascii=False)

    with matlab_engine_session(engine, extra_paths=(matlab_dir,)) as matlab_engine:
        matlab_result = matlab_engine.feval(entrypoint, payload, nargout=1)

    result = _convert_matlab_value(matlab_result)
    if not isinstance(result, dict):
        raise TypeError("MATLAB model must return a struct-like result.")

    _attach_parameter_metadata(result, resolved_config)
    return result


def save_result_npz(result: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).resolve()
    np.savez(
        destination,
        wavelength_nm=np.asarray(result["wavelength_nm"]),
        frequency_hz=np.asarray(result["frequency_hz"]),
        complex_response=np.asarray(result["complex_response"]),
        power_linear=np.asarray(result["power_linear"]),
        power_db=np.asarray(result["power_db"]),
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MATLAB photonic model from a YAML config.")
    parser.add_argument("config", type=Path, help="Path to a .yml/.yaml model config file.")
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path to save the main simulation arrays as a .npz file.",
    )
    args = parser.parse_args(argv)

    result = simulate_from_config(args.config)
    if args.save_npz is not None:
        output_file = save_result_npz(result, args.save_npz)
        print(f"saved_npz: {output_file}")

    _print_summary(result)
    return 0


def _resolve_model_config(raw_config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    model_section = _require_mapping(raw_config.get("model"), "model")
    simulation_section = _require_mapping(raw_config.get("simulation"), "simulation")
    wavelength_section = _require_mapping(simulation_section.get("wavelength_nm"), "simulation.wavelength_nm")
    outputs_section = _optional_mapping(raw_config.get("outputs"))
    parameters_section = _require_mapping(raw_config.get("parameters"), "parameters")

    config_dir = config_path.parent
    matlab_dir = _resolve_existing_path(
        model_section.get("matlab_dir", "backend/model/MATLAB"),
        config_dir=config_dir,
    )

    resolved = {
        "version": int(raw_config.get("version", 1)),
        "config_path": str(config_path),
        "model": {
            "name": _require_text(model_section.get("name"), "model.name"),
            "entrypoint": _require_text(
                model_section.get("entrypoint", model_section.get("name")),
                "model.entrypoint",
            ),
            "matlab_dir": str(matlab_dir),
        },
        "simulation": {
            "wavelength_nm": {
                "start": _require_float(wavelength_section.get("start"), "simulation.wavelength_nm.start"),
                "stop": _require_float(wavelength_section.get("stop"), "simulation.wavelength_nm.stop"),
                "step": _require_positive_float(
                    wavelength_section.get("step"),
                    "simulation.wavelength_nm.step",
                ),
            }
        },
        "outputs": {
            "observe_port": _require_text(outputs_section.get("observe_port", "C2"), "outputs.observe_port").upper()
        },
        "parameters": {
            "tunable": _resolve_tunable_parameters(parameters_section.get("tunable", {})),
            "fixed": _resolve_fixed_parameters(parameters_section.get("fixed", {})),
        },
    }

    resolved["parameters"]["constraints"] = _resolve_parameter_constraints(
        parameters_section.get("constraints"),
        tunable_names=set(resolved["parameters"]["tunable"]),
        fixed_names=set(resolved["parameters"]["fixed"]),
    )

    wavelength_config = resolved["simulation"]["wavelength_nm"]
    if wavelength_config["stop"] <= wavelength_config["start"]:
        raise ValueError("simulation.wavelength_nm.stop must be greater than start.")

    return resolved


def _resolve_tunable_parameters(raw_tunable: Any) -> dict[str, dict[str, Any]]:
    tunable_section = _optional_mapping(raw_tunable)
    resolved: dict[str, dict[str, Any]] = {}
    for name, spec in tunable_section.items():
        spec_mapping = _require_mapping(spec, f"parameters.tunable.{name}")
        value = _normalize_yaml_scalar(spec_mapping.get("value"), f"parameters.tunable.{name}.value")
        bounds = spec_mapping.get("bounds")
        if bounds is not None:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"parameters.tunable.{name}.bounds must be a 2-element list.")
            resolved_bounds = [
                _require_float(bounds[0], f"parameters.tunable.{name}.bounds[0]"),
                _require_float(bounds[1], f"parameters.tunable.{name}.bounds[1]"),
            ]
            if resolved_bounds[0] > resolved_bounds[1]:
                raise ValueError(f"parameters.tunable.{name}.bounds must satisfy lower <= upper.")
        else:
            resolved_bounds = None

        resolved[name] = {
            "value": value,
            "bounds": resolved_bounds,
        }

    return resolved


def _resolve_fixed_parameters(raw_fixed: Any) -> dict[str, Any]:
    fixed_section = _optional_mapping(raw_fixed)
    return {
        name: _normalize_yaml_scalar(value, f"parameters.fixed.{name}")
        for name, value in fixed_section.items()
    }


def evaluate_parameter_constraints(
    tunable_values: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]] | None,
    *,
    atol: float = CONSTRAINT_ABS_TOL,
    rtol: float = CONSTRAINT_REL_TOL,
) -> dict[str, Any]:
    resolved_values = {
        str(name): _require_real_numeric_scalar(value, f"parameters.tunable.{name}")
        for name, value in tunable_values.items()
    }

    evaluations: list[dict[str, Any]] = []
    for index, constraint in enumerate(constraints or (), start=1):
        lhs_expr = _require_text(constraint.get("lhs"), f"parameters.constraints[{index - 1}].lhs")
        rhs_expr = _require_text(constraint.get("rhs"), f"parameters.constraints[{index - 1}].rhs")
        operator = _require_text(
            constraint.get("operator"),
            f"parameters.constraints[{index - 1}].operator",
        )
        if operator not in SUPPORTED_CONSTRAINT_OPERATORS:
            raise ValueError(
                f"parameters.constraints[{index - 1}].operator must be one of "
                f"{sorted(SUPPORTED_CONSTRAINT_OPERATORS)}."
            )

        lhs_value = _evaluate_constraint_expression(
            lhs_expr,
            resolved_values,
            field_name=f"parameters.constraints[{index - 1}].lhs",
        )
        rhs_value = _evaluate_constraint_expression(
            rhs_expr,
            resolved_values,
            field_name=f"parameters.constraints[{index - 1}].rhs",
        )
        residual = lhs_value - rhs_value
        tolerance = max(atol, rtol * max(abs(lhs_value), abs(rhs_value), 1.0))

        if operator == "==":
            satisfied = abs(residual) <= tolerance
            violation = max(abs(residual) - tolerance, 0.0)
        elif operator == "<=":
            satisfied = residual <= tolerance
            violation = max(residual - tolerance, 0.0)
        else:
            satisfied = residual >= -tolerance
            violation = max(-residual - tolerance, 0.0)

        evaluations.append(
            {
                "name": str(constraint.get("name", f"constraint_{index}")),
                "lhs": lhs_expr,
                "operator": operator,
                "rhs": rhs_expr,
                "variables": list(constraint.get("variables", [])),
                "lhs_value": lhs_value,
                "rhs_value": rhs_value,
                "residual": residual,
                "tolerance": tolerance,
                "violation": violation,
                "satisfied": satisfied,
            }
        )

    violations = [evaluation for evaluation in evaluations if not evaluation["satisfied"]]
    return {
        "satisfied": not violations,
        "evaluations": evaluations,
        "violations": violations,
    }


def _resolve_parameter_constraints(
    raw_constraints: Any,
    *,
    tunable_names: set[str],
    fixed_names: set[str],
) -> list[dict[str, Any]]:
    if raw_constraints is None:
        return []

    if not isinstance(raw_constraints, list):
        raise ValueError("parameters.constraints must be a list.")

    resolved: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, constraint in enumerate(raw_constraints):
        field_name = f"parameters.constraints[{index}]"
        constraint_mapping = _require_mapping(constraint, field_name)

        raw_name = constraint_mapping.get("name")
        if raw_name is None:
            name = f"constraint_{index + 1}"
        else:
            name = _require_text(raw_name, f"{field_name}.name")

        if name in seen_names:
            raise ValueError(f"Duplicate constraint name: {name}")
        seen_names.add(name)

        operator = _require_text(constraint_mapping.get("operator"), f"{field_name}.operator")
        if operator not in SUPPORTED_CONSTRAINT_OPERATORS:
            raise ValueError(
                f"{field_name}.operator must be one of {sorted(SUPPORTED_CONSTRAINT_OPERATORS)}."
            )

        lhs, lhs_variables = _normalize_constraint_expression(
            constraint_mapping.get("lhs"),
            f"{field_name}.lhs",
            allowed_names=tunable_names | fixed_names,
        )
        rhs, rhs_variables = _normalize_constraint_expression(
            constraint_mapping.get("rhs"),
            f"{field_name}.rhs",
            allowed_names=tunable_names | fixed_names,
        )
        variables = sorted(set(lhs_variables) | set(rhs_variables))
        if not variables:
            raise ValueError(
                f"{field_name} must reference at least one parameters.tunable or parameters.fixed entry."
            )

        resolved.append(
            {
                "name": name,
                "lhs": lhs,
                "operator": operator,
                "rhs": rhs,
                "variables": variables,
            }
        )

    return resolved


def _normalize_yaml_scalar(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        keys = set(value.keys())
        if keys <= {"real", "imag"} and "real" in keys:
            return {
                "real": _require_float(value.get("real"), f"{field_name}.real"),
                "imag": _require_float(value.get("imag", 0.0), f"{field_name}.imag"),
            }

        return {str(key): _normalize_yaml_scalar(child, f"{field_name}.{key}") for key, child in value.items()}

    if isinstance(value, list):
        return [_normalize_yaml_scalar(item, f"{field_name}[]") for item in value]

    if isinstance(value, (bool, int, float, str)):
        return value

    raise TypeError(f"{field_name} must be a scalar, a complex mapping, or a list.")


def _convert_matlab_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _convert_matlab_value(child) for key, child in value.items()}

    if isinstance(value, (str, bool, int, float, complex)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [_convert_matlab_value(item) for item in value]

    try:
        array = np.asarray(value)
    except Exception:
        return value

    if array.dtype == object:
        return [_convert_matlab_value(item) for item in array.reshape(-1)]

    if array.shape == ():
        return array.item()

    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)

    return array


def _attach_parameter_metadata(result: dict[str, Any], resolved_config: Mapping[str, Any]) -> None:
    parameters_result = result.get("parameters")
    if isinstance(parameters_result, Mapping):
        normalized_parameters = dict(parameters_result)
    else:
        normalized_parameters = {}

    tunable_values = normalized_parameters.get("tunable")
    if isinstance(tunable_values, Mapping):
        normalized_tunable_values = {
            str(name): _require_real_numeric_scalar(value, f"result.parameters.tunable.{name}")
            for name, value in tunable_values.items()
        }
    else:
        normalized_tunable_values = _extract_tunable_values(resolved_config["parameters"]["tunable"])

    normalized_parameters["tunable"] = normalized_tunable_values
    normalized_parameters.setdefault("fixed", resolved_config["parameters"]["fixed"])
    normalized_parameters["tunable_specs"] = resolved_config["parameters"]["tunable"]
    normalized_parameters["constraints"] = resolved_config["parameters"]["constraints"]
    constraint_values = dict(normalized_tunable_values)
    constraint_values.update(_extract_fixed_constraint_values(resolved_config["parameters"]["fixed"]))
    normalized_parameters["constraint_status"] = evaluate_parameter_constraints(
        constraint_values,
        resolved_config["parameters"]["constraints"],
    )

    result["parameters"] = normalized_parameters


def _extract_tunable_values(tunable_specs: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): _require_real_numeric_scalar(spec["value"], f"parameters.tunable.{name}.value")
        for name, spec in tunable_specs.items()
    }


def _extract_fixed_constraint_values(fixed_values: Mapping[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for name, value in fixed_values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, Real):
            resolved[str(name)] = float(value)
    return resolved


def _resolve_existing_path(raw_path: Any, *, config_dir: Path) -> Path:
    path_text = _require_text(raw_path, "model.matlab_dir")
    candidate = Path(path_text)

    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"MATLAB directory does not exist: {resolved}")
        return resolved

    for base_dir in (config_dir, REPO_ROOT):
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"MATLAB directory does not exist: {path_text}")


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): child for key, child in value.items()}


def _optional_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping.")
    return {str(key): child for key, child in value.items()}


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_positive_float(value: Any, field_name: str) -> float:
    numeric_value = _require_float(value, field_name)
    if numeric_value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return numeric_value


def _require_real_numeric_scalar(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real numeric scalar.")
    return float(value)


def _normalize_constraint_expression(
    value: Any,
    field_name: str,
    *,
    allowed_names: set[str],
) -> tuple[str, list[str]]:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric or a valid expression string.")

    if isinstance(value, Real):
        expression_text = repr(float(value))
        return expression_text, []

    expression_text = _require_text(value, field_name)
    parsed_expression, variables = _parse_constraint_expression(
        expression_text,
        field_name=field_name,
        allowed_names=allowed_names,
    )
    return ast.unparse(parsed_expression), variables


def _parse_constraint_expression(
    expression_text: str,
    *,
    field_name: str,
    allowed_names: set[str],
) -> tuple[ast.AST, list[str]]:
    try:
        parsed = ast.parse(expression_text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{field_name} contains invalid expression syntax: {expression_text}") from exc

    referenced_names: set[str] = set()
    _validate_constraint_node(
        parsed,
        field_name=field_name,
        allowed_names=allowed_names,
        referenced_names=referenced_names,
    )
    return parsed, sorted(referenced_names)


def _validate_constraint_node(
    node: ast.AST,
    *,
    field_name: str,
    allowed_names: set[str],
    referenced_names: set[str],
) -> None:
    if isinstance(node, ast.Expression):
        _validate_constraint_node(
            node.body,
            field_name=field_name,
            allowed_names=allowed_names,
            referenced_names=referenced_names,
        )
        return

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        _validate_constraint_node(
            node.left,
            field_name=field_name,
            allowed_names=allowed_names,
            referenced_names=referenced_names,
        )
        _validate_constraint_node(
            node.right,
            field_name=field_name,
            allowed_names=allowed_names,
            referenced_names=referenced_names,
        )
        return

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_constraint_node(
            node.operand,
            field_name=field_name,
            allowed_names=allowed_names,
            referenced_names=referenced_names,
        )
        return

    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ValueError(
                f"{field_name} references unknown parameter {node.id!r}. "
                "Constraint expressions may only reference parameters.tunable and parameters.fixed entries."
            )
        referenced_names.add(node.id)
        return

    if isinstance(node, ast.Constant) and isinstance(node.value, Real) and not isinstance(node.value, bool):
        return

    raise ValueError(
        f"{field_name} may only contain numeric literals, tunable names, parentheses, "
        "and the operators +, -, *, /, **."
    )


def _evaluate_constraint_expression(
    expression_text: str,
    variable_values: Mapping[str, float],
    *,
    field_name: str,
) -> float:
    parsed, _ = _parse_constraint_expression(
        expression_text,
        field_name=field_name,
        allowed_names=set(variable_values),
    )
    return _evaluate_constraint_node(parsed.body, variable_values)


def _evaluate_constraint_node(node: ast.AST, variable_values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Name):
        return float(variable_values[node.id])

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_constraint_node(node.operand, variable_values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand

    if isinstance(node, ast.BinOp):
        left = _evaluate_constraint_node(node.left, variable_values)
        right = _evaluate_constraint_node(node.right, variable_values)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right

    raise TypeError(f"Unsupported constraint expression node: {ast.dump(node)}")


def _print_summary(result: Mapping[str, Any]) -> None:
    wavelength_nm = np.asarray(result["wavelength_nm"], dtype=float)
    power_db = np.asarray(result["power_db"], dtype=float)

    print(f"model_name: {result['model_name']}")
    print(f"port_name: {result['port_name']}")
    print(f"samples: {wavelength_nm.size}")
    print(f"wavelength_range_nm: {wavelength_nm.min():.6f} -> {wavelength_nm.max():.6f}")
    print(f"power_db_range: {power_db.min():.6f} -> {power_db.max():.6f}")

    parameters = result.get("parameters")
    if isinstance(parameters, Mapping):
        constraint_status = parameters.get("constraint_status")
        if isinstance(constraint_status, Mapping) and constraint_status.get("evaluations"):
            print(f"constraints_satisfied: {bool(constraint_status['satisfied'])}")
            print(f"constraint_count: {len(constraint_status['evaluations'])}")


if __name__ == "__main__":
    raise SystemExit(main())
