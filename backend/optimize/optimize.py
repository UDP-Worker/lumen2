from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.optimize import Bounds, differential_evolution, minimize

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.model import load_model_config, save_result_npz, simulate_from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_DATA_ROOT = REPO_ROOT / "backend" / "optimize" / "data"
EPSILON = 1e-12


@dataclass(slots=True)
class OptimizationTarget:
    center_nm: float
    bandwidth_3db_nm: float
    transition_nm: float


@dataclass(slots=True)
class LossWeights:
    passband: float = 4.0
    stopband: float = 4.0
    ripple: float = 0.5
    center: float = 1.5
    bandwidth: float = 1.5
    constraint: float = 100.0


@dataclass(slots=True)
class GlobalSettings:
    enabled: bool = True
    maxiter: int = 1
    popsize: int = 2
    polish: bool = False
    seed: int = 42
    strategy: str = "best1bin"
    tol: float = 0.01
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7


@dataclass(slots=True)
class LocalSettings:
    enabled: bool = True
    method: str = "Powell"
    maxiter: int = 12
    maxfev: int | None = None
    xtol: float = 1e-3
    ftol: float = 1e-3


@dataclass(slots=True)
class LoggingSettings:
    tensorboard: bool = True
    plot_every: int = 4


DEFAULT_LOSS_WEIGHTS = LossWeights()
DEFAULT_GLOBAL_SETTINGS = GlobalSettings()
DEFAULT_LOCAL_SETTINGS = LocalSettings()
DEFAULT_LOGGING_SETTINGS = LoggingSettings()


@dataclass(slots=True)
class DecisionVariable:
    name: str
    members: tuple[str, ...]
    initial: float
    bounds: tuple[float, float]


@dataclass(slots=True)
class Parameterization:
    mode: str
    decision_variables: tuple[DecisionVariable, ...]
    frozen_tunables: dict[str, float]
    tunable_order: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.decision_variables)

    def initial_vector(self) -> NDArray[np.float64]:
        return np.asarray([variable.initial for variable in self.decision_variables], dtype=float)

    def bounds(self) -> list[tuple[float, float]]:
        return [variable.bounds for variable in self.decision_variables]

    def lower_bounds(self) -> NDArray[np.float64]:
        return np.asarray([variable.bounds[0] for variable in self.decision_variables], dtype=float)

    def upper_bounds(self) -> NDArray[np.float64]:
        return np.asarray([variable.bounds[1] for variable in self.decision_variables], dtype=float)

    def decision_values(self, vector: Sequence[float]) -> dict[str, float]:
        array = np.asarray(vector, dtype=float)
        if array.shape != (self.dimension,):
            raise ValueError(f"Decision vector must have shape ({self.dimension},), got {array.shape}.")
        return {
            variable.name: float(value)
            for variable, value in zip(self.decision_variables, array, strict=True)
        }

    def expand(self, vector: Sequence[float]) -> dict[str, float]:
        values = dict(self.frozen_tunables)
        array = np.asarray(vector, dtype=float)
        if array.shape != (self.dimension,):
            raise ValueError(f"Decision vector must have shape ({self.dimension},), got {array.shape}.")
        for variable, value in zip(self.decision_variables, array, strict=True):
            scalar = float(value)
            for member in variable.members:
                values[member] = scalar

        missing = [name for name in self.tunable_order if name not in values]
        if missing:
            raise ValueError(f"Expanded decision vector is missing tunables: {missing}")

        return {name: float(values[name]) for name in self.tunable_order}

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "decision_variables": [
                {
                    "name": variable.name,
                    "members": list(variable.members),
                    "initial": float(variable.initial),
                    "bounds": [float(variable.bounds[0]), float(variable.bounds[1])],
                }
                for variable in self.decision_variables
            ],
            "frozen_tunables": {
                name: float(value)
                for name, value in self.frozen_tunables.items()
            },
            "tunable_order": list(self.tunable_order),
        }


@dataclass(slots=True)
class EvaluationRecord:
    step: int
    stage: str
    total_loss: float
    loss_terms: dict[str, float]
    metrics: dict[str, float]
    decision_values: dict[str, float]
    tunable_values: dict[str, float]
    result: dict[str, Any]
    elapsed_seconds: float
    from_cache: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "stage": self.stage,
            "total_loss": float(self.total_loss),
            "loss_terms": {
                key: float(value)
                for key, value in self.loss_terms.items()
            },
            "metrics": {
                key: float(value)
                for key, value in self.metrics.items()
            },
            "decision_values": {
                key: float(value)
                for key, value in self.decision_values.items()
            },
            "tunable_values": {
                key: float(value)
                for key, value in self.tunable_values.items()
            },
            "elapsed_seconds": float(self.elapsed_seconds),
            "from_cache": bool(self.from_cache),
        }


class OptimizationBudgetExceeded(RuntimeError):
    pass


class TensorBoardScalarLogger:
    def __init__(self, log_dir: Path) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(str(log_dir.resolve()))

    def log_scalars(self, step: int, values: Mapping[str, float]) -> None:
        summary_values = [
            self._summary_cls.Value(tag=str(tag), simple_value=float(value))
            for tag, value in values.items()
            if np.isfinite(float(value))
        ]
        if not summary_values:
            return
        event = self._event_cls(
            wall_time=time.time(),
            step=int(step),
            summary=self._summary_cls(value=summary_values),
        )
        self._writer.add_event(event)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class OptimizationRecorder:
    def __init__(
        self,
        *,
        run_dir: Path,
        target: OptimizationTarget,
        parameterization: Parameterization,
        logging_settings: LoggingSettings,
    ) -> None:
        self.run_dir = run_dir
        self.target = target
        self.parameterization = parameterization
        self.logging_settings = logging_settings
        self.history_csv_path = self.run_dir / "history.csv"
        self.history_jsonl_path = self.run_dir / "history.jsonl"
        self.history_plot_path = self.run_dir / "history.png"
        self.best_plot_path = self.run_dir / "best_spectrum.png"
        self.best_npz_path = self.run_dir / "best_result.npz"
        self.optimized_config_path = self.run_dir / "optimized_config.yml"
        self.summary_json_path = self.run_dir / "optimization_summary.json"
        self.tensorboard_log_dir = self.run_dir / "tensorboard"
        self._history_rows: list[dict[str, Any]] = []
        self._jsonl_handle = self.history_jsonl_path.open("w", encoding="utf-8")
        self._csv_handle = self.history_csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_handle,
            fieldnames=self._build_csv_fieldnames(),
        )
        self._csv_writer.writeheader()
        self._tensorboard_logger = self._create_tensorboard_logger()

    def _create_tensorboard_logger(self) -> TensorBoardScalarLogger | None:
        if not self.logging_settings.tensorboard:
            return None
        try:
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            return TensorBoardScalarLogger(self.tensorboard_log_dir)
        except Exception:
            return None

    def _build_csv_fieldnames(self) -> list[str]:
        fields = [
            "step",
            "stage",
            "total_loss",
            "elapsed_seconds",
            "peak_wavelength_nm",
            "peak_power_linear",
            "peak_power_db",
            "measured_center_nm",
            "measured_bandwidth_3db_nm",
            "passband_mean_linear",
            "stopband_mean_linear",
            "stopband_max_linear",
            "passband_ripple_linear",
            "extinction_ratio_db",
            "loss_passband",
            "loss_stopband",
            "loss_ripple",
            "loss_center",
            "loss_bandwidth",
            "loss_constraint",
        ]
        fields.extend(f"decision__{variable.name}" for variable in self.parameterization.decision_variables)
        fields.extend(f"tunable__{name}" for name in self.parameterization.tunable_order)
        return fields

    def record(self, evaluation: EvaluationRecord) -> None:
        self._history_rows.append(evaluation.to_json())
        json.dump(evaluation.to_json(), self._jsonl_handle, ensure_ascii=False)
        self._jsonl_handle.write("\n")
        self._jsonl_handle.flush()

        csv_row = {
            "step": int(evaluation.step),
            "stage": evaluation.stage,
            "total_loss": float(evaluation.total_loss),
            "elapsed_seconds": float(evaluation.elapsed_seconds),
            "peak_wavelength_nm": float(evaluation.metrics["peak_wavelength_nm"]),
            "peak_power_linear": float(evaluation.metrics["peak_power_linear"]),
            "peak_power_db": float(evaluation.metrics["peak_power_db"]),
            "measured_center_nm": float(evaluation.metrics["measured_center_nm"]),
            "measured_bandwidth_3db_nm": float(evaluation.metrics["measured_bandwidth_3db_nm"]),
            "passband_mean_linear": float(evaluation.metrics["passband_mean_linear"]),
            "stopband_mean_linear": float(evaluation.metrics["stopband_mean_linear"]),
            "stopband_max_linear": float(evaluation.metrics["stopband_max_linear"]),
            "passband_ripple_linear": float(evaluation.metrics["passband_ripple_linear"]),
            "extinction_ratio_db": float(evaluation.metrics["extinction_ratio_db"]),
            "loss_passband": float(evaluation.loss_terms["passband"]),
            "loss_stopband": float(evaluation.loss_terms["stopband"]),
            "loss_ripple": float(evaluation.loss_terms["ripple"]),
            "loss_center": float(evaluation.loss_terms["center"]),
            "loss_bandwidth": float(evaluation.loss_terms["bandwidth"]),
            "loss_constraint": float(evaluation.loss_terms["constraint"]),
        }
        csv_row.update(
            {
                f"decision__{key}": float(value)
                for key, value in evaluation.decision_values.items()
            }
        )
        csv_row.update(
            {
                f"tunable__{key}": float(value)
                for key, value in evaluation.tunable_values.items()
            }
        )
        self._csv_writer.writerow(csv_row)
        self._csv_handle.flush()

        if self._tensorboard_logger is not None:
            tensorboard_scalars = {
                "loss/total": evaluation.total_loss,
                "loss/passband": evaluation.loss_terms["passband"],
                "loss/stopband": evaluation.loss_terms["stopband"],
                "loss/ripple": evaluation.loss_terms["ripple"],
                "loss/center": evaluation.loss_terms["center"],
                "loss/bandwidth": evaluation.loss_terms["bandwidth"],
                "loss/constraint": evaluation.loss_terms["constraint"],
                "metrics/peak_wavelength_nm": evaluation.metrics["peak_wavelength_nm"],
                "metrics/peak_power_linear": evaluation.metrics["peak_power_linear"],
                "metrics/measured_center_nm": evaluation.metrics["measured_center_nm"],
                "metrics/measured_bandwidth_3db_nm": evaluation.metrics["measured_bandwidth_3db_nm"],
                "metrics/passband_mean_linear": evaluation.metrics["passband_mean_linear"],
                "metrics/stopband_mean_linear": evaluation.metrics["stopband_mean_linear"],
                "metrics/stopband_max_linear": evaluation.metrics["stopband_max_linear"],
                "metrics/passband_ripple_linear": evaluation.metrics["passband_ripple_linear"],
                "metrics/extinction_ratio_db": evaluation.metrics["extinction_ratio_db"],
                "meta/elapsed_seconds": evaluation.elapsed_seconds,
            }
            tensorboard_scalars.update(
                {
                    f"decision/{key}": float(value)
                    for key, value in evaluation.decision_values.items()
                }
            )
            self._tensorboard_logger.log_scalars(evaluation.step, tensorboard_scalars)

    def maybe_update_history_plot(self, *, force: bool = False) -> None:
        if not force and len(self._history_rows) % max(self.logging_settings.plot_every, 1) != 0:
            return
        if not self._history_rows:
            return

        steps = np.asarray([row["step"] for row in self._history_rows], dtype=float)
        total_loss = np.asarray([row["total_loss"] for row in self._history_rows], dtype=float)
        centers = np.asarray([row["metrics"]["measured_center_nm"] for row in self._history_rows], dtype=float)
        bandwidths = np.asarray(
            [row["metrics"]["measured_bandwidth_3db_nm"] for row in self._history_rows],
            dtype=float,
        )
        passband_mean = np.asarray(
            [row["metrics"]["passband_mean_linear"] for row in self._history_rows],
            dtype=float,
        )
        stopband_mean = np.asarray(
            [row["metrics"]["stopband_mean_linear"] for row in self._history_rows],
            dtype=float,
        )

        figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(steps, total_loss, color="#0f766e", linewidth=1.8)
        axes[0].set_ylabel("total loss")
        axes[0].grid(alpha=0.25)

        axes[1].plot(steps, centers, label="measured center", color="#1d4ed8", linewidth=1.5)
        axes[1].axhline(self.target.center_nm, color="#1d4ed8", linestyle="--", alpha=0.7, label="target center")
        axes[1].plot(steps, bandwidths, label="measured 3 dB BW", color="#b45309", linewidth=1.5)
        axes[1].axhline(
            self.target.bandwidth_3db_nm,
            color="#b45309",
            linestyle="--",
            alpha=0.7,
            label="target 3 dB BW",
        )
        axes[1].set_ylabel("nm")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        axes[2].plot(steps, passband_mean, label="passband mean", color="#047857", linewidth=1.5)
        axes[2].plot(steps, stopband_mean, label="stopband mean", color="#be123c", linewidth=1.5)
        axes[2].set_xlabel("evaluation")
        axes[2].set_ylabel("linear power")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best")

        figure.tight_layout()
        figure.savefig(self.history_plot_path, dpi=180)
        plt.close(figure)

    def update_best_artifacts(
        self,
        *,
        evaluation: EvaluationRecord,
        raw_config: Mapping[str, Any],
        base_resolved_config: Mapping[str, Any],
    ) -> None:
        self._save_best_spectrum_plot(evaluation)
        save_result_npz(evaluation.result, self.best_npz_path)
        optimized_raw_config = _build_optimized_raw_config(raw_config, evaluation.tunable_values)
        _write_yaml(self.optimized_config_path, optimized_raw_config)
        self.maybe_update_history_plot(force=True)
        _write_json(
            self.run_dir / "best_metrics.json",
            {
                "step": int(evaluation.step),
                "stage": evaluation.stage,
                "total_loss": float(evaluation.total_loss),
                "loss_terms": {
                    key: float(value)
                    for key, value in evaluation.loss_terms.items()
                },
                "metrics": {
                    key: float(value)
                    for key, value in evaluation.metrics.items()
                },
                "decision_values": {
                    key: float(value)
                    for key, value in evaluation.decision_values.items()
                },
                "tunable_values": {
                    key: float(value)
                    for key, value in evaluation.tunable_values.items()
                },
                "artifacts": {
                    "best_result_npz": str(self.best_npz_path),
                    "optimized_config": str(self.optimized_config_path),
                    "best_spectrum_plot": str(self.best_plot_path),
                },
                "model_name": str(base_resolved_config["model"]["name"]),
            },
        )

    def _save_best_spectrum_plot(self, evaluation: EvaluationRecord) -> None:
        wavelength_nm = np.asarray(evaluation.result["wavelength_nm"], dtype=float)
        power_linear = np.asarray(evaluation.result["power_linear"], dtype=float)
        power_db = np.asarray(evaluation.result["power_db"], dtype=float)
        passband_half_width = self.target.bandwidth_3db_nm / 2.0
        stopband_guard = passband_half_width + self.target.transition_nm

        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(wavelength_nm, power_linear, color="#0f172a", linewidth=1.8)
        axes[0].axvspan(
            self.target.center_nm - passband_half_width,
            self.target.center_nm + passband_half_width,
            color="#93c5fd",
            alpha=0.35,
            label="target passband",
        )
        axes[0].axvspan(
            self.target.center_nm - stopband_guard,
            self.target.center_nm - passband_half_width,
            color="#fef08a",
            alpha=0.25,
            label="transition",
        )
        axes[0].axvspan(
            self.target.center_nm + passband_half_width,
            self.target.center_nm + stopband_guard,
            color="#fef08a",
            alpha=0.25,
        )
        axes[0].set_ylabel("linear power")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(wavelength_nm, power_db, color="#1d4ed8", linewidth=1.5)
        axes[1].axvline(
            evaluation.metrics["measured_center_nm"],
            color="#dc2626",
            linestyle="--",
            linewidth=1.2,
            label="measured center",
        )
        axes[1].axvline(
            self.target.center_nm,
            color="#0f766e",
            linestyle="--",
            linewidth=1.2,
            label="target center",
        )
        axes[1].set_xlabel("wavelength (nm)")
        axes[1].set_ylabel("power (dB)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        figure.suptitle(
            (
                f"Best loss={evaluation.total_loss:.4f}, "
                f"center={evaluation.metrics['measured_center_nm']:.6f} nm, "
                f"BW={evaluation.metrics['measured_bandwidth_3db_nm']:.6f} nm"
            ),
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(self.best_plot_path, dpi=180)
        plt.close(figure)

    def close(self) -> None:
        if self._tensorboard_logger is not None:
            self._tensorboard_logger.close()
        self._jsonl_handle.close()
        self._csv_handle.close()


class OptimizationRunner:
    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        parameterization: Parameterization,
        target: OptimizationTarget,
        weights: LossWeights,
        global_settings: GlobalSettings,
        local_settings: LocalSettings,
        logging_settings: LoggingSettings,
        run_dir: Path,
        max_evaluations: int,
        engine: Any | None = None,
    ) -> None:
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be greater than 0.")

        self.raw_config = copy.deepcopy(dict(raw_config))
        self.resolved_config = copy.deepcopy(dict(resolved_config))
        self.parameterization = parameterization
        self.target = target
        self.weights = weights
        self.global_settings = global_settings
        self.local_settings = local_settings
        self.logging_settings = logging_settings
        self.run_dir = run_dir
        self.max_evaluations = int(max_evaluations)
        self.engine = engine
        self.recorder = OptimizationRecorder(
            run_dir=self.run_dir,
            target=self.target,
            parameterization=self.parameterization,
            logging_settings=self.logging_settings,
        )
        self.stage = "initial"
        self._cache: dict[tuple[float, ...], EvaluationRecord] = {}
        self._history: list[EvaluationRecord] = []
        self._best_record: EvaluationRecord | None = None
        self._best_key: tuple[float, ...] | None = None

    @property
    def evaluation_count(self) -> int:
        return len(self._history)

    @property
    def best_record(self) -> EvaluationRecord | None:
        return self._best_record

    def initial_vector(self) -> NDArray[np.float64]:
        return self.parameterization.initial_vector()

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def objective(self, vector: Sequence[float]) -> float:
        return self.evaluate(vector).total_loss

    def evaluate(self, vector: Sequence[float]) -> EvaluationRecord:
        clipped_vector = self._clip_vector(vector)
        cache_key = tuple(np.round(clipped_vector, decimals=12))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if self.evaluation_count >= self.max_evaluations:
            raise OptimizationBudgetExceeded(
                f"Reached the evaluation budget ({self.max_evaluations} simulations)."
            )

        start_time = time.perf_counter()
        decision_values = self.parameterization.decision_values(clipped_vector)
        tunable_values = self.parameterization.expand(clipped_vector)
        resolved_config = self._build_resolved_config(tunable_values)
        simulation_result = simulate_from_dict(resolved_config, engine=self.engine)
        total_loss, loss_terms, metrics = self._compute_loss(simulation_result)
        elapsed_seconds = time.perf_counter() - start_time

        record = EvaluationRecord(
            step=self.evaluation_count + 1,
            stage=self.stage,
            total_loss=total_loss,
            loss_terms=loss_terms,
            metrics=metrics,
            decision_values=decision_values,
            tunable_values=tunable_values,
            result=simulation_result,
            elapsed_seconds=elapsed_seconds,
        )
        self._cache[cache_key] = record
        self._history.append(record)
        self.recorder.record(record)
        if self._best_record is None or record.total_loss < self._best_record.total_loss:
            self._best_record = record
            self._best_key = cache_key
            self.recorder.update_best_artifacts(
                evaluation=record,
                raw_config=self.raw_config,
                base_resolved_config=self.resolved_config,
            )
        else:
            self.recorder.maybe_update_history_plot()

        return record

    def _clip_vector(self, vector: Sequence[float]) -> NDArray[np.float64]:
        array = np.asarray(vector, dtype=float)
        if array.shape != (self.parameterization.dimension,):
            raise ValueError(
                f"Expected a decision vector with shape ({self.parameterization.dimension},), got {array.shape}."
            )
        return np.clip(array, self.parameterization.lower_bounds(), self.parameterization.upper_bounds())

    def _build_resolved_config(self, tunable_values: Mapping[str, float]) -> dict[str, Any]:
        resolved_config = copy.deepcopy(self.resolved_config)
        for name, value in tunable_values.items():
            resolved_config["parameters"]["tunable"][name]["value"] = float(value)
        return resolved_config

    def _compute_loss(self, simulation_result: Mapping[str, Any]) -> tuple[float, dict[str, float], dict[str, float]]:
        wavelength_nm = np.asarray(simulation_result["wavelength_nm"], dtype=float)
        power_linear = np.asarray(simulation_result["power_linear"], dtype=float)
        power_db = np.asarray(simulation_result["power_db"], dtype=float)
        if wavelength_nm.ndim != 1 or power_linear.ndim != 1 or wavelength_nm.size != power_linear.size:
            raise ValueError("Simulation result must contain 1D wavelength_nm and power_linear arrays of equal length.")

        peak_index = int(np.argmax(power_linear))
        peak_wavelength_nm = float(wavelength_nm[peak_index])
        peak_power_linear = float(power_linear[peak_index])
        peak_power_db = float(power_db[peak_index])
        measured_center_nm, measured_bandwidth_nm = _estimate_center_and_bandwidth_nm(
            wavelength_nm,
            power_linear,
        )

        passband_mask = _passband_mask(wavelength_nm, self.target)
        stopband_mask = _stopband_mask(wavelength_nm, self.target)
        passband_values = np.asarray(power_linear[passband_mask], dtype=float)
        stopband_values = np.asarray(power_linear[stopband_mask], dtype=float)
        passband_values_db = np.asarray(power_db[passband_mask], dtype=float)

        passband_error = float(np.mean(np.square(1.0 - passband_values)))
        stopband_error = float(np.mean(np.square(stopband_values)))
        ripple_error = float(np.var(passband_values))
        normalization = max(self.target.bandwidth_3db_nm, EPSILON)
        center_error = float(((measured_center_nm - self.target.center_nm) / normalization) ** 2)
        bandwidth_error = float(((measured_bandwidth_nm - self.target.bandwidth_3db_nm) / normalization) ** 2)
        constraint_penalty = _constraint_penalty(simulation_result)

        loss_terms = {
            "passband": self.weights.passband * passband_error,
            "stopband": self.weights.stopband * stopband_error,
            "ripple": self.weights.ripple * ripple_error,
            "center": self.weights.center * center_error,
            "bandwidth": self.weights.bandwidth * bandwidth_error,
            "constraint": self.weights.constraint * constraint_penalty,
        }
        total_loss = float(sum(loss_terms.values()))

        passband_mean_linear = float(np.mean(passband_values))
        stopband_mean_linear = float(np.mean(stopband_values))
        stopband_max_linear = float(np.max(stopband_values))
        passband_ripple_linear = float(np.ptp(passband_values))
        extinction_ratio_db = float(np.mean(passband_values_db) - _linear_power_to_db(stopband_max_linear))
        metrics = {
            "peak_wavelength_nm": peak_wavelength_nm,
            "peak_power_linear": peak_power_linear,
            "peak_power_db": peak_power_db,
            "measured_center_nm": measured_center_nm,
            "measured_bandwidth_3db_nm": measured_bandwidth_nm,
            "passband_mean_linear": passband_mean_linear,
            "stopband_mean_linear": stopband_mean_linear,
            "stopband_max_linear": stopband_max_linear,
            "passband_ripple_linear": passband_ripple_linear,
            "extinction_ratio_db": extinction_ratio_db,
        }
        return total_loss, loss_terms, metrics

    def close(self) -> None:
        self.recorder.close()


def run_filter_optimization(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_evaluations: int | None = None,
    global_maxiter: int | None = None,
    global_popsize: int | None = None,
    local_maxiter: int | None = None,
    skip_global: bool = False,
    skip_local: bool = False,
    tensorboard: bool | None = None,
    plot_every: int | None = None,
    seed: int | None = None,
    engine: Any | None = None,
) -> dict[str, Any]:
    raw_config = _load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = str(resolved_config["model"]["name"])
    optimization_settings = _resolve_optimization_settings(raw_config, resolved_config)
    target = optimization_settings["target"]
    weights = optimization_settings["weights"]
    parameterization = optimization_settings["parameterization"]
    global_settings = optimization_settings["global_settings"]
    local_settings = optimization_settings["local_settings"]
    logging_settings = optimization_settings["logging_settings"]
    default_budget = int(optimization_settings["max_evaluations"])

    if parameterization.dimension == 0:
        raise ValueError("Optimization parameterization must expose at least one decision variable.")

    if max_evaluations is not None:
        default_budget = int(max_evaluations)
    if global_maxiter is not None:
        global_settings.maxiter = int(global_maxiter)
    if global_popsize is not None:
        global_settings.popsize = int(global_popsize)
    if local_maxiter is not None:
        local_settings.maxiter = int(local_maxiter)
    if tensorboard is not None:
        logging_settings.tensorboard = bool(tensorboard)
    if plot_every is not None:
        logging_settings.plot_every = int(plot_every)
    if seed is not None:
        global_settings.seed = int(seed)
    if skip_global:
        global_settings.enabled = False
    if skip_local:
        local_settings.enabled = False

    run_dir = _resolve_run_dir(output_dir=output_dir, model_name=model_name)
    runner = OptimizationRunner(
        raw_config=raw_config,
        resolved_config=resolved_config,
        parameterization=parameterization,
        target=target,
        weights=weights,
        global_settings=global_settings,
        local_settings=local_settings,
        logging_settings=logging_settings,
        run_dir=run_dir,
        max_evaluations=default_budget,
        engine=engine,
    )

    global_result: dict[str, Any] | None = None
    local_result: dict[str, Any] | None = None
    termination_reason = "completed"
    start_time = time.perf_counter()
    try:
        initial_record = runner.evaluate(runner.initial_vector())
        current_best_vector = np.asarray(
            [initial_record.decision_values[variable.name] for variable in parameterization.decision_variables],
            dtype=float,
        )

        if global_settings.enabled:
            runner.set_stage("global")
            try:
                differential_result = differential_evolution(
                    runner.objective,
                    bounds=parameterization.bounds(),
                    strategy=global_settings.strategy,
                    maxiter=global_settings.maxiter,
                    popsize=global_settings.popsize,
                    tol=global_settings.tol,
                    mutation=global_settings.mutation,
                    recombination=global_settings.recombination,
                    polish=global_settings.polish,
                    seed=global_settings.seed,
                    workers=1,
                    updating="immediate",
                )
                global_result = _serialize_optimize_result(differential_result)
                current_best_vector = np.asarray(differential_result.x, dtype=float)
            except OptimizationBudgetExceeded as exc:
                termination_reason = str(exc)
                global_result = {
                    "status": "budget_exhausted",
                    "message": termination_reason,
                }
            except Exception as exc:
                termination_reason = f"global stage failed: {exc}"
                global_result = {
                    "status": "failed",
                    "message": termination_reason,
                }

        if local_settings.enabled and runner.best_record is not None:
            runner.set_stage("local")
            current_best_vector = np.asarray(
                [runner.best_record.decision_values[variable.name] for variable in parameterization.decision_variables],
                dtype=float,
            )
            remaining_budget = max(default_budget - runner.evaluation_count, 1)
            local_maxfev = (
                min(local_settings.maxfev, remaining_budget)
                if local_settings.maxfev is not None
                else remaining_budget
            )
            try:
                local_minimize_result = minimize(
                    runner.objective,
                    x0=current_best_vector,
                    method=local_settings.method,
                    bounds=Bounds(parameterization.lower_bounds(), parameterization.upper_bounds()),
                    options={
                        "maxiter": local_settings.maxiter,
                        "maxfev": local_maxfev,
                        "xtol": local_settings.xtol,
                        "ftol": local_settings.ftol,
                    },
                )
                local_result = _serialize_optimize_result(local_minimize_result)
            except OptimizationBudgetExceeded as exc:
                termination_reason = str(exc)
                local_result = {
                    "status": "budget_exhausted",
                    "message": termination_reason,
                }
            except Exception as exc:
                termination_reason = f"local stage failed: {exc}"
                local_result = {
                    "status": "failed",
                    "message": termination_reason,
                }
    finally:
        runner.recorder.maybe_update_history_plot(force=True)
        runner.close()

    elapsed_seconds = time.perf_counter() - start_time
    if runner.best_record is None:
        raise RuntimeError("Optimization did not complete a single simulation.")

    summary = {
        "model_name": model_name,
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir),
        "elapsed_seconds": float(elapsed_seconds),
        "termination_reason": termination_reason,
        "evaluation_count": int(runner.evaluation_count),
        "max_evaluations": int(default_budget),
        "parameterization": parameterization.summary(),
        "target": asdict(target),
        "weights": asdict(weights),
        "global_stage": {
            **asdict(global_settings),
            "result": global_result,
        },
        "local_stage": {
            **asdict(local_settings),
            "result": local_result,
        },
        "logging": {
            **asdict(logging_settings),
            "tensorboard_log_dir": (
                str(run_dir / "tensorboard")
                if logging_settings.tensorboard and (run_dir / "tensorboard").exists()
                else None
            ),
        },
        "best": runner.best_record.to_json(),
        "artifacts": {
            "history_csv": str(run_dir / "history.csv"),
            "history_jsonl": str(run_dir / "history.jsonl"),
            "history_plot": str(run_dir / "history.png"),
            "best_spectrum_plot": str(run_dir / "best_spectrum.png"),
            "best_result_npz": str(run_dir / "best_result.npz"),
            "optimized_config": str(run_dir / "optimized_config.yml"),
        },
    }
    _write_json(run_dir / "optimization_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize photonic filter tunable parameters against a band-pass target. "
            "Defaults are intentionally conservative for limited compute budgets."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--max-evaluations", type=int, default=None, help="Hard cap on MATLAB simulations.")
    parser.add_argument("--global-maxiter", type=int, default=None)
    parser.add_argument("--global-popsize", type=int, default=None)
    parser.add_argument("--local-maxiter", type=int, default=None)
    parser.add_argument("--skip-global", action="store_true", help="Disable differential evolution.")
    parser.add_argument("--skip-local", action="store_true", help="Disable local refinement.")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=None)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--plot-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_filter_optimization(
        args.config,
        output_dir=args.output_dir,
        max_evaluations=args.max_evaluations,
        global_maxiter=args.global_maxiter,
        global_popsize=args.global_popsize,
        local_maxiter=args.local_maxiter,
        skip_global=args.skip_global,
        skip_local=args.skip_local,
        tensorboard=args.tensorboard,
        plot_every=args.plot_every,
        seed=args.seed,
    )
    best = summary["best"]
    print(f"saved_summary: {summary['run_dir']}/optimization_summary.json")
    print(f"run_dir: {summary['run_dir']}")
    print(f"evaluation_count: {summary['evaluation_count']}")
    print(f"best_loss: {best['total_loss']:.6f}")
    print(f"best_center_nm: {best['metrics']['measured_center_nm']:.6f}")
    print(f"best_bandwidth_3db_nm: {best['metrics']['measured_bandwidth_3db_nm']:.6f}")
    tensorboard_dir = summary["logging"].get("tensorboard_log_dir")
    if tensorboard_dir:
        print(f"tensorboard_log_dir: {tensorboard_dir}")
    return 0


def _resolve_optimization_settings(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    optimization_section = _mapping(raw_config.get("optimization"), "optimization")
    target_section = _mapping(optimization_section.get("target"), "optimization.target")
    weights_section = _mapping(optimization_section.get("weights"), "optimization.weights")
    strategy_section = _mapping(optimization_section.get("strategy"), "optimization.strategy")
    global_section = _mapping(strategy_section.get("global"), "optimization.strategy.global")
    local_section = _mapping(strategy_section.get("local"), "optimization.strategy.local")
    logging_section = _mapping(optimization_section.get("logging"), "optimization.logging")

    wavelength_step = float(resolved_config["simulation"]["wavelength_nm"]["step"])
    bandwidth_3db_nm = float(target_section.get("bandwidth_3db_nm", 0.005))
    if bandwidth_3db_nm <= 0.0:
        raise ValueError("optimization.target.bandwidth_3db_nm must be greater than 0.")
    target = OptimizationTarget(
        center_nm=float(target_section.get("center_nm", 1550.0)),
        bandwidth_3db_nm=bandwidth_3db_nm,
        transition_nm=float(target_section.get("transition_nm", max(bandwidth_3db_nm * 0.5, 4.0 * wavelength_step))),
    )
    if target.transition_nm < 0.0:
        raise ValueError("optimization.target.transition_nm must be non-negative.")

    weights = LossWeights(
        passband=float(weights_section.get("passband", DEFAULT_LOSS_WEIGHTS.passband)),
        stopband=float(weights_section.get("stopband", DEFAULT_LOSS_WEIGHTS.stopband)),
        ripple=float(weights_section.get("ripple", DEFAULT_LOSS_WEIGHTS.ripple)),
        center=float(weights_section.get("center", DEFAULT_LOSS_WEIGHTS.center)),
        bandwidth=float(weights_section.get("bandwidth", DEFAULT_LOSS_WEIGHTS.bandwidth)),
        constraint=float(weights_section.get("constraint", DEFAULT_LOSS_WEIGHTS.constraint)),
    )
    global_settings = GlobalSettings(
        enabled=bool(global_section.get("enabled", DEFAULT_GLOBAL_SETTINGS.enabled)),
        maxiter=int(global_section.get("maxiter", DEFAULT_GLOBAL_SETTINGS.maxiter)),
        popsize=int(global_section.get("popsize", DEFAULT_GLOBAL_SETTINGS.popsize)),
        polish=bool(global_section.get("polish", DEFAULT_GLOBAL_SETTINGS.polish)),
        seed=int(global_section.get("seed", DEFAULT_GLOBAL_SETTINGS.seed)),
        strategy=str(global_section.get("strategy", DEFAULT_GLOBAL_SETTINGS.strategy)),
        tol=float(global_section.get("tol", DEFAULT_GLOBAL_SETTINGS.tol)),
        mutation=_resolve_mutation(global_section.get("mutation", list(DEFAULT_GLOBAL_SETTINGS.mutation))),
        recombination=float(global_section.get("recombination", DEFAULT_GLOBAL_SETTINGS.recombination)),
    )
    local_settings = LocalSettings(
        enabled=bool(local_section.get("enabled", DEFAULT_LOCAL_SETTINGS.enabled)),
        method=str(local_section.get("method", DEFAULT_LOCAL_SETTINGS.method)),
        maxiter=int(local_section.get("maxiter", DEFAULT_LOCAL_SETTINGS.maxiter)),
        maxfev=_optional_int(local_section.get("maxfev")),
        xtol=float(local_section.get("xtol", local_section.get("xatol", DEFAULT_LOCAL_SETTINGS.xtol))),
        ftol=float(local_section.get("ftol", local_section.get("fatol", DEFAULT_LOCAL_SETTINGS.ftol))),
    )
    logging_settings = LoggingSettings(
        tensorboard=bool(logging_section.get("tensorboard", DEFAULT_LOGGING_SETTINGS.tensorboard)),
        plot_every=max(1, int(logging_section.get("plot_every", DEFAULT_LOGGING_SETTINGS.plot_every))),
    )
    parameterization = _resolve_parameterization(raw_config, resolved_config)

    return {
        "target": target,
        "weights": weights,
        "parameterization": parameterization,
        "global_settings": global_settings,
        "local_settings": local_settings,
        "logging_settings": logging_settings,
        "max_evaluations": int(optimization_section.get("max_evaluations", 24)),
    }


def _resolve_parameterization(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> Parameterization:
    optimization_section = _mapping(raw_config.get("optimization"), "optimization")
    parameterization_section = _mapping(
        optimization_section.get("parameterization"),
        "optimization.parameterization",
    )
    tunable_specs = _mapping(resolved_config["parameters"]["tunable"], "parameters.tunable")
    tunable_order = tuple(tunable_specs.keys())
    mode = str(parameterization_section.get("mode", "direct")).strip().lower()
    raw_groups = parameterization_section.get("groups")

    if mode == "symmetric4" and raw_groups is None:
        raw_groups = {
            "fai_pair_1": ["fai1", "fai3"],
            "fai_pair_2": ["fai2", "fai4"],
            "theta_pair_1": ["theta1", "theta3"],
            "theta_pair_2": ["theta2", "theta4"],
        }

    include_unlisted = bool(parameterization_section.get("include_unlisted", raw_groups is None))
    decision_variables: list[DecisionVariable] = []
    assigned_tunables: set[str] = set()

    if raw_groups is not None:
        for group_name, member_names, initial_override, bounds_override in _iter_parameter_groups(raw_groups):
            members = tuple(member_names)
            if not members:
                raise ValueError(f"Optimization parameter group {group_name!r} must contain at least one member.")
            unknown_members = [name for name in members if name not in tunable_specs]
            if unknown_members:
                raise ValueError(f"Optimization parameter group {group_name!r} references unknown tunables: {unknown_members}")
            duplicates = [name for name in members if name in assigned_tunables]
            if duplicates:
                raise ValueError(f"Tunable parameters cannot appear in multiple groups: {duplicates}")
            assigned_tunables.update(members)
            initial, bounds = _resolve_group_value_and_bounds(
                group_name=group_name,
                member_names=members,
                tunable_specs=tunable_specs,
                initial_override=initial_override,
                bounds_override=bounds_override,
            )
            decision_variables.append(
                DecisionVariable(
                    name=group_name,
                    members=members,
                    initial=initial,
                    bounds=bounds,
                )
            )

    if raw_groups is None or include_unlisted:
        for tunable_name in tunable_order:
            if tunable_name in assigned_tunables:
                continue
            spec = _mapping(tunable_specs[tunable_name], f"parameters.tunable.{tunable_name}")
            bounds = _require_bounds(spec.get("bounds"), f"parameters.tunable.{tunable_name}.bounds")
            decision_variables.append(
                DecisionVariable(
                    name=tunable_name,
                    members=(tunable_name,),
                    initial=float(spec["value"]),
                    bounds=bounds,
                )
            )
            assigned_tunables.add(tunable_name)

    frozen_tunables = {
        name: float(_mapping(tunable_specs[name], f"parameters.tunable.{name}")["value"])
        for name in tunable_order
        if name not in assigned_tunables
    }
    return Parameterization(
        mode=mode,
        decision_variables=tuple(decision_variables),
        frozen_tunables=frozen_tunables,
        tunable_order=tunable_order,
    )


def _iter_parameter_groups(raw_groups: Any) -> list[tuple[str, list[str], float | None, tuple[float, float] | None]]:
    groups: list[tuple[str, list[str], float | None, tuple[float, float] | None]] = []
    if isinstance(raw_groups, Mapping):
        for raw_name, raw_spec in raw_groups.items():
            group_name = str(raw_name).strip()
            if isinstance(raw_spec, Mapping):
                members = _string_list(raw_spec.get("members"), f"optimization.parameterization.groups.{group_name}.members")
                initial_override = _optional_float(raw_spec.get("initial"))
                bounds_override = (
                    _require_bounds(raw_spec.get("bounds"), f"optimization.parameterization.groups.{group_name}.bounds")
                    if raw_spec.get("bounds") is not None
                    else None
                )
            else:
                members = _string_list(raw_spec, f"optimization.parameterization.groups.{group_name}")
                initial_override = None
                bounds_override = None
            groups.append((group_name, members, initial_override, bounds_override))
        return groups

    if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes, bytearray)):
        for index, item in enumerate(raw_groups):
            item_mapping = _mapping(item, f"optimization.parameterization.groups[{index}]")
            group_name = _require_text(item_mapping.get("name"), f"optimization.parameterization.groups[{index}].name")
            members = _string_list(
                item_mapping.get("members"),
                f"optimization.parameterization.groups[{index}].members",
            )
            initial_override = _optional_float(item_mapping.get("initial"))
            bounds_override = (
                _require_bounds(
                    item_mapping.get("bounds"),
                    f"optimization.parameterization.groups[{index}].bounds",
                )
                if item_mapping.get("bounds") is not None
                else None
            )
            groups.append((group_name, members, initial_override, bounds_override))
        return groups

    raise ValueError("optimization.parameterization.groups must be a mapping or a sequence.")


def _resolve_group_value_and_bounds(
    *,
    group_name: str,
    member_names: Sequence[str],
    tunable_specs: Mapping[str, Any],
    initial_override: float | None,
    bounds_override: tuple[float, float] | None,
) -> tuple[float, tuple[float, float]]:
    member_initials: list[float] = []
    member_lowers: list[float] = []
    member_uppers: list[float] = []
    for member_name in member_names:
        spec = _mapping(tunable_specs[member_name], f"parameters.tunable.{member_name}")
        member_initials.append(float(spec["value"]))
        lower, upper = _require_bounds(spec.get("bounds"), f"parameters.tunable.{member_name}.bounds")
        member_lowers.append(lower)
        member_uppers.append(upper)

    bounds = bounds_override or (max(member_lowers), min(member_uppers))
    if bounds[0] > bounds[1]:
        raise ValueError(
            f"Optimization parameter group {group_name!r} has incompatible member bounds: "
            f"{list(zip(member_names, member_lowers, member_uppers, strict=True))}"
        )

    initial = float(np.mean(member_initials)) if initial_override is None else float(initial_override)
    initial = float(np.clip(initial, bounds[0], bounds[1]))
    return initial, bounds


def _estimate_center_and_bandwidth_nm(
    wavelength_nm: NDArray[np.float64],
    power_linear: NDArray[np.float64],
) -> tuple[float, float]:
    if wavelength_nm.size == 0:
        raise ValueError("wavelength_nm must not be empty.")

    peak_index = int(np.argmax(power_linear))
    peak_wavelength_nm = float(wavelength_nm[peak_index])
    peak_power = float(power_linear[peak_index])
    if not math.isfinite(peak_power) or peak_power <= 0.0:
        return peak_wavelength_nm, 0.0

    threshold = peak_power / 2.0
    left_nm = _find_half_power_crossing(wavelength_nm, power_linear, peak_index, threshold, search_left=True)
    right_nm = _find_half_power_crossing(wavelength_nm, power_linear, peak_index, threshold, search_left=False)
    bandwidth_nm = max(right_nm - left_nm, 0.0)
    center_nm = 0.5 * (left_nm + right_nm)
    return float(center_nm), float(bandwidth_nm)


def _find_half_power_crossing(
    wavelength_nm: NDArray[np.float64],
    power_linear: NDArray[np.float64],
    peak_index: int,
    threshold: float,
    *,
    search_left: bool,
) -> float:
    if search_left:
        for index in range(peak_index, 0, -1):
            y0 = float(power_linear[index - 1])
            y1 = float(power_linear[index])
            if y0 <= threshold <= y1 or y1 <= threshold <= y0:
                x0 = float(wavelength_nm[index - 1])
                x1 = float(wavelength_nm[index])
                return _linear_interpolate(x0, y0, x1, y1, threshold)
        return float(wavelength_nm[0])

    for index in range(peak_index, wavelength_nm.size - 1):
        y0 = float(power_linear[index])
        y1 = float(power_linear[index + 1])
        if y0 >= threshold >= y1 or y1 >= threshold >= y0:
            x0 = float(wavelength_nm[index])
            x1 = float(wavelength_nm[index + 1])
            return _linear_interpolate(x0, y0, x1, y1, threshold)
    return float(wavelength_nm[-1])


def _linear_interpolate(x0: float, y0: float, x1: float, y1: float, target_y: float) -> float:
    if math.isclose(y0, y1):
        return float(0.5 * (x0 + x1))
    return float(x0 + (target_y - y0) * (x1 - x0) / (y1 - y0))


def _passband_mask(wavelength_nm: NDArray[np.float64], target: OptimizationTarget) -> NDArray[np.bool_]:
    half_width = target.bandwidth_3db_nm / 2.0
    mask = np.abs(wavelength_nm - target.center_nm) <= half_width + EPSILON
    if np.any(mask):
        return mask
    nearest = int(np.argmin(np.abs(wavelength_nm - target.center_nm)))
    fallback = np.zeros_like(wavelength_nm, dtype=bool)
    fallback[nearest] = True
    return fallback


def _stopband_mask(wavelength_nm: NDArray[np.float64], target: OptimizationTarget) -> NDArray[np.bool_]:
    cutoff = target.bandwidth_3db_nm / 2.0 + target.transition_nm
    mask = np.abs(wavelength_nm - target.center_nm) >= cutoff - EPSILON
    if np.any(mask):
        return mask
    return np.logical_not(_passband_mask(wavelength_nm, target))


def _constraint_penalty(simulation_result: Mapping[str, Any]) -> float:
    parameters = simulation_result.get("parameters")
    if not isinstance(parameters, Mapping):
        return 0.0
    constraint_status = parameters.get("constraint_status")
    if not isinstance(constraint_status, Mapping):
        return 0.0
    violations = constraint_status.get("violations")
    if not isinstance(violations, Sequence):
        return 0.0
    penalty = 0.0
    for violation in violations:
        if not isinstance(violation, Mapping):
            continue
        penalty += float(violation.get("violation", 0.0)) ** 2
    return float(penalty)


def _linear_power_to_db(power_linear: float) -> float:
    return float(10.0 * math.log10(max(power_linear, np.finfo(float).tiny)))


def _resolve_run_dir(*, output_dir: str | Path | None, model_name: str) -> Path:
    if output_dir is not None:
        run_dir = Path(output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (OPTIMIZATION_DATA_ROOT / model_name / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_optimize_result(result: Any) -> dict[str, Any]:
    payload = {
        "success": bool(getattr(result, "success", False)),
        "status": int(getattr(result, "status", 0)),
        "message": str(getattr(result, "message", "")),
        "fun": float(getattr(result, "fun", np.nan)),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
    }
    x_value = getattr(result, "x", None)
    if x_value is not None:
        payload["x"] = [float(value) for value in np.asarray(x_value, dtype=float)]
    return payload


def _build_optimized_raw_config(
    raw_config: Mapping[str, Any],
    tunable_values: Mapping[str, float],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(raw_config))
    parameters = payload.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping.")
    tunable = parameters.setdefault("tunable", {})
    if not isinstance(tunable, dict):
        raise ValueError("parameters.tunable must be a mapping.")

    for name, value in tunable_values.items():
        existing = tunable.get(name)
        if isinstance(existing, Mapping):
            updated = dict(existing)
            updated["value"] = float(value)
            tunable[name] = updated
        else:
            tunable[name] = {"value": float(value)}
    return payload


def _load_raw_yaml(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).resolve().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a top-level YAML mapping.")
    return payload


def _write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)
    return destination


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    return destination


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): child for key, child in value.items()}


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric value.")
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Expected an integer value.")
    return int(value)


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    result = []
    for index, item in enumerate(value):
        result.append(_require_text(item, f"{field_name}[{index}]"))
    return result


def _require_bounds(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-element numeric sequence.")
    lower = float(value[0])
    upper = float(value[1])
    if lower > upper:
        raise ValueError(f"{field_name} must satisfy lower <= upper.")
    return lower, upper


def _resolve_mutation(value: Any) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar, scalar
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 2:
        lower = float(value[0])
        upper = float(value[1])
        if lower > upper:
            raise ValueError("optimization.strategy.global.mutation must satisfy lower <= upper.")
        return lower, upper
    raise ValueError("optimization.strategy.global.mutation must be a number or a 2-element sequence.")


if __name__ == "__main__":
    raise SystemExit(main())
