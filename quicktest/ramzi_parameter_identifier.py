import numpy as np
from scipy.optimize import least_squares

try:
    from .ramzi_spectrum_simulator import (
        CONTROL_PARAM_NAMES,
        DEFAULT_CONTROL_PARAMS,
        simulate_clean_spectrum,
        simulate_noisy_spectrum,
    )
except ImportError:
    from ramzi_spectrum_simulator import (
        CONTROL_PARAM_NAMES,
        DEFAULT_CONTROL_PARAMS,
        simulate_clean_spectrum,
        simulate_noisy_spectrum,
    )


PHASE_LOWER_BOUND = -np.pi
PHASE_UPPER_BOUND = np.pi


def wrap_phase(x):
    """将相位包裹到 [-pi, pi) 区间。

    用法：
    1. 输入标量或数组 `x`（单位弧度）。
    2. 返回同形状数组，便于相位优化与误差统计。
    """
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def pack_control_params(control_params):
    """把控制参数字典打包为向量。

    用法：
    1. 传入包含 `CONTROL_PARAM_NAMES` 所有键的字典。
    2. 返回按固定顺序排列的一维 `numpy` 向量。
    """
    return np.array([control_params[name] for name in CONTROL_PARAM_NAMES], dtype=float)


def unpack_control_params(vector):
    """把控制参数向量还原为字典。

    用法：
    1. 传入参数向量（可超出主值区间）。
    2. 函数内部会先做相位包裹，再返回参数字典。
    """
    vector = wrap_phase(np.asarray(vector, dtype=float))
    return {name: float(value) for name, value in zip(CONTROL_PARAM_NAMES, vector)}


DEFAULT_VECTOR = pack_control_params(DEFAULT_CONTROL_PARAMS)
LOWER_BOUNDS = np.full(DEFAULT_VECTOR.shape, PHASE_LOWER_BOUND, dtype=float)
UPPER_BOUNDS = np.full(DEFAULT_VECTOR.shape, PHASE_UPPER_BOUND, dtype=float)


def interpolate_simulated_power(wavelength_nm, control_vector, fixed_params=None, observe_port="C2"):
    """在测量波长点上插值仿真功率。

    用法：
    1. 输入目标波长数组 `wavelength_nm` 与控制向量 `control_vector`。
    2. 内部调用 `simulate_clean_spectrum` 生成全谱。
    3. 返回插值后的仿真功率数组，用于与测量值对齐比较。
    """
    control_params = unpack_control_params(control_vector)
    sim = simulate_clean_spectrum(
        control_params=control_params,
        fixed_params=fixed_params,
        observe_port=observe_port,
        wavelength_window_nm=None,
    )
    sim_wl = np.asarray(sim["lambda_nm"], dtype=float)
    sim_power = np.asarray(sim["clean_power"], dtype=float)
    order = np.argsort(sim_wl)
    sim_wl = sim_wl[order]
    sim_power = sim_power[order]
    return np.interp(np.asarray(wavelength_nm, dtype=float), sim_wl, sim_power)


def fit_affine_scale(sim_power, measured_power):
    """拟合仿真功率到测量功率的仿射映射。

    用法：
    1. 输入仿真功率 `sim_power` 与测量功率 `measured_power`。
    2. 通过最小二乘求解 `measured ~= scale * sim + offset`。
    3. 返回 `(scale, offset)`。
    """
    sim_power = np.asarray(sim_power, dtype=float)
    measured_power = np.asarray(measured_power, dtype=float)
    A = np.column_stack([sim_power, np.ones_like(sim_power)])
    coeffs, _, _, _ = np.linalg.lstsq(A, measured_power, rcond=None)
    scale, offset = coeffs
    return float(scale), float(offset)


def build_residual_vector(
    control_vector,
    wavelength_nm,
    measured_power,
    fixed_params=None,
    observe_port="C2",
    fit_scale_and_offset=True,
    slope_weight=0.15,
):
    """构建优化残差向量。

    用法：
    1. 根据 `control_vector` 生成并插值仿真功率。
    2. 可选拟合线性幅度/偏置 (`fit_scale_and_offset`)。
    3. 同时拼接功率残差与斜率残差（由 `slope_weight` 控制）。
    4. 返回给 `scipy.optimize.least_squares` 使用的一维残差。
    """
    sim_power = interpolate_simulated_power(
        wavelength_nm=wavelength_nm,
        control_vector=control_vector,
        fixed_params=fixed_params,
        observe_port=observe_port,
    )

    measured_power = np.asarray(measured_power, dtype=float)
    sim_power = np.asarray(sim_power, dtype=float)

    if fit_scale_and_offset:
        scale, offset = fit_affine_scale(sim_power, measured_power)
    else:
        scale, offset = 1.0, 0.0

    aligned_sim = scale * sim_power + offset

    power_norm = np.maximum(np.std(measured_power), 1e-12)
    power_residual = (aligned_sim - measured_power) / power_norm

    if slope_weight > 0.0:
        slope_sim = np.gradient(aligned_sim, wavelength_nm)
        slope_meas = np.gradient(measured_power, wavelength_nm)
        slope_norm = np.maximum(np.std(slope_meas), 1e-12)
        slope_residual = slope_weight * (slope_sim - slope_meas) / slope_norm
        residual = np.concatenate([power_residual, slope_residual])
    else:
        residual = power_residual

    return residual


def objective_cost(
    control_vector,
    wavelength_nm,
    measured_power,
    fixed_params=None,
    observe_port="C2",
    fit_scale_and_offset=True,
    slope_weight=0.15,
):
    """计算二范数目标函数值（0.5 * ||residual||^2）。

    用法：
    1. 输入与 `build_residual_vector` 相同参数。
    2. 返回标量损失，便于评估不同解的优劣。
    """
    residual = build_residual_vector(
        control_vector=control_vector,
        wavelength_nm=wavelength_nm,
        measured_power=measured_power,
        fixed_params=fixed_params,
        observe_port=observe_port,
        fit_scale_and_offset=fit_scale_and_offset,
        slope_weight=slope_weight,
    )
    return 0.5 * float(np.dot(residual, residual))


def generate_initial_guesses(
    n_starts=24,
    seed=None,
    center=None,
    perturbation_std=0.35,
):
    """生成多起点优化的初始猜测集合。

    用法：
    1. `n_starts` 控制初值数量，`seed` 控制随机性复现。
    2. 以 `center` 为中心随机扰动，并混入全局均匀采样。
    3. 返回包裹到 [-pi, pi) 的向量列表。
    """
    rng = np.random.default_rng(seed)
    if center is None:
        center = DEFAULT_VECTOR
    center = np.asarray(center, dtype=float)

    guesses = [wrap_phase(center)]
    for _ in range(max(n_starts - 1, 0)):
        if rng.random() < 0.5:
            guess = center + perturbation_std * np.pi * rng.standard_normal(center.shape)
        else:
            guess = rng.uniform(PHASE_LOWER_BOUND, PHASE_UPPER_BOUND, size=center.shape)
        guesses.append(wrap_phase(guess))
    return guesses


def circular_mean(vectors):
    """计算角度向量的圆统计均值。

    用法：
    1. 输入形状为 `(n_samples, n_params)` 的角度数组。
    2. 返回每个参数维度的圆均值（弧度）。
    """
    vectors = np.asarray(vectors, dtype=float)
    return np.arctan2(np.mean(np.sin(vectors), axis=0), np.mean(np.cos(vectors), axis=0))


def circular_std(vectors):
    """计算角度向量的圆统计标准差。

    用法：
    1. 输入形状为 `(n_samples, n_params)` 的角度数组。
    2. 返回每个参数维度的圆标准差（弧度）。
    """
    vectors = np.asarray(vectors, dtype=float)
    R = np.sqrt(np.mean(np.cos(vectors), axis=0) ** 2 + np.mean(np.sin(vectors), axis=0) ** 2)
    R = np.clip(R, 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def identify_control_params_from_power_spectrum(
    wavelength_nm,
    measured_power,
    fixed_params=None,
    observe_port="C2",
    n_starts=24,
    top_k=5,
    seed=None,
    fit_scale_and_offset=True,
    slope_weight=0.15,
    max_nfev=300,
    elite_cost_ratio=2.0,
    elite_cost_abs_margin=0.1,
    verbose=False,
):
    """从线性功率光谱反演控制参数。

    用法：
    1. 输入测量波长 `wavelength_nm` 与测量功率 `measured_power`。
    2. 通过 `n_starts` 多起点最小二乘搜索最优参数。
    3. 返回最优解、精英集合统计、拟合曲线与每次运行记录。
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    measured_power = np.asarray(measured_power, dtype=float)

    initial_guesses = generate_initial_guesses(n_starts=n_starts, seed=seed)
    runs = []

    for idx, x0 in enumerate(initial_guesses):
        result = least_squares(
            fun=build_residual_vector,
            x0=x0,
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            args=(
                wavelength_nm,
                measured_power,
                fixed_params,
                observe_port,
                fit_scale_and_offset,
                slope_weight,
            ),
            method="trf",
            max_nfev=max_nfev,
            x_scale="jac",
        )

        solution_vector = wrap_phase(result.x)
        final_sim = interpolate_simulated_power(
            wavelength_nm=wavelength_nm,
            control_vector=solution_vector,
            fixed_params=fixed_params,
            observe_port=observe_port,
        )
        if fit_scale_and_offset:
            scale, offset = fit_affine_scale(final_sim, measured_power)
        else:
            scale, offset = 1.0, 0.0
        aligned_sim = scale * final_sim + offset
        rmse = float(np.sqrt(np.mean((aligned_sim - measured_power) ** 2)))

        run_info = {
            "start_index": idx,
            "x0": wrap_phase(x0),
            "solution_vector": solution_vector,
            "solution_params": unpack_control_params(solution_vector),
            "cost": float(result.cost),
            "rmse": rmse,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "scale": float(scale),
            "offset": float(offset),
        }
        runs.append(run_info)
        if verbose:
            print(f"run {idx:02d}: cost={run_info['cost']:.6g}, rmse={rmse:.6g}, success={result.success}")

    runs.sort(key=lambda item: item["cost"])
    best = runs[0]

    elite_candidates = [
        item
        for item in runs
        if item["cost"] <= max(best["cost"] * elite_cost_ratio, best["cost"] + elite_cost_abs_margin)
    ]
    elites = elite_candidates[: max(1, min(top_k, len(elite_candidates)))]
    elite_vectors = np.array([item["solution_vector"] for item in elites], dtype=float)
    ensemble_vector = wrap_phase(circular_mean(elite_vectors))
    ensemble_std = circular_std(elite_vectors)

    best_sim = interpolate_simulated_power(
        wavelength_nm=wavelength_nm,
        control_vector=best["solution_vector"],
        fixed_params=fixed_params,
        observe_port=observe_port,
    )
    best_aligned = best["scale"] * best_sim + best["offset"]

    ensemble_sim = interpolate_simulated_power(
        wavelength_nm=wavelength_nm,
        control_vector=ensemble_vector,
        fixed_params=fixed_params,
        observe_port=observe_port,
    )
    if fit_scale_and_offset:
        ensemble_scale, ensemble_offset = fit_affine_scale(ensemble_sim, measured_power)
    else:
        ensemble_scale, ensemble_offset = 1.0, 0.0
    ensemble_aligned = ensemble_scale * ensemble_sim + ensemble_offset
    ensemble_rmse = float(np.sqrt(np.mean((ensemble_aligned - measured_power) ** 2)))

    return {
        "best_params": best["solution_params"],
        "best_vector": best["solution_vector"],
        "best_cost": best["cost"],
        "best_rmse": best["rmse"],
        "best_scale": best["scale"],
        "best_offset": best["offset"],
        "best_fitted_power": best_aligned,
        "ensemble_params": unpack_control_params(ensemble_vector),
        "ensemble_vector": ensemble_vector,
        "ensemble_circular_std": {name: float(std) for name, std in zip(CONTROL_PARAM_NAMES, ensemble_std)},
        "ensemble_rmse": ensemble_rmse,
        "ensemble_scale": float(ensemble_scale),
        "ensemble_offset": float(ensemble_offset),
        "ensemble_fitted_power": ensemble_aligned,
        "wavelength_nm": wavelength_nm,
        "measured_power": measured_power,
        "observe_port": observe_port,
        "fit_scale_and_offset": fit_scale_and_offset,
        "slope_weight": slope_weight,
        "n_elites": len(elites),
        "elite_cost_ratio": float(elite_cost_ratio),
        "elite_cost_abs_margin": float(elite_cost_abs_margin),
        "runs": runs,
    }


def identify_control_params_from_db_spectrum(
    wavelength_nm,
    measured_db,
    power_floor=1e-12,
    **kwargs,
):
    """从 dB 光谱反演控制参数。

    用法：
    1. 输入 `measured_db`（dB）与波长数组。
    2. 内部先转换到线性功率，再调用功率域识别函数。
    3. 其余参数通过 `**kwargs` 透传。
    """
    measured_power = np.maximum(10.0 ** (np.asarray(measured_db, dtype=float) / 10.0), power_floor)
    return identify_control_params_from_power_spectrum(
        wavelength_nm=wavelength_nm,
        measured_power=measured_power,
        **kwargs,
    )


def summarize_parameter_error(estimated_params, reference_params):
    """汇总估计参数与参考参数的包裹误差。

    用法：
    1. 输入两个参数字典（估计值与参考真值）。
    2. 返回每个参数在弧度和角度下的包裹误差统计。
    """
    summary = {}
    for name in CONTROL_PARAM_NAMES:
        delta = wrap_phase(estimated_params[name] - reference_params[name])
        summary[name] = {
            "estimated": float(estimated_params[name]),
            "reference": float(reference_params[name]),
            "wrapped_error_rad": float(delta),
            "wrapped_error_deg": float(np.degrees(delta)),
        }
    return summary


if __name__ == "__main__":
    demo = simulate_noisy_spectrum(
        seed=7,
        relative_noise_std=0.01,
        additive_noise_std=1e-4,
        observe_port="C2",
    )

    result = identify_control_params_from_power_spectrum(
        wavelength_nm=demo["lambda_nm"],
        measured_power=demo["noisy_power"],
        fixed_params=demo["fixed_params"],
        observe_port=demo["observe_port"],
        n_starts=20,
        top_k=5,
        seed=123,
        fit_scale_and_offset=True,
        slope_weight=0.15,
        verbose=True,
    )

    print("\nBest recovered parameters:")
    for key, value in result["best_params"].items():
        print(f"  {key}: {value:.6f}")

    print("\nWrapped parameter errors against ground truth:")
    error_summary = summarize_parameter_error(result["best_params"], demo["control_params"])
    for key, info in error_summary.items():
        print(f"  {key}: {info['wrapped_error_deg']:.4f} deg")

    print(f"\nBest RMSE in power domain: {result['best_rmse']:.6e}")
    print(f"Ensemble RMSE in power domain: {result['ensemble_rmse']:.6e}")
