import numpy as np

from ramzi_model import simulate_ramzi


CONTROL_PARAM_NAMES = (
    "thetai",
    "thetao",
    "fait",
    "faib",
    "fai1",
    "fai2",
    "fai3",
    "fai4",
)


DEFAULT_FIXED_PARAMS = {
    "E1": 0.0 + 0.0j,
    "E2": 1.0 + 0.0j,
    "Ki": 0.5,
    "Ko": 0.5,
    "theta1": -0.622 * np.pi,
    "theta2": -0.73 * np.pi,
    "theta3": -0.622 * np.pi,
    "theta4": -0.73 * np.pi,
    "Alfadb": 15.0,
    "ng": 4.3,
    "L1": 350e-6,
    "L2": 3000e-6,
    "L3": 350e-6,
    "L4": 3000e-6,
    "c": 3e8,
}


DEFAULT_CONTROL_PARAMS = {
    "thetai": 0.5 * np.pi,
    "thetao": 0.5 * np.pi,
    "fait": 0.495 * np.pi,
    "faib": -0.495 * np.pi,
    "fai1": -0.0468 * np.pi,
    "fai2": -0.6842 * np.pi,
    "fai3": -0.0518 * np.pi,
    "fai4": -0.6198 * np.pi,
}


def original_frequency_grid() -> np.ndarray:
    """返回原 MATLAB 脚本使用的频率采样网格。

    用法：
    1. 直接调用得到一维频率数组（Hz）。
    2. 该网格用于与原始仿真配置保持一致。
    """
    return np.arange(1.93523e14, 1.93558e14 + 0.5 * 0.00000025e14, 0.00000025e14)


def _merge_params(control_params=None, fixed_params=None):
    """合并默认参数、固定参数和可调参数。

    用法：
    1. `fixed_params` 用于覆盖结构常量。
    2. `control_params` 用于覆盖第一层可控相位参数。
    3. 返回可直接传给 `simulate_ramzi` 的参数字典。
    """
    merged = dict(DEFAULT_FIXED_PARAMS)
    merged.update(DEFAULT_CONTROL_PARAMS)
    if fixed_params is not None:
        merged.update(fixed_params)
    if control_params is not None:
        merged.update(control_params)
    return merged


def _sort_and_crop(lambda_nm, arrays, wavelength_window_nm=(1549.9, 1550.2)):
    """按波长升序排序并按窗口裁剪数组。

    用法：
    1. 传入波长数组和同长度数据字典 `arrays`。
    2. `wavelength_window_nm=None` 时仅排序不裁剪。
    3. 返回排序/裁剪后的波长和数据字典。
    """
    order = np.argsort(lambda_nm)
    lambda_sorted = lambda_nm[order]
    arrays_sorted = {name: value[order] for name, value in arrays.items()}

    if wavelength_window_nm is None:
        return lambda_sorted, arrays_sorted

    wl_min, wl_max = wavelength_window_nm
    mask = (lambda_sorted >= wl_min) & (lambda_sorted <= wl_max)
    lambda_cropped = lambda_sorted[mask]
    arrays_cropped = {name: value[mask] for name, value in arrays_sorted.items()}
    return lambda_cropped, arrays_cropped


def simulate_clean_spectrum(
    control_params=None,
    fixed_params=None,
    observe_port="C2",
    wavelength_window_nm=(1549.9, 1550.2),
):
    """仿真无噪声光谱。

    用法：
    1. `control_params` 传入待调相位参数（如 `thetai`、`thetao` 等）。
    2. `fixed_params` 传入固定结构参数；未传时使用默认值。
    3. `observe_port` 选择输出端口 `"C1"` 或 `"C2"`。
    4. 返回波长、线性功率、dB 功率及完整参数信息。
    """
    params = _merge_params(control_params=control_params, fixed_params=fixed_params)
    result = simulate_ramzi(f=original_frequency_grid(), **params)

    lambda_nm = result["lambda"] * 1e9
    lambda_nm, arrays = _sort_and_crop(
        lambda_nm,
        {
            "C1": result["C1"],
            "C2": result["C2"],
            "CdB1": result["CdB1"],
            "CdB2": result["CdB2"],
        },
        wavelength_window_nm=wavelength_window_nm,
    )

    if observe_port not in ("C1", "C2"):
        raise ValueError("observe_port must be either 'C1' or 'C2'.")

    selected_power = arrays[observe_port]
    selected_db = 10.0 * np.log10(np.maximum(selected_power, 1e-15))

    return {
        "lambda_nm": lambda_nm,
        "clean_power": selected_power,
        "clean_db": selected_db,
        "all_clean_power": {"C1": arrays["C1"], "C2": arrays["C2"]},
        "all_clean_db": {"C1": arrays["CdB1"], "C2": arrays["CdB2"]},
        "observe_port": observe_port,
        "control_params": {k: params[k] for k in CONTROL_PARAM_NAMES},
        "fixed_params": {k: params[k] for k in DEFAULT_FIXED_PARAMS.keys()},
    }


def add_noise_to_power_spectrum(
    clean_power,
    rng=None,
    relative_noise_std=0.01,
    additive_noise_std=1e-4,
    baseline_offset=0.0,
    power_floor=1e-12,
):
    """在线性功率域加入测量噪声。

    用法：
    1. 输入 `clean_power` 为无噪功率数组。
    2. 通过 `relative_noise_std`、`additive_noise_std` 和 `baseline_offset` 控制噪声。
    3. 返回 `(noisy_power, noisy_db)`，并使用 `power_floor` 防止对数下溢。
    """
    if rng is None:
        rng = np.random.default_rng()

    clean_power = np.asarray(clean_power, dtype=float)
    relative_term = relative_noise_std * rng.standard_normal(clean_power.shape)
    additive_term = additive_noise_std * rng.standard_normal(clean_power.shape)

    noisy_power = clean_power * (1.0 + relative_term) + additive_term + baseline_offset
    noisy_power = np.maximum(noisy_power, power_floor)
    noisy_db = 10.0 * np.log10(noisy_power)
    return noisy_power, noisy_db


def simulate_noisy_spectrum(
    control_params=None,
    fixed_params=None,
    observe_port="C2",
    wavelength_window_nm=(1549.9, 1550.2),
    relative_noise_std=0.01,
    additive_noise_std=1e-4,
    baseline_offset=0.0,
    seed=None,
):
    """基于控制参数仿真带噪声光谱。

    用法：
    1. 先内部调用 `simulate_clean_spectrum` 生成理想光谱。
    2. 再调用 `add_noise_to_power_spectrum` 注入噪声。
    3. 返回含 clean/noisy 数据和噪声配置的结果字典。
    """
    rng = np.random.default_rng(seed)
    clean = simulate_clean_spectrum(
        control_params=control_params,
        fixed_params=fixed_params,
        observe_port=observe_port,
        wavelength_window_nm=wavelength_window_nm,
    )
    noisy_power, noisy_db = add_noise_to_power_spectrum(
        clean["clean_power"],
        rng=rng,
        relative_noise_std=relative_noise_std,
        additive_noise_std=additive_noise_std,
        baseline_offset=baseline_offset,
    )

    return {
        "lambda_nm": clean["lambda_nm"],
        "clean_power": clean["clean_power"],
        "clean_db": clean["clean_db"],
        "noisy_power": noisy_power,
        "noisy_db": noisy_db,
        "observe_port": observe_port,
        "control_params": clean["control_params"],
        "fixed_params": clean["fixed_params"],
        "noise_config": {
            "relative_noise_std": relative_noise_std,
            "additive_noise_std": additive_noise_std,
            "baseline_offset": baseline_offset,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    demo = simulate_noisy_spectrum(seed=42)
    print("observe_port:", demo["observe_port"])
    print("wavelength range (nm):", float(demo["lambda_nm"].min()), float(demo["lambda_nm"].max()))
    print("number of samples:", demo["lambda_nm"].size)
    print("clean power range:", float(demo["clean_power"].min()), float(demo["clean_power"].max()))
    print("noisy power range:", float(demo["noisy_power"].min()), float(demo["noisy_power"].max()))
