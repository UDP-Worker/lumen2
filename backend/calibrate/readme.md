`calibrate_model.py` 将对齐可调节参数与消光比之间的关系。

`calibrate_model.py` 将逐一对齐每一个可调节参数与消光比的关系。当对齐某一个可调节参数时，`calibrate_model.py` 将从模型运行配置文件`.yml`中读取“零配置”，并将除了目前正在对齐的可调节参数以外的其它可调节参数设置为“零配置”中的值。（“零配置”即指这些相位设置将不影响光芯片的响应，将处于全通状态的参数设置）

接着，`calibrate_model.py` 将不断改变正在对齐的那个参数运行MATLAB模型，记录正在对齐的参数的不同取值下的复振幅响应曲线，并调用`utils`目录中的脚本，显示一个可视化的窗口，展示这些曲线，允许用户指定求解消光比的波长和基准。一旦用户完成指定，`calibrate_model.py`将记录正在对齐的参数的不同取值与消光比的对应关系。完成后，存储在以对应模型命名的数据目录下。

`calibrate_control.py`将逐一对齐控制电压与消光比之间的关系。在对齐每一个电压前，程序需要进行初始化，以使得目前并未在对齐中的其它电压处于“零配置”的情况。这一初始化过程首先将略微改变各个未在对齐的电压，并从OSA中读取不同未在对齐的电压的不同取值的复振幅响应，并同时调用`utils`目录中的脚本，显示一个可视化的窗口，展示这些曲线，并要求用户指定各通道电压将各自改变哪一个位置的波长，以及消光比基准。一旦用户完成指定，`calibrate_control.py`将逐一改变未在对齐的电压，并在用户指定的、该未在对齐的电压将会影响的波长处计算消光比，并改变此未在对齐的电压，使得此处的消光比接近基准，也即“零配置”电压。当每一个未在对齐的电压都被置于“零配置”电压时，初始化过程完成，这些未在对齐的电压将不再影响复振幅响应了。该程序还将记录这些“零配置”电压、将要影响的波长，以便对齐下一个电压时复用。完成初始化后，`calibrate_control.py`将改变正在对齐的电压，并记录不同取值的正在对齐的电压的复振幅曲线，并在该正在对齐的电压将要影响的波长处（如果没有该信息，再次调用一次`utils`目录中的脚本，显示一个可视化的窗口，展示曲线，要求用户指定计算消光比的波长位置和基准），计算正在对齐的电压的不同取值对消光比的影响。完成后，存储在以对应模型命名的数据目录下。

`calibrate.py` 将利用上述两程序的结果，利用消光比作为桥梁，得到每一个通道电压到对应的可调节参数的映射关系。完成后，存储在以对应模型命名的数据目录下。

## 代码说明

### `calibrate_model.py`

- `run_model_calibration(...)` 是主入口：读取 YAML、解析模型名和输出目录、解析零配置、逐个参数做扫描，并最终写出 `model_calibration.json` 和曲线归档。
- `_resolve_parameter_sweep_values(...)` 负责决定每个 tunable 参数怎么扫：优先用 `calibration.model_sweep.values.<参数名>`，否则退回到 `num_samples + bounds` 或默认跨度。
- `_simulate_parameter_sweep(...)` 负责真正调用 MATLAB 模型：每次只改一个 tunable 参数，记录同一波长网格上的复响应和功率曲线，整理成 `CurveSweepBundle`。
- `select_extinction_reference(...)` 的调用是人工选点环节：用户在可视化窗口里选一个波长和 baseline，随后代码把各扫描点在该波长处的功率转换成消光比并落盘。

### `calibrate_control.py`

- `run_control_calibration(...)` 是主入口：读取控制校准配置、连接电源、先做初始化零点搜索，再做每个通道的正式扫描，最后输出 `control_calibration.json` 和两份曲线归档。
- `_resolve_settings(...)` 负责把 YAML 和 CLI 参数合并成一套控制参数，包括通道列表、`com_port`、电流电压限制、OSA 参数、初始化偏移量和正式标定偏移量。
- 初始化阶段分两步：先用 `_capture_spectrum_bundle(...)` 对每个通道做小范围扫压并采整段光谱，再用 `select_variable_targets(...)` 让用户给每个通道选目标波长和 baseline。
- 选点之后，代码用 `_measure_target_powers(...)` 在目标波长处重新采样，并用 `estimate_zero_crossing(...)` 估计每个通道的“零配置”电压；这些结果写进 `zero_voltages` 和 `initialization.zeroing_history`。
- 正式标定阶段再次扫每个通道，复用前面记录的目标波长和 baseline，生成“电压值 -> 消光比”的关系并写入 `results`。

### `calibrate.py`

- `run_voltage_to_tunable_calibration(...)` 是映射求解入口：读取 `model_calibration.json` 和 `control_calibration.json`，把每个通道和每个 tunable 参数的消光比曲线拿出来两两比较。
- `_compare_calibration_curves(...)` 先找两条消光比曲线的重叠区间，再把它们插值到同一组消光比采样点上，比较“同向”和“反向”两种关系的误差。
- `_linear_sum_assignment(...)` 负责全局匹配：优先用 SciPy 的匈牙利算法，没有 SciPy 时退回一个纯 Python 解法。
- 对每个匹配成功的通道，代码会输出 `parameter_name`、匹配方向、匹配代价，以及一组 `voltage_grid -> parameter_grid` 采样点，供后续拟合或查表使用。

### `_shared.py`

- 这个文件放公共能力：YAML 读取、输出目录管理、曲线打包、消光比计算、零点估计、JSON/NPZ 写盘等。
- 三个主脚本的共通数据结构 `CurveSweepBundle` 也在这里，方便统一地把曲线传给可视化组件和序列化逻辑。

## `calibrate_model.py` 具体用法

### 前提

- 运行正式模型标定前，配置文件中必须已经有 `calibration.zero_config.tunable`。
- 如果配置文件里还没有零配置，先运行零配置编辑器：

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml --edit-zero-config
```

### 零配置编辑器做什么

- 它会打开一个可视化窗口，并用当前 tunable 参数值运行 MATLAB 模型。
- 窗口左侧会列出所有 tunable 参数；有 `bounds` 的参数会显示滑条，没有 `bounds` 的参数可以直接改输入框。
- 点击 `Simulate` 后，右侧会刷新当前参数组合对应的复响应幅值曲线（以 `20*log10(|E|)` 显示）。
- 你可以反复调参，找到你认为对应“全通 / 零配置”的参数组合后，点击 `Save Zero Config`。
- 保存后，程序会把当前 tunable 参数值写回到模型 YAML 的 `calibration.zero_config.tunable` 段。

### 正式标定命令

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml
```

常用可选参数：

- `--parameters fai1 fai2 ...`：只标定指定的 tunable 参数子集。
- `--num-samples 11`：覆盖默认扫描点数。
- `--output-dir path/to/output`：把结果写到自定义目录；不指定时默认写到 `backend/calibrate/data/<model_name>/`。

示例：

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml --parameters fai1 theta1 --num-samples 13
```

### 正式标定时程序会做什么

1. 读取 `calibration.zero_config.tunable`，把所有 tunable 参数先放到零配置。
2. 对于当前正在标定的那个参数，按扫描点逐个改值并运行 MATLAB 模型。
3. 把这一组扫描得到的复响应曲线送进可视化窗口。
4. 用户在窗口里选定一个用于计算消光比的波长和 baseline。
5. 程序把每个扫描点在该波长处的功率换算成消光比，形成“参数值 -> 消光比”的标定结果。

### 输出结果

- `model_calibration.json`：主结果文件，记录每个 tunable 参数的目标波长、baseline、零配置值、扫描点、消光比结果等。
- `model_calibration_curves.npz`：原始曲线归档，保存每个参数扫描时的波长、功率和复响应数组。

### 注意事项

- `--edit-zero-config` 是一个单独的交互步骤；带上这个参数时，程序会进入零配置编辑器并在关闭后结束，不会继续执行正式标定。
- 如果配置文件中缺少 `calibration.zero_config.tunable`，正式标定会直接报错，并提示先运行 `--edit-zero-config`。
- 这套流程依赖 MATLAB 引擎和图形界面环境；远程无桌面环境下不能直接使用交互窗口。
