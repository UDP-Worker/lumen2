`calibrate_model.py`、`calibrate_control.py`、`calibrate.py` 现在的目标关系如下：

- `calibrate_model.py` 负责建立“可调参数 -> 曲线形状”的标定数据。
- `calibrate_control.py` 负责建立“控制电压 -> 曲线形状”的标定数据。
- `calibrate.py` 后续将把这两份曲线数据作为桥梁，结合用户提供的“通道影响哪个可调参数”的先验关系，求出“通道电压 -> 可调参数”的映射。

这套新方案不再使用消光比作为中间桥梁。

#### `calibrate.py`技术建议

拟合
$$
T_{\text{real}}(\lambda)\approx a\,T_{\text{model}}(\lambda+\delta; p)+b,
$$
其中 $p$ 是你要找的模型参数，$a$ 用来吸收整体损耗或增益缩放，$b$ 用来吸收基线泄漏或探测器零漂，$\delta$ 用来吸收少量波长标定偏差。这样你比较的是“在允许这些现实误差之后，两条曲线是否属于同一形状族”。这个思路在工程上通常比直接比消光比稳得多，因为完整曲线包含的信息远多于单个标量指标。用于做这种带边界、带鲁棒损失的非线性最小二乘，SciPy（Scientific Python，科学计算 Python 库）里的 `scipy.optimize.least_squares` 很合适，它原生支持变量边界和多种鲁棒损失函数；如果你后面要把离散扫描结果变成平滑单调映射，`scipy.interpolate.PchipInterpolator` 是保形的单调三次插值器，适合做一维标定曲线。

优先考虑一套很朴素但很够用的工具链：`numpy + scipy + scikit-learn`。其中 `SciPy` 负责预处理、插值、相似度计算和参数优化；`scikit-learn` 里的 `IsotonicRegression` 可以在你已经得到一串离散对应点 $(v_i,p_i)$ 之后，拟合一个“不预设具体函数形式、但强制单调”的映射，这特别适合你这种“已知一一对应、但未必线性”的电压到相位标定。`IsotonicRegression` 拟合的正是单调非减的一维函数，而 `SciPy` 近年的 `scipy.optimize.isotonic_regression` 也提供了同类的保序回归能力。对你的场景来说，这比上来就拟合高阶多项式更稳，因为它不会为了追几个噪声点而在中间乱摆。

真正做曲线匹配时，我建议你先不要直接上动态时间规整。你现在的横轴是波长，不是“可任意伸缩的时间序列”，所以如果物理上只是存在很小的波长轴偏移，那么先用插值统一到同一波长网格，再用一个小范围的 $\delta$ 去优化，通常更符合物理意义。`scipy.interpolate` 提供了比较完整的一维插值工具，`PCHIP` 的好处是尽量保持形状，不容易像普通三次样条那样在峰谷附近产生过冲。只有当你后面发现真实曲线相对仿真曲线不仅仅是整体平移，而是存在局部横向拉伸压缩时，才考虑用 tslearn 里的 DTW（Dynamic Time Warping，动态时间规整）一类方法；它确实是成熟库，但它允许沿横轴寻找最优配对路径，这种自由度对很多谱线问题来说有点“过于灵活”，容易把本来不该对上的结构也硬对上，所以更适合作为辅助诊断工具，而不是主标定方法。

从实现角度，我觉得你可以把 `calibrate.py` 里的核心步骤设成下面这个形式。先对每条真实曲线做预处理，例如去异常点、轻微平滑、统一波长采样；然后对每个候选模型参数 $p$，求解
$$
\min_{a,b,\delta}\ \rho\!\left(T_{\text{real}}(\lambda)-\big[a\,T_{\text{model}}(\lambda+\delta;p)+b\big]\right),
$$
这里 $\rho$ 可以对应 `least_squares` 的鲁棒损失，比如 soft-l1 或 Huber 型思路，用来减弱局部噪声和坏点的影响。对每个真实电压 $v$，你最后会得到一个最优参数 $p^\*(v)$。如果这串 $p^\*(v)$ 基本平滑且单调，再用 `IsotonicRegression` 或者先做保序回归、再用 `PCHIP` 平滑成连续映射，就能得到你要的 $p=h(v)$。这里有一个非常实用的小技巧：除了残差平方和，你还可以同时记录相关系数，作为辅助指标。`scipy.stats.pearsonr` 给出线性相关系数，`scipy.signal.correlate` 则可以用来粗估横向偏移量；前者可以帮你判断“形状是否像”，后者可以帮你快速给 $\delta$ 一个初值。

## 当前实现状态

- `calibrate_model.py` 已重构为纯曲线扫描流程。
- `calibrate_control.py` 已重构为纯曲线扫描流程。
- 零配置编辑器仍然保留，用于写入 `calibration.zero_config.tunable`。
- `calibrate.py` 已重构为“曲线形状匹配 + 单调映射求解”流程。

## 代码说明

### `calibrate_model.py`

- `run_model_calibration(...)` 是模型侧标定入口。它会读取 YAML、校验 `calibration.zero_config.tunable`、把所有 tunable 先放到零配置，然后对每个待标参数单独扫描。
- `_resolve_parameter_sweep_values(...)` 负责决定某个参数的扫描取值。优先读取 `calibration.model_sweep.values.<parameter>`，否则使用 `num_samples + bounds` 自动生成扫描点。
- `_simulate_parameter_sweep(...)` 负责实际调用 MATLAB 模型。每次只改变一个 tunable 参数，其它 tunable 固定为零配置，最后把所有曲线整理成 `CurveSweepBundle`。
- `edit_model_zero_config(...)` 是零配置编辑器入口。它提供一个交互窗口，用当前 tunable 参数组合运行模型、显示曲线，并把用户确认后的零配置写回 YAML。
- 正式运行模型标定时，不再弹出逐曲线选点窗口；程序只负责扫参数、记录曲线、写结果文件。

### `calibrate_control.py`

- `run_control_calibration(...)` 是控制侧标定入口。它会读取通道列表、零电压、OSA 参数和电源限制，连接电源后逐通道扫压并采集整段光谱。
- `_resolve_settings(...)` 负责把 YAML 和 CLI 参数合并成最终配置，并强制校验：
  - 每个通道都必须有零电压；
  - `vmax`、`imax` 必须显式给出；
  - `vmax`、`imax` 必须为正；
  - 任何零电压或扫压值都不能超过 `|vmax|`。
- `_capture_spectrum_bundle(...)` 负责对单个通道执行一次完整扫描：其它通道保持在零电压，当前通道依次取多个电压值，逐条读取 OSA 曲线并打包。
- 正式运行控制标定时，也不再弹出选点窗口；程序只负责扫电压、记录曲线、写结果文件。

### `_shared.py`

- `CurveSweepBundle` 是三条脚本共用的曲线数据结构，统一保存扫描值、波长轴、功率矩阵和复响应矩阵。
- `save_curve_archives(...)` 负责把一组 `CurveSweepBundle` 写成 `.npz` 曲线归档。
- 其它公共能力包括 YAML 读写、输出目录管理、JSON 写盘、零配置读写等。

### `calibrate.py`

- `run_voltage_to_tunable_calibration(...)` 是映射求解入口。它会读取模型侧和控制侧的曲线归档，并读取用户提供的“channel -> parameter”关系。
- 对每个通道-参数配对，代码会逐条比较控制曲线与模型曲线，拟合
  `T_real(lambda) ~= a * T_model(lambda + delta) + b`
  中的 `a`、`b`、`delta`，用拟合残差和相关系数衡量形状是否匹配。
- 在每个通道内部，代码会对“电压排序后的控制曲线”和“参数排序后的模型曲线”做单调路径搜索，确保得到的离散对应关系整体上是单调的。
- 得到离散对应点后，代码再做带权保序回归，并用 `PCHIP` 生成连续的 `voltage -> parameter` 映射采样表。

## `calibrate_model.py` 用法

### 前提

- 正式运行模型侧标定前，配置文件中必须已经存在 `calibration.zero_config.tunable`。
- 如果配置文件里还没有零配置，先启动零配置编辑器：

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml --edit-zero-config
```

### 零配置编辑器做什么

- 它会打开一个交互窗口，并用当前 tunable 参数值运行 MATLAB 模型。
- 左侧列出所有 tunable 参数；右侧显示当前参数组合对应的曲线。
- 你可以反复调整参数，找到“其它调节器都不会显著影响芯片响应”的那组参数值。
- 点击 `Save Zero Config` 后，程序会把当前 tunable 参数值写回 YAML 的 `calibration.zero_config.tunable`。

### 正式标定命令

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml
```

常用参数：

- `--parameters fai1 theta1`：只扫描指定的 tunable 参数子集。
- `--num-samples 13`：覆盖默认扫描点数。
- `--output-dir path/to/output`：把结果写到自定义目录。

示例：

```bash
python -m backend.calibrate.calibrate_model backend/model/YAML/ramzi.yml --parameters fai1 theta1 --num-samples 13
```

### 正式标定时程序会做什么

1. 读取 `calibration.zero_config.tunable`。
2. 把所有 tunable 参数先放到零配置。
3. 对每个待标参数单独扫描：
   当前参数按扫描点逐个取值；
   其它 tunable 参数始终固定在零配置。
4. 对每个扫描点调用 MATLAB 模型，记录整条曲线。
5. 把结果写入模型对应的数据目录。

### 输出结果

- `model_calibration.json`
  - 记录模型名、配置路径、零配置、参数顺序以及每个参数对应的曲线归档前缀、扫描值列表和曲线摘要。
- `model_calibration_curves.npz`
  - 保存每个参数扫描得到的完整波长轴、功率曲线和复响应曲线。
- `model_calibration.log`
  - 记录模型侧曲线扫描过程。
- `zero_config_editor.log`
  - 记录零配置编辑器的交互和仿真过程。

### 结果语义

- `model_calibration.json` 中的 `results.<parameter>.sweep_values` 表示该参数本次实际扫描的参数值。
- `results.<parameter>.curve_archive_prefix` 用于在 `model_calibration_curves.npz` 中定位这一组曲线。
- `results.<parameter>.curve_summary` 给出该组曲线的波长范围、采样点数等摘要信息。

## `calibrate_control.py` 用法

### 前提

- 用户必须明确给出每个通道的“零电压”，即芯片处于全通/不显著受该通道影响时的基准电压。
- 必须明确给出电源限制 `vmax` 和 `imax`；没有这两个值时程序不会运行。

可以写在 YAML 里，例如：

```yaml
calibration:
  control:
    channels: [1, 2, 3, 4]
    zero_voltages:
      1: 0.0
      2: 0.0
      3: 0.0
      4: 0.0
    calibration_offsets: [-0.4, -0.2, 0.0, 0.2, 0.4]
    voltage_source:
      com_port: 3
      vmax: 6.0
      imax: 0.02
```

也可以通过 CLI 覆盖其中一部分参数。

### 正式标定命令

```bash
python -m backend.calibrate.calibrate_control backend/model/YAML/ramzi.yml
```

常用参数：

- `--channels 1 2 3 4`
- `--com-port 3`
- `--vmax 6.0`
- `--imax 0.02`
- `--zero-voltages 1=0.0 2=0.0 3=0.0 4=0.0`
- `--calibration-offsets -0.4 -0.2 0.0 0.2 0.4`
- `--output-dir path/to/output`

示例：

```bash
python -m backend.calibrate.calibrate_control backend/model/YAML/ramzi.yml --channels 1 2 --com-port 3 --vmax 6.0 --imax 0.02 --zero-voltages 1=0.0 2=0.1
```

### 正式标定时程序会做什么

1. 解析通道列表、零电压、OSA 参数和电源限制。
2. 校验每个通道都存在零电压，且所有待设置电压都不会超过 `|vmax|`。
3. 连接电源并配置通道限制。
4. 先把所有通道设置到零电压。
5. 对每个通道单独扫描：
   当前通道取 `zero_voltage + calibration_offsets`；
   其它通道始终保持在零电压。
6. 每次改压后等待稳定，再读取整段 OSA 曲线。
7. 把结果写入模型对应的数据目录。

### 输出结果

- `control_calibration.json`
  - 记录模型名、配置路径、通道顺序、零电压、扫描偏移、电源限制、OSA 参数以及每个通道的曲线摘要。
- `control_calibration_curves.npz`
  - 保存每个通道扫描得到的完整波长轴和功率曲线。
- `control_calibration.log`
  - 记录控制侧曲线扫描过程。

### 结果语义

- `control_calibration.json` 中的 `results.<channel>.zero_voltage` 表示该通道的基准零电压。
- `results.<channel>.sweep_values` 表示该通道本次实际施加的电压值。
- `results.<channel>.curve_archive_prefix` 用于在 `control_calibration_curves.npz` 中定位这一组曲线。

## `calibrate.py` 用法

### 前提

- 必须先已经运行过 `calibrate_model.py` 和 `calibrate_control.py`，得到各自的 `.json` 和 `.npz` 文件。
- 必须明确给出“哪个通道影响哪个可调参数”。可以写在 YAML 里：

```yaml
calibration:
  voltage_to_tunable:
    channel_to_parameter:
      1: fai1
      2: theta1
```

- 也可以通过 CLI 覆盖：

```bash
python -m backend.calibrate.calibrate backend/model/YAML/ramzi.yml --channel-parameter-map 1=fai1 2=theta1
```

### 正式求解命令

```bash
python -m backend.calibrate.calibrate backend/model/YAML/ramzi.yml
```

常用参数：

- `--model-calibration path/to/model_calibration.json`
- `--control-calibration path/to/control_calibration.json`
- `--channel-parameter-map 1=fai1 2=theta1`
- `--max-delta-nm 0.05`
- `--smooth-window 7`
- `--acceptance-cost 0.2`
- `--min-correlation 0.85`
- `--mapping-grid-points 256`

### 程序会做什么

1. 读取模型侧和控制侧的曲线结果。
2. 对用户指定的每个通道-参数配对，逐条比较控制曲线和模型曲线的形状。
3. 对每一对候选曲线拟合 `a`、`b`、`delta`，记录拟合误差、相关系数和波长偏移。
4. 在每个通道内部，求一条整体单调的离散匹配路径。
5. 对离散对应点做保序回归和平滑插值，得到最终的 `voltage -> parameter` 映射表。

### 输出结果

- `voltage_to_tunable_mapping.json`
  - 记录每个通道对应的参数名、匹配方向、离散匹配点、保序后的参数值、连续映射采样表以及逐曲线的拟合诊断信息。
