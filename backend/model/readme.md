# `lumen2/backend/model`

这个目录定义了一套统一的“YAML 配置 -> MATLAB 模型 -> Python 结果”仿真协议。

## 目录职责

- `MATLAB/`
  - 放置 MATLAB 光子模型入口函数。
- `YAML/`
  - 放置模型运行配置。
- `simulate.py`
  - 读取 `.yml/.yaml` 配置。
  - 启动 MATLAB Engine for Python。
  - 调用配置里指定的 MATLAB 模型入口。
  - 返回统一结构的仿真结果。

## `simulate.py` 的使用方式

作为模块：

```python
from backend.model.simulate import simulate_from_config

result = simulate_from_config("backend/model/YAML/ramzi.yml")
print(result["wavelength_nm"].shape)
print(result["parameters"]["constraint_status"]["satisfied"])
```

作为脚本：

```bash
python backend/model/simulate.py backend/model/YAML/ramzi.yml
python backend/model/simulate.py backend/model/YAML/ramzi.yml --save-npz ramzi_output.npz
```

## YAML 协议

每个配置文件至少包含下面几部分：

```yaml
version: 1

model:
  name: ramzi
  entrypoint: ramzi
  matlab_dir: backend/model/MATLAB

simulation:
  wavelength_nm:
    start: 1549.9230
    stop: 1550.2032
    step: 0.0002

outputs:
  observe_port: C2

parameters:
  tunable:
    thetai:
      value: 1.5707963267948966
      bounds: [0.0, 6.283185307179586]
    thetao:
      value: 1.5707963267948966
      bounds: [0.0, 6.283185307179586]

  constraints:
    - name: shared_outer_phase
      lhs: thetai
      operator: "=="
      rhs: thetao
    - name: balanced_ring_bias
      lhs: "fait + faib"
      operator: "=="
      rhs: 0.0
    - name: ring_phase_order
      lhs: fai2
      operator: "<="
      rhs: fai1

  fixed:
    Ki: 0.5
    E1:
      real: 0.0
      imag: 0.0
```

### 字段约定

- `model.name`
  - 模型名，仅用于标识。
- `model.entrypoint`
  - MATLAB 顶层函数名。
- `model.matlab_dir`
  - MATLAB 模型所在目录，可以写相对路径。
- `simulation.wavelength_nm`
  - 波长范围和采样间隔，单位 `nm`。
- `outputs.observe_port`
  - 需要观察的输出端口名。
- `parameters.tunable`
  - 可调参数定义。
  - 每个参数至少包含 `value`。
  - `bounds` 可选，格式固定为 `[lower, upper]`。
- `parameters.fixed`
  - 不可调参数。
  - 标量直接写数值；复数写成 `{real: ..., imag: ...}`。
- `parameters.constraints`
  - 可调参数之间的约束列表。
  - 每条约束包含 `lhs`、`operator`、`rhs`。
  - `operator` 目前支持 `==`、`<=`、`>=`。

## 约束定义方式

约束是对 `parameters.tunable` 中已声明参数的标量关系描述。

### 支持的表达式

- 数值字面量，例如 `0.0`、`3.141592653589793`
- 已声明的 tunable 参数名，例如 `thetai`
- 括号
- 运算符：`+`、`-`、`*`、`/`、`**`

### 约束示例

```yaml
parameters:
  tunable:
    a:
      value: 1.0
    b:
      value: 1.0
    c:
      value: 2.0

  constraints:
    - lhs: a
      operator: "=="
      rhs: b
    - lhs: "a + b"
      operator: "<="
      rhs: c
    - lhs: "2 * c"
      operator: ">="
      rhs: "a + b"
```

### 约束规则

- 约束表达式只能引用 `parameters.tunable` 中已经声明的参数。
- 每条约束至少要引用一个 tunable 参数。
- 约束名称 `name` 可选；若省略，加载时会自动生成 `constraint_1`、`constraint_2` 这类名字。
- Python 侧会在加载 YAML 时校验表达式语法和参数引用。

## MATLAB 模型函数协议

每个 MATLAB 模型都必须导出 YAML 中 `model.entrypoint` 指定的顶层函数。推荐“文件名 = 函数名”。

函数签名：

```matlab
function result = model_entrypoint(config)
```

其中：

- `config` 可以是 MATLAB `struct`，也可以是 Python 传入的 JSON 字符串。
- 模型函数至少需要读取：
  - `simulation.wavelength_nm.start`
  - `simulation.wavelength_nm.stop`
  - `simulation.wavelength_nm.step`
  - `outputs.observe_port`
  - `parameters.tunable.<name>.value`
  - `parameters.fixed.<name>`

返回值 `result` 必须是 MATLAB `struct`，并至少包含：

- `model_name`
- `wavelength_nm`
- `frequency_hz`
- `port_name`
- `complex_response`
- `power_linear`
- `power_db`

推荐额外返回：

- `all_complex_response`
- `all_power_linear`
- `all_power_db`
- `parameters`
- `simulation`

## Python 返回结果中的参数信息

`simulate.py` 返回的 `result["parameters"]` 现在会同时保留数值和元数据：

- `tunable`
  - 当前实际用于仿真的 tunable 数值。
- `fixed`
  - 当前 fixed 数值。
- `tunable_specs`
  - 原始 tunable 定义，包含 `value` 和可选 `bounds`。
- `constraints`
  - 规范化后的约束列表。
- `constraint_status`
  - 当前 tunable 数值对约束的检查结果。

`constraint_status` 的主要字段：

- `satisfied`
  - 当前参数是否满足全部约束。
- `evaluations`
  - 每条约束的求值结果。
- `violations`
  - 当前不满足的约束子集。

## MATLAB Engine 启动说明

- 项目默认优先使用 `matlab.engine.start_matlab()`。
- 在 `Windows + MATLAB R2025a Update 1 + 非 UTF-8 系统代码页` 环境下，直接启动可能抛出 `UnicodeDecodeError`。
- 当前仓库已内置 fallback：
  - 如果直接启动失败，会改为外部拉起一个共享 MATLAB 会话，再由 Python `connect_matlab()` 连接。
- 如果 `matlab.exe` 不在 `PATH` 中，请设置环境变量 `MATLAB_EXE` 指向可执行文件，例如：

```powershell
$env:MATLAB_EXE = 'D:\Program Files\MATLAB\R2025a\bin\matlab.exe'
```

## 当前示例

- 标准化后的 RAMZI MATLAB 模型：`MATLAB/ramzi.m`
- 对应 YAML 配置：`YAML/ramzi.yml`

后续增加新模型时，只要遵守上面的 YAML 协议和 MATLAB 返回协议，`simulate.py` 就可以直接复用。
