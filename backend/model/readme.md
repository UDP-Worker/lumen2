# `lumen2/backend/model`

这个目录定义了一套统一的“YAML 配置 -> MATLAB 模型 -> Python 结果”仿真协议。

## 目录职责

- `MATLAB/`
  - 存放 MATLAB 光芯片模型入口函数。
- `YAML/`
  - 存放模型运行配置。
- `simulate.py`
  - 读取 `.yml/.yaml` 配置。
  - 启动 MATLAB Engine for Python。
  - 调用配置中指定的 MATLAB 模型入口。
  - 返回统一结构的复振幅响应和能量响应。

## `simulate.py` 的使用方式

作为模块：

```python
from backend.model.simulate import simulate_from_config

result = simulate_from_config("backend/model/YAML/ramzi.yml")
print(result["wavelength_nm"].shape)
print(result["power_db"].min(), result["power_db"].max())
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
  fixed:
    Ki: 0.5
    E1:
      real: 0.0
      imag: 0.0
```

字段约定：

- `model.name`
  - 模型名称，仅用于标识。
- `model.entrypoint`
  - MATLAB 顶层函数名。
- `model.matlab_dir`
  - MATLAB 模型所在目录。可以写相对路径。
- `simulation.wavelength_nm`
  - 波长范围和采样间隔，单位 nm。
- `outputs.observe_port`
  - 需要观察的输出端口名。
- `parameters.tunable`
  - 可调参数。每个参数至少包含 `value`，建议提供 `bounds` 供后续优化器使用。
- `parameters.fixed`
  - 不可调参数。标量直接写数值；复数写成 `{real: ..., imag: ...}`。

## MATLAB 模型函数协议

每个 MATLAB 模型都必须导出 YAML 中 `model.entrypoint` 指定的顶层函数。推荐“文件名 = 函数名”。

函数签名：

```matlab
function result = model_entrypoint(config)
```

其中：

- `config` 可以是 MATLAB `struct`，也可以是由 Python 传入的 JSON 字符串。
- 模型函数必须读取：
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

## MATLAB Engine 启动说明

- 项目默认使用 `matlab.engine.start_matlab()`。
- 在 Windows + MATLAB R2025a Update 1 + 非 UTF-8 系统代码页环境下，`start_matlab()` 可能抛出 `UnicodeDecodeError`。
- 当前仓库已经内置 fallback：如果直接启动失败，会自动改为“外部拉起共享 MATLAB 会话，再由 Python `connect_matlab()` 连接”。
- 如果 `matlab.exe` 不在 `PATH` 中，请设置环境变量 `MATLAB_EXE` 指向可执行文件，例如：

```powershell
$env:MATLAB_EXE = 'D:\Program Files\MATLAB\R2025a\bin\matlab.exe'
```

## 当前示例

- 标准化后的 RAMZI MATLAB 模型：`MATLAB/ramzi.m`
- 对应 YAML 配置：`YAML/ramzi.yml`

后续增加新模型时，只要遵守上面的 YAML 协议和 MATLAB 返回协议，`simulate.py` 就可以直接复用。
