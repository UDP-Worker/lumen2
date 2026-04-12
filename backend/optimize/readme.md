`optimize`模块在仿真中优化滤波器的可调参数，使其符合用户的预期。

具体的，它将和`calibrate`模块中的方法类似，接受一个YAML文件，其中指定了光子滤波器的MATLAB模型，优化YAML文件中指定的，可以优化的参数，以达到用户理想的波形。

#### 优化目标

我们的目标函数是加权拟合。设光子滤波器的观测端口功率谱为 $T(\lambda;x)$，这里 $x$ 就是所有可调参数，比如 $\theta_1\sim\theta_4,\ \phi_1\sim\phi_4$。然后你指定一个目标带通窗口 $[\lambda_a,\lambda_b]$，在这个窗口内希望透过率接近 1，在窗口外希望透过率接近 0。于是你可以定义一个目标模板 $T^\star(\lambda)$：通带里取 1，阻带里取 0，中间过渡带暂时不管。这样你的损失函数最简单可以写成


$$
L(x)=
w_p\,\mathrm{mean}_{B_p}(1-T)^2
+
w_s\,\mathrm{mean}_{B_s}T^2
+
w_r\,\mathrm{var}_{B_p}(T)
+
w_c(\hat\lambda_c-\lambda_0)^2.
$$
其中 $B_p$ 是 passband（通带），$B_s$ 是 stopband（阻带），$w_p,w_s$ 是权重。这个写法的好处是直观，而且非常容易调。你觉得阻带压得不够深，就把 $w_s$ 调大；你觉得通带插损太大，就把 $w_p$ 调大。

但只靠这个还不够，因为它没有显式要求“通带要平”。所以通常还要补一个 ripple（通带起伏）惩罚项。一个很自然的写法是
$$
L_{\text{ripple}}(x)
=
w_r\,\mathrm{var}_{\lambda\in B_p}\bigl(T(\lambda;x)\bigr),
$$
或者用通带内偏离均值的平方和。这样优化器就不会只追求“通带里某几个点很高”，而是更倾向于把整个通带抬平。你如果还想让通带中心对准某个目标波长 $\lambda_0$，还可以再加一个中心漂移惩罚。比如把通带光谱的重心 $\hat\lambda_c(x)$ 算出来，然后加上
$$
L_{\text{center}}(x)=w_c\bigl(\hat\lambda_c(x)-\lambda_0\bigr)^2.
$$
如果你已经对带宽也有明确要求，比如希望 3 dB 带宽接近某个值 $BW_0$，那就再加一个带宽误差项。这样一来，目标函数就不再是模糊的“越尖越深越好”，而是“在指定中心、指定宽度附近，做一个通带高、阻带低、起伏小的响应”。

请注意，第一个是“带通”本身不是一个单一数字，它至少包含了通带中心、通带宽度、通带内起伏、阻带泄漏、左右阻带是否对称这些要求。第二个是如果你只写“最大化峰值和谷值之差”，优化器很容易钻空子，给你找出一个非常尖、非常窄的谐振尖峰；它在数学上看似消光比很高，但工程上根本不是你想要的“带通滤波器”。所以比起直接写“最大化 ER（extinction ratio，消光比）”，更稳妥的思路是先定义一个目标光谱，再让仿真光谱去逼近它。

模型的YAML里面应该指定优化目标（不是整个目标函数，而是目标函数的关键参数），现在RAMZI还没有相应的定义，在开始前先写一个目标函数，用来优化：中心波长 1550 nm，3 dB 带宽 0.005 nm。

#### 优化方法

至于优化方法，这个问题的结构决定了它大概率是一个强非凸、多峰、带周期性的优化问题。原因很简单：你的很多参数本质上都是相位，$\theta$ 和 $\phi$ 通常都带 $2\pi$ 周期；而多个 ring 和 MZI（Mach-Zehnder interferometer，马赫—曾德尔干涉仪）主干相位叠起来之后，目标函数表面往往会有很多局部极值。对这种问题，如果只从一个初值出发做局部优化，常常会陷进一个不太好的解里。

所以最实用的路线通常是“先全局，后局部”。如果你用 Python，SciPy 的 `differential_evolution` 是一个很合适的第一步。它是随机的、无梯度的全局优化方法，擅长在较大的参数空间里搜索，但通常会比传统梯度法需要更多函数评估。SciPy 文档就是这么描述它的：它不依赖梯度，能搜索较大的候选空间，但往往需要更多仿真次数。

在 `differential_evolution` 找到一批较好的候选点之后，再拿最好的几个点去做局部精修，会比单独用任何一种方法都稳。局部阶段如果你的目标函数是光滑的、可以放心做数值梯度，那么 `L-BFGS-B` 很合适，因为它就是为带边界约束的标量函数最小化准备的。 如果你把问题写成“残差向量”的形式，也就是前面说的模板拟合，那么 `least_squares(method='trf')` 很适合做这一层精修，因为它支持参数边界，而且文档明确说明 `trf` 是一个总体上比较稳健、适合带 bounds 的方法。 如果你不太信任数值梯度，或者目标函数里用了分位数这类不太光滑的操作，那就可以把局部阶段换成 `Powell`，它是无导数的局部优化方法。

另外，你现在先别急着优化全 8 个自由度。我更建议你先做一个“对称版”的 4 参数优化，也就是先强制
$$
fai_1=fai_3,\quad fai_2=fai_4,\quad \theta_1=\theta_3,\quad \theta_2=\theta_4,
$$
先在这个低维子空间里看看，结构本身能不能比较自然地长出你想要的 1550 nm、5 pm 左右的窄带带通。如果 4 参数对称版已经能做出一个还不错的初解，再把自由度放开到 8 参数精修，通常会比一上来全 8 维乱搜更稳，也更容易解释结果。

还有一个很重要的现实问题是，你到底要优化线性功率还是 dB（decibel，分贝）谱。我的建议是，主体损失函数先在线性功率域里写，因为它更平滑，也更适合平方误差；而“消光比”这种动态范围指标，可以单独在 dB 域上作为辅助项。也就是说，通带模板匹配、阻带压低、通带平坦这些主要项用线性功率写；只有像 “阻带至少低于通带多少 dB” 这种约束，再在 dB 域补进去。这样数值上一般更稳定。

## 当前实现的用法

现在 `backend/optimize/optimize.py` 已经实现了一版可直接运行的优化流程。它会：

- 读取模型 YAML 和其中的 `optimization:` 配置；
- 调用 `backend.model.simulate` 对 MATLAB 模型做重复仿真；
- 按当前实现的带通损失函数优化 tunable 参数；
- 把优化历史、最优结果和可视化文件写到输出目录。

### 作为命令行脚本运行

最简单的用法：

```bash
python -m backend.optimize.optimize backend/model/YAML/ramzi.yml
```

常用参数：

- `--output-dir path/to/output`
  - 指定输出目录；如果不写，默认写到 `backend/optimize/data/<model>/<timestamp>/`。
- `--max-evaluations N`
  - 限制最大 MATLAB 仿真次数。
- `--global-maxiter N`
  - 覆盖全局优化 `differential_evolution` 的 `maxiter`。
- `--global-popsize N`
  - 覆盖全局优化种群大小。
- `--local-maxiter N`
  - 覆盖局部优化 `Powell` 的 `maxiter`。
- `--skip-global`
  - 跳过全局优化，只保留初值和局部精修。
- `--skip-local`
  - 跳过局部优化，只保留初值和全局阶段。
- `--tensorboard` / `--no-tensorboard`
  - 显式开启或关闭 TensorBoard 事件写入。
- `--plot-every N`
  - 每做多少次新评估就更新一次 `history.png`。
- `--seed N`
  - 覆盖全局优化随机种子。

对计算资源有限的机器，比较推荐先从下面这种命令开始：

```bash
python -m backend.optimize.optimize backend/model/YAML/ramzi.yml \
  --max-evaluations 12 \
  --global-maxiter 1 \
  --global-popsize 2 \
  --local-maxiter 8
```

如果你只想看当前初值附近能不能稍微修一修，可以进一步用：

```bash
python -m backend.optimize.optimize backend/model/YAML/ramzi.yml \
  --skip-global \
  --max-evaluations 10 \
  --local-maxiter 6
```

### 作为 Python 函数调用

```python
from backend.optimize import run_filter_optimization

summary = run_filter_optimization("backend/model/YAML/ramzi.yml")
print(summary["best"]["total_loss"])
print(summary["artifacts"]["optimized_config"])
```

返回值是一个 `dict`，其中会包含：

- 本次运行的 `run_dir`
- 优化目标和权重
- 全局阶段和局部阶段配置
- 当前最优参数、最优指标、损失项
- 生成的结果文件路径

## TensorBoard 和输出文件

### TensorBoard 怎么看

如果 YAML 或 CLI 中启用了 TensorBoard，程序会在本次 `run_dir` 下生成 `tensorboard/` 目录。可以这样启动：

```bash
uv run tensorboard --logdir backend/optimize/data
```

或者只看某一次运行：

```bash
uv run tensorboard --logdir backend/optimize/data/ramzi/<timestamp>
```

说明：

- 在 Windows 下，如果 `.venv/Scripts` 没有加到 `PATH`，直接输入 `tensorboard ...` 可能会提示命令不存在；这时优先使用 `uv run tensorboard ...`。
- 当前项目已经额外约束了 `setuptools<81`，因为 `tensorboard 2.20.0` 仍然依赖 `pkg_resources`。
- 启动时如果看到 `TensorFlow installation not found - running with reduced feature set.`，这是正常现象；我们这里只把 TensorBoard 当日志查看器，不依赖完整 TensorFlow。

当前会写入这些标量：

- `loss/total`
- `loss/passband`
- `loss/stopband`
- `loss/ripple`
- `loss/center`
- `loss/bandwidth`
- `loss/constraint`
- `metrics/*`
  - 包括峰值波长、估计中心波长、估计 3 dB 带宽、通带平均功率、阻带平均功率、阻带最大泄漏、通带 ripple、估计消光比等
- `decision/*`
  - 当前优化变量值
- `meta/elapsed_seconds`
  - 单次仿真耗时

### 每次运行会产出什么

默认输出目录中会有：

- `optimization_summary.json`
  - 本次优化的总摘要，最适合程序读。
- `best_metrics.json`
  - 当前最优点的详细指标。
- `optimized_config.yml`
  - 把最优 tunable 值写回后的 YAML。
- `best_result.npz`
  - 最优点对应的主仿真数组归档。
- `best_spectrum.png`
  - 最优点光谱图，带目标通带标注。
- `history.csv`
  - 每一步优化记录，适合表格查看。
- `history.jsonl`
  - 每一步优化记录，适合程序逐行处理。
- `history.png`
  - 优化过程曲线图。
- `tensorboard/`
  - TensorBoard event 文件，仅在启用时生成。

## `optimization:` YAML 怎么写

当前实现并不支持在 YAML 里自由写任意目标函数表达式；YAML 负责定义“目标函数的关键参数”和“优化器参数”。也就是说：

- 你可以定义目标中心、目标 3 dB 带宽、过渡带、损失权重、参数分组方式、优化轮数和日志行为；
- 你不能直接在 YAML 里写一个自定义 Python/数学表达式替换整个损失函数。

当前实现的损失函数实际上是：

$$
L =
w_p \cdot \mathrm{MSE}_{B_p}(1-T)
+
w_s \cdot \mathrm{MSE}_{B_s}(T)
+
w_r \cdot \mathrm{Var}_{B_p}(T)
+
w_c \cdot \left(\frac{\hat{\lambda}_c-\lambda_0}{BW_0}\right)^2
+
w_b \cdot \left(\frac{\widehat{BW}_{3\mathrm{dB}}-BW_0}{BW_0}\right)^2
+
w_k \cdot L_{\text{constraint}}.
$$

其中：

- `T` 使用线性功率谱；
- `B_p` 是目标通带；
- `B_s` 是去掉过渡带后的阻带；
- `\hat{\lambda}_c` 和 `\widehat{BW}_{3\mathrm{dB}}` 由当前仿真光谱自动估计；
- 当前实现还会额外加入峰值项和对比度项，用来抑制“全暗端口”和“全通端口”这类病态解；
- `L_constraint` 来自 `parameters.constraints` 的违反量。

### 推荐写法示例

```yaml
optimization:
  target:
    center_nm: 1550.0
    bandwidth_3db_nm: 0.005
    transition_nm: 0.005
    contrast_min_linear: 0.8

  weights:
    passband: 8.0
    stopband: 12.0
    ripple: 0.5
    center: 0.15
    bandwidth: 0.1
    peak: 10.0
    contrast: 50.0
    constraint: 100.0

  parameterization:
    mode: direct

  strategy:
    global:
      enabled: true
      maxiter: 1
      popsize: 2
      polish: false
      seed: 42
      tol: 0.01
      mutation: [0.5, 1.0]
      recombination: 0.7
    local:
      enabled: true
      method: Powell
      maxiter: 12
      xtol: 0.001
      ftol: 0.001

  max_evaluations: 24

  logging:
    tensorboard: true
    plot_every: 2
```

### 字段说明和默认值

#### `optimization.target`

- `center_nm`
  - 目标中心波长，单位 `nm`。
  - 默认值：`1550.0`
- `bandwidth_3db_nm`
  - 目标 3 dB 带宽，单位 `nm`。
  - 默认值：`0.005`
- `transition_nm`
  - 通带外侧留出的过渡带宽度，过渡带不计入阻带损失。
  - 默认值：`max(bandwidth_3db_nm / 2, 4 * simulation.wavelength_nm.step)`
- `contrast_min_linear`
  - 期望“峰值功率 - 阻带平均功率”至少达到的最小线性对比度。
  - 默认值：`0.8`

#### `optimization.weights`

- `passband`
  - 通带内接近 1 的权重。
  - 默认值：`4.0`
- `stopband`
  - 阻带内接近 0 的权重。
  - 默认值：`4.0`
- `ripple`
  - 通带平坦度权重。
  - 默认值：`0.5`
- `center`
  - 中心波长误差权重。
  - 默认值：`1.5`
- `bandwidth`
  - 3 dB 带宽误差权重。
  - 默认值：`1.5`
- `constraint`
  - 参数约束违反惩罚权重。
  - 默认值：`100.0`
- `peak`
  - 峰值功率不足时的惩罚权重，用来避免优化器把整个端口压成全暗。
  - 默认值：`10.0`
- `contrast`
  - 峰值与阻带平均功率对比度不足时的惩罚权重，用来避免“全暗”或“全通”病态解。
  - 默认值：`50.0`

#### `optimization.parameterization`

- `mode`
  - 参数化模式名称。
  - 当前实现里它主要用于说明模式；只有当 `mode: symmetric4` 且未显式写 `groups` 时，程序会自动采用：
    - `fai1 = fai3`
    - `fai2 = fai4`
    - `theta1 = theta3`
    - `theta2 = theta4`
  - 默认值：`direct`
- `groups`
  - 显式定义优化变量分组。每个组共享一个优化变量。
  - 可写成 mapping，也可写成带 `name/members/initial/bounds` 的列表。
  - `members` 必填，`initial` 和 `bounds` 可选。
- `include_unlisted`
  - 当你写了 `groups` 之后，未出现在任何组里的 tunable 是否继续单独参与优化。
  - 默认值：
    - 如果没写 `groups`，默认 `true`
    - 如果写了 `groups`，默认 `false`

关于分组的补充规则：

- 如果组内没写 `initial`，默认取组内成员当前值的平均值。
- 如果组内没写 `bounds`，默认取各成员 bounds 的交集。
- 如果某个 tunable 同时出现在多个组里，程序会报错。
- 如果你当前 YAML 里的初值本身带有重要的非对称性，那么 `grouped` 或 `symmetric4` 可能会在优化开始前就把这个结构压坏；这时更建议先用 `mode: direct`。

#### `optimization.strategy.global`

这一节控制全局优化，也就是 `scipy.optimize.differential_evolution`。

- `enabled`
  - 是否启用全局优化。
  - 默认值：`true`
- `maxiter`
  - 全局优化迭代轮数。
  - 默认值：`1`
- `popsize`
  - 全局优化种群规模系数。
  - 默认值：`2`
- `polish`
  - 是否让 `differential_evolution` 自己在末尾做 polish。
  - 默认值：`false`
- `seed`
  - 随机种子。
  - 默认值：`42`
- `strategy`
  - DE 策略名。
  - 默认值：`best1bin`
- `tol`
  - 收敛容差。
  - 默认值：`0.01`
- `mutation`
  - 变异系数，可以写单个数，也可以写 `[low, high]`。
  - 默认值：`[0.5, 1.0]`
- `recombination`
  - 重组概率。
  - 默认值：`0.7`

#### `optimization.strategy.local`

这一节控制局部优化，当前默认使用 `scipy.optimize.minimize(..., method="Powell")`。

- `enabled`
  - 是否启用局部优化。
  - 默认值：`true`
- `method`
  - 当前推荐写 `Powell`。
  - 默认值：`Powell`
- `maxiter`
  - 局部优化最大迭代数。
  - 默认值：`12`
- `maxfev`
  - 局部优化最大函数评估次数。
  - 默认值：`null`
  - 解释：如果不写，程序会自动用“剩余预算”作为上限。
- `xtol`
  - 参数收敛阈值。
  - 默认值：`0.001`
- `ftol`
  - 目标函数收敛阈值。
  - 默认值：`0.001`

兼容性说明：

- 旧写法 `xatol` / `fatol` 也会被接受，但现在更推荐写成 `xtol` / `ftol`。

#### `optimization.max_evaluations`

- 含义
  - 单次优化允许的最大 MATLAB 仿真次数。
- 默认值：`24`

这是一个硬限制。达到这个次数后，程序会停止继续搜索，并保留当前最优结果。

补充说明：

- 如果启用了局部优化，程序当前会自动为局部阶段预留一部分预算，避免全局阶段把全部仿真次数提前耗尽。

## `max_evaluations`、`global.maxiter`、`local.maxiter` 到底是什么关系

这一部分最容易混淆。最简单的理解方式是：

- `max_evaluations`
  - 管的是“整次优化最多允许做多少次 MATLAB 仿真”。
  - 这是最硬的上限。
- `strategy.global.maxiter`
  - 管的是“全局优化最多跑多少代”。
  - 它不是直接等于仿真次数。
- `strategy.local.maxiter`
  - 管的是“局部优化最多做多少轮迭代”。
  - 它也不是直接等于仿真次数。
- `strategy.local.maxfev`
  - 管的是“局部优化最多允许做多少次目标函数评估”。
  - 这个更接近真实的仿真次数限制。

### 先记住一个总原则

在当前实现里，真正决定“会不会提前停”的第一优先级是：

1. `max_evaluations`
2. 全局阶段预留给局部阶段后的 `global budget`
3. 局部阶段自己的 `maxfev`
4. `global.maxiter` / `local.maxiter`

也就是说：

- `global.maxiter` 和 `local.maxiter` 更像“算法层面的意愿上限”；
- `max_evaluations` 才是“资源层面的硬上限”。

如果预算不够，优化器还没把 `maxiter` 跑完，也会提前停。

### 当前实现里实际怎么分配预算

假设你写了：

```yaml
optimization:
  max_evaluations: 60
```

并且 `local.enabled: true`，程序会先自动给局部阶段预留一部分预算：

```text
reserved_local_budget = min(max(8, 参数维度 * 4), max(1, max_evaluations // 3))
global_budget_limit = max_evaluations - reserved_local_budget
```

所以：

- 全局阶段不能无限吃掉全部预算；
- 局部阶段至少还能拿到一部分仿真次数做精修。

### 对你这个项目，参数维度会直接影响预算切分

如果你是 `mode: direct`，RAMZI 现在有 8 个 tunable，所以参数维度是 8。

这时如果：

```yaml
max_evaluations: 60
```

那么：

- `参数维度 * 4 = 32`
- `max(8, 32) = 32`
- `max_evaluations // 3 = 20`
- 所以 `reserved_local_budget = min(32, 20) = 20`
- 于是 `global_budget_limit = 60 - 20 = 40`

也就是说：

- 全局阶段最多大约只能用 40 次仿真；
- 局部阶段至少会被留出大约 20 次仿真空间。

如果你是 4 参数对称版，那么维度是 4：

- `参数维度 * 4 = 16`
- `max(8, 16) = 16`
- `max_evaluations // 3 = 20`
- 所以 `reserved_local_budget = 16`
- 于是 `global_budget_limit = 44`

### 为什么 `global.maxiter` 不是“全局会跑多少次仿真”

因为你现在用的是 `scipy.optimize.differential_evolution`。它每一代不是只评估 1 次，而是要评估一整个人群。

粗略地说，全局阶段的函数评估次数大约是：

```text
global_evals ≈ population_size * (1 + global.maxiter)
```

而这里：

```text
population_size ≈ popsize * 参数维度
```

所以在你的项目里：

- 如果是 8 维 direct
- `global.popsize: 4`

那么一代的人群大约就是：

```text
4 * 8 = 32
```

如果：

```yaml
global:
  maxiter: 2
  popsize: 4
```

那么全局阶段大致就会想要：

```text
32 * (1 + 2) = 96 次评估
```

但如果你的 `global_budget_limit` 只有 40，那么它根本跑不满 2 代，就会被预算提前截断。

这也是为什么有时候你感觉：

- 明明写了很大的 `global.maxiter`
- 结果全局阶段却很快停了

原因通常不是 `maxiter` 没生效，而是预算先到了。

### 为什么 `local.maxiter` 也不是“局部会跑多少次仿真”

因为 `Powell` 每一轮迭代内部也可能调用很多次目标函数。

所以：

- `local.maxiter` 是“最多做多少轮 Powell 迭代”
- `local.maxfev` 才更接近“最多做多少次局部仿真”

如果你没有显式写 `local.maxfev`，当前实现会自动把它设成“局部阶段剩余预算”。

所以局部阶段真实会停下来的原因通常是：

- 先碰到剩余预算；
- 或者先碰到你显式写的 `local.maxfev`；
- 最后才是 `local.maxiter`。

### 一个最实用的经验公式

如果你只是想“先大概配得合理一点”，可以这样想：

1. 先定 `max_evaluations`
   - 这是你愿意付出的总仿真成本。
2. 再看参数维度
   - 8 维 direct 会比 4 维 grouped 贵很多。
3. 再用 `global.popsize * 参数维度 * (1 + global.maxiter)` 估算全局想吃掉多少预算。
4. 确保这个值不要明显大于 `global_budget_limit`。
5. 再确认局部阶段至少还剩 `10~30` 次评估空间。

### 结合你现在的 RAMZI，给几个实用配置

#### 低成本试跑

适合先看趋势，不追求最后结果：

```yaml
optimization:
  max_evaluations: 30
  strategy:
    global:
      maxiter: 1
      popsize: 3
    local:
      maxiter: 10
```

如果是 8 维 direct：

- 人群大约 `3 * 8 = 24`
- 全局想要大约 `24 * (1 + 1) = 48` 次
- 但预算只有 30，所以全局会被明显截断

这类配置的意义主要是“摸地形”。

#### 中等预算

更适合认真找一个可用初解：

```yaml
optimization:
  max_evaluations: 60
  strategy:
    global:
      maxiter: 1
      popsize: 4
    local:
      maxiter: 20
```

如果是 8 维 direct：

- 全局人群大约 `32`
- 全局想要大约 `64` 次
- 但当前实现只会给全局大约 `40` 次

所以这时全局通常只能跑一部分，然后把剩余预算留给局部精修。

#### 你容易踩坑的一种配置

```yaml
optimization:
  max_evaluations: 60
  strategy:
    global:
      maxiter: 10
      popsize: 8
```

如果是 8 维 direct：

- 人群大约 `8 * 8 = 64`
- 全局想要大约 `64 * 11 = 704` 次

这和 `max_evaluations: 60` 完全不在一个量级上。结果通常就是：

- 你以为会跑很多代；
- 实际上很早就被预算截断；
- 局部阶段也会很紧张。

### 一句话总结

可以把它记成：

- `max_evaluations` 决定“这次总共花多少钱”
- `global.maxiter + global.popsize` 决定“全局阶段有多贪”
- `local.maxiter + local.maxfev` 决定“局部阶段能修多久”

如果只想先避免配错，最稳妥的顺序是：

1. 先定 `max_evaluations`
2. 再把 `global.popsize` 设小
3. 再把 `global.maxiter` 设成 `1` 或 `2`
4. 最后给局部阶段留够预算

#### `optimization.logging`

- `tensorboard`
  - 是否写入 TensorBoard event。
  - 默认值：`true`
- `plot_every`
  - 每多少次新评估就刷新一次 `history.png`。
  - 默认值：`4`

## 低算力机器上的建议

如果当前机器资源紧张，建议遵循下面的顺序：

1. 先用 4 参数对称版，也就是 `groups` 或 `mode: symmetric4`。
2. 先把 `max_evaluations` 压低到 `8~20` 之间。
3. 先只看趋势和记录，不追求一次就找到最终好解。
4. 先观察 `history.png` 和 TensorBoard 里的 `loss/*`、`metrics/*` 曲线，再决定要不要增加预算。
5. 如果结果明显卡在错误的带宽或中心位置，再调 `weights.center`、`weights.bandwidth`、`weights.stopband`。
