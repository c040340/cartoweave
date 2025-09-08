# agents.md / 代理说明

> **English + 中文双语版本**，便于国内外协作开发者快速理解 CartoWeave 的“功能代理（agents）”分工与协作方式。

## 0. TL;DR / 摘要
- **Goal / 核心目标**: Describe the system as a pipeline of agents linking **data → scene script → solve plan → numerical solver → recording & visualization**. / 用“代理”的视角把数据、脚本、求解、记录、可视化串联起来。
- **Key APIs / 核心接口**: `solve_frame`, `solve_plan`, `record(snapshot)`, `forces.REGISTRY`. / 核心函数与注册表。
- **Most common extensions / 最常见扩展**: add new force terms, solver stages, or visualization metrics. / 新增力项、求解阶段或可视化指标。

---

## 1. High‑level View / 顶层视图

```
[DataGenAgent] → [SceneScriptAgent] → [SolvePlanAgent] → [SolverAgents]
  → [FieldEvalAgent] → [RecorderAgent] → [VizBridgeAgent]
```

- **Data Flow / 数据流**: Scene evolves along steps (script) & stages (plan). / 场景在步骤与阶段双重时间轴上演进。
- **Control Flow / 控制流**: scene_script_runner runs steps, solve_plan executes staged solvers. / 脚本驱动步骤，求解计划驱动数值求解。

---

## 2. Agents Overview / 代理一览

### DataGenAgent — Scene Generation / 场景生成
- **Module / 模块**: `data/random.py`
- **Responsibility / 职责**: Generate reproducible scenes or load from cache. / 生成可复现实验场景或从缓存读取。
- **Input / 输入**: frame_size, sampling config, random seed. / 画幅大小、采样配置、随机种子。
- **Output / 输出**: `Scene` dict with points, lines, areas, labels_init, WH, frame_size. / 标准化的场景数据。

### SceneScriptAgent — Scene Timeline / 场景脚本
- **Module**: `orchestrators/scene_script_runner.py`
- **Role / 职责**: Apply step updates, call solve_plan each step. / 应用场景步骤，逐步调用求解计划。

### SolvePlanAgent — Stage Composition / 阶段编排
- **Module**: `orchestrators/solve_plan.py`
- **Role**: Compose stage configs, enable/disable force terms, call solver agents. / 合成阶段配置，启用/禁用力项，调度求解器。

### SolverAgents — Numerical Solvers / 数值求解器
- **Modules**: `engine/solvers/semi_newton.py`, `lbfgs.py`, `hybrid.py`
- **Role**: Minimize energy E(P), produce solution trajectory. / 最小化能量函数，返回轨迹。

### FieldEvalAgent — Force Evaluation / 力场评估
- **Module**: `engine/core_eval.py`
- **Role**: Aggregate force terms, compute E & ∇E. / 聚合力项，计算能量与梯度。

### RecorderAgent — History Recording / 记录代理
- **Module**: `orchestrators/solve_plan_runner.py`
- **Role**: Save snapshots with stage_id & step boundaries. / 保存求解快照，标注阶段与步骤边界。

### VizBridgeAgent — Visualization Bridge / 可视化桥接
- **Modules**: `viz/view.py`, `viz/panels.py`
- **Role**: Build payload for UI, interactive visualization. / 构建可视化载荷，提供交互界面。

### ConfigAgent — Config Loader / 配置代理
- **Modules**: `config/schema.py`, `loader.py`
- **Role**: Load, merge, and validate configs. / 加载、合并、校验配置。
- **Config namespace**: only `compute.*` is recognised; legacy keys raise errors.

---

## 3. Key Schemas / 核心数据契约

### SceneData / 场景数据
`points (Np,2)`, `lines (Nl,2,2)`, `areas (...)`, `labels_init (N,2)`, `WH (N,2)`

### Step / 脚本步骤
```json
{"op":"enter|exit|update","label_id":3,"mode":"small","wh":[80,28]}
```

### Stage / 阶段
```json
{"name":"pre_anchor","scale":{"anchor.k_local.spring":0.4}}
```

### History Record / 求解记录
Contains `P`, `components`, `meta(stage_id, step_idx)`. / 记录每次能量评估与位姿。

---

## 4. Extensibility / 扩展指南
- **Add Force Term / 新增力项**: create file in `engine/forces/`, register with `@register`. / 新增文件并注册。
- **Add Stage / 阶段**: edit `solve_plan.py`, update stage list. / 编辑阶段列表。
- **Add Metric / 指标**: implement in `viz/metrics.py`, integrate into panels. / 在指标模块实现并集成。

---

## 5. Observability / 可观测性
- **Logging / 日志**: enable `CFG_DEBUG_FORCES=1` to see norm stats. / 打开调试环境变量可观察范数。
- **Gradient Check / 梯度检查**: use `utils/checks.check_force_grad_consistency`. / 调用工具函数验证梯度。

---

## 6. Roadmap / 未来规划
- Standardize history schema to Parquet. / 历史记录结构标准化。
- WebGL visualization. / 支持 WebGL 可视化。
- Auto‑tuning force weights. / 自动调参。
- Learnable terms / 支持可学习力项。

