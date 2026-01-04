# ISAB‑NSGBS（Set Transformer / ISAB）实施 TODO

目标：在不改动现有吞吐/链路评估管线与“硬约束贪心解码”的前提下，把 NS‑GBS 的动作打分从“独立 MLP”升级为“候选集合上下文（ISAB）”，并用 **listwise 排序 + 回归**（可选不确定性头）显著提升推理效果；先用 **全量候选 M** 跑通并验证收益，再考虑任何剪枝/加速。

已具备的基础（当前 repo 已完成）：
- 数据采集：每步候选集合 `features` + 真值边际收益 `deltas` 已能离线 dump（`tools/generate_nsgbs_dataset.py`）。
- 推理接入：每步对候选进行 batch 打分并 argmax 选动作（`pf_schedule_radiomap_blocks`）。

约束/设定（论文口径固定）：
- `Z=51`（PRB 数保持不变），`N_UE=100`（主要评测规模），seed Top‑`B=4`，grow 仅 L/R。
- 主指标：`avg_se_radiomap`；次指标：goodput / NACK（启用 HARQ 时）、Jain fairness（如需要）。

---

## Phase 0：评测协议与基线对齐（先把“怎么比”定死）

- [ ] 固定评测场景与随机种子：至少 `toronto_single/shanghai_single`，每场景 `seed∈{1..5}`。
- [ ] 固定关键规模参数：`N_UE=100, Z=51, T`（建议 `T>=1000` 做论文主实验；`T=200~500` 用于快速迭代）。
- [ ] 明确对比组（同一观测链路，只换打分器）：
  - Heuristic（`Δ/Rbar`）
  - MLP‑NSGBS（现有）
  - ISAB‑NSGBS（新增）
- [ ] 记录推理开销：每 TTI / 每步的平均候选数 `M`、模型前向耗时（ms/TTI）。

---

## Phase 1：数据集（N_UE=100）与特征口径

- [ ] 用 `tools/generate_nsgbs_dataset.py` 生成 `N_UE=100` 数据集（先小后大）：
  - 小规模 sanity：`T=200~500`
  - 论文规模：`T>=1000`（必要时多场景合并）
- [ ] 训练时用 **全量候选集合** 的 `deltas` 做监督（不只用 argmax 标签）。
- [ ] 特征扩展策略（建议不改采集脚本也能做）：
  - token 侧补充 `z/Z`、`step/Z`（可由样本里的 `actions` 和 `step` 计算）
  - 动作类型 embedding：复用已有 `kind_id`
- [ ] 归一化：沿用现有 mean/std（按候选维度统计），并在 meta 中记录 `feature_dim` 与扩展策略版本号。

---

## Phase 2：ISAB‑NSGBS 模型定义（效果优先的默认配置）

- [ ] 模型形态：Set Transformer（ISAB）对候选集合做上下文编码，输出每个候选动作的 score。
- [ ] 支持 padding+mask（每步候选数 `M` 可变）。
- [ ] 默认超参（先保效果，再做降参曲线）：
  - `d_model=128, heads=4, isab_layers=2, inducing_m=32`
- [ ] 可选 global token：汇总本步全局信息（`assigned_cnt/Z`、剩余 PRB、HARQ 统计等）。

---

## Phase 3：训练目标（listwise 排序 + 回归；可选不确定性头）

- [ ] listwise 排序损失（推荐作为主损失）：
  - 目标分布：`p = softmax(deltas / tau)`（`tau` 作为超参）
  - 预测分布：`q = softmax(scores)`
  - 损失：`KL(p||q)` 或 `CE(p,q)`（mask 掉 padding）
- [ ] 回归项（用于尺度校准与泛化）：Huber(`μ - deltas`)，权重 `λ`。
- [ ] 可选不确定性头（异方差）：输出 `μ, logσ²`，回归用 Gaussian NLL；推理可用 `μ - κσ` 做鲁棒选择（`κ` 由验证集扫描）。
- [ ] 离线指标：Top‑1 accuracy、NDCG@K、Spearman（排序质量）+（可选）NLL/校准曲线。

---

## Phase 4：训练脚本与模型导出

- [x] 新增训练脚本 `train_nsgbs_isab.py`（ISAB + listwise/回归）：
  - 输入：`features (M,F)`、`deltas (M,)`、`mask (M,)`（batch 内 pad）
  - 输出：`*.pt` + `*.pt.json`（含 `model_kind=isab`、超参、norm、feature 版本）
- [ ] 训练/验证划分：按样本随机划分 +（可选）按场景留一验证（检验泛化）。
- [ ] 可选 TorchScript 导出（用于后续推理加速对比）。

---

## Phase 5：推理接入（全量 M；不做剪枝）

- [ ] 在 `code/nsgbs.py` 中增加 ISAB 模型加载分支（从 meta 识别 `model_kind`）。
- [ ] 在 `pf_schedule_radiomap_blocks` 中保持候选枚举不变，每步构造 **全量候选集合** 的特征矩阵与 mask，单次前向得到所有 score，再 argmax 选动作。
- [ ] 验证一致性：不启用 ISAB 时，现有 heuristic/MLP 行为完全不变。

---

## Phase 6：离线与系统级验证（收益优先）

- [ ] 离线：同一数据集上比较 MLP vs ISAB 的 Top‑1 / NDCG 提升，并做温度 `tau`、回归权重 `λ` 扫描。
- [ ] 系统：在 `N_UE=100, Z=51` 下跑多 seed，多场景对比三组的 `avg_se_radiomap`（均值+方差）。
- [ ] 鲁棒性（论文加分）：扫描 `radiomap_est_error_db`、`baseline_csi_delay_ttis`，比较曲线斜率与稳定性。
- [ ] 复杂度报告：统计每 TTI 平均候选数与推理耗时（ms/TTI）。

---

## Phase 7：性能优化（确认收益后再做）

- [ ] 降参曲线：`layers∈{1,2}`、`d_model∈{64,128}`、`m∈{16,32}`，画性能‑时延折中。
- [ ] 仅在 ISAB 明显有收益后再引入“两阶段粗排+重排”（先便宜打分选 Top‑K，再 ISAB 重排）。
- [ ] 尝试 TorchScript / 量化（如需要），并与原生 PyTorch 推理耗时对比。

---

## 里程碑（建议）

- [ ] M1：`N_UE=100` 数据集生成 + 离线训练跑通（有 NDCG/Top‑1 指标）
- [ ] M2：ISAB 推理接入跑通（全量 M），系统指标优于 MLP（至少 2 场景×多 seed）
- [ ] M3：完成鲁棒性扫描 + 复杂度报告（ms/TTI、M 分布）
- [ ] M4：完成性能‑时延折中（降参/可选两阶段）并产出论文图表
