# NS‑GBS（Neural‑Scored Greedy Block Scheduler）实施 TODO

目标：在不改动现有吞吐/链路评估管线的前提下，将 `pf_schedule_radiomap_blocks` 的“动作打分”从启发式 `Δ/Rbar` 升级为小型神经网络打分，并完成可复现实验用于论文写作。

---

## Phase 0：范围与对齐（先定边界）

- [ ] 明确论文主指标：`avg_se_radiomap`（SE）还是 HARQ `goodput`（更贴近系统吞吐）；优先与现有 `run_test.py` 输出对齐。
- [ ] 明确训练标签口径（和评估尽量一致）：
  - 选项 A：用真实 `snr_true` + 同款 `EESM+MCS` 计算候选动作的 `Δ_true`（最对齐，成本较高）。
  - 选项 B：用真实 per‑PRB MCS‑SE 的 block mean 近似 `Δ_true`（更快，gap 更大）。
- [ ] 决定 NS‑GBS 形式：回归 `Δ_true`（再除以 `Rbar`）或分类 `argmax Δ_true`（更稳）。
- [ ] 决定候选集规模：seed Top‑`B`（建议 2/4/8 做消融），grow 仅 L/R（与现有一致）。
- [ ] 决定训练/验证数据规模：先用较小 `T/N_UE` 跑通闭环，再扩展到论文规模。

---

## Phase 1：代码结构调整（为接入 NN 做“最小重构”）

- [ ] 在 `code/main.py` 中把“动作枚举/打分/选择”拆出清晰函数（不改现有默认行为）：
  - `enumerate_actions(...) -> list[action]`
  - `score_actions_heuristic(...) -> np.ndarray`
  - `apply_action(...)`（复用现有 `apply_assign`）
- [ ] 增加配置开关（建议放在 `code/config.py` + test configs 中覆盖）：
  - `scheduler_kind: {"heuristic","nsgbs"}`
  - `nsgbs_model_path`（可为空表示不用 NN）
  - `nsgbs_topB`, `nsgbs_window`, `nsgbs_use_harq_features`, `nsgbs_score_mode`
- [ ] 保持输出与现有一致：`winners`、`record_assignments`、HARQ 逻辑与 water‑filling 后处理不变。

---

## Phase 2：数据采集（构造 NS‑GBS 训练集）

- [ ] 在调度循环中记录“训练样本”（建议先做离线 dump，后续再考虑在线）：
  - 状态：`se_obs`（观测）、`Rbar`、`k_assigned/l_idx/r_idx`、剩余 PRB、HARQ gating/retx 标记（可选）
  - 候选动作集合 `A`（seed/grow 的 `(u,z,type)`）
  - 标签：每个候选动作的 `Δ_true` 或最优动作 `a*`
- [ ] 实现 `Δ_true` 的计算函数（尽量复用仓库已有 block‑SE 逻辑）：
  - 输入：`snr_true[u, li:ri]`（真实链路），输出：block 吞吐/SE
  - 计算：`Δ_true = thr(new_block) - thr(old_block)`；seed 对应 old=0
- [ ] 采样/降成本策略（建议做成可配开关）：
  - 每个 TTI 只采样部分 step（例如每 2/4 步采一次）
  - 对 seed 只保留 Top‑B + 随机负样本（避免全量候选过大）
- [ ] 新增脚本（建议放 `tools/`）：
  - `tools/generate_nsgbs_dataset.py`：运行指定场景/配置，输出 `output/datasets/*.npz`
- [ ] 输出数据格式建议（便于训练）：NPZ + json sidecar（记录 config、特征归一化参数、版本信息、随机种子）。

---

## Phase 3：训练（小模型、可复现）

- [ ] 训练依赖策略：
  - 选项 A：新增 `environment_nn.yml`（含 PyTorch），训练与仿真环境分离；
  - 选项 B：本仓库只保留推理代码，训练在外部环境完成后导出权重。
- [ ] 新增训练脚本：
  - `train_nsgbs.py`（读取 NPZ，训练 MLP 或 Window‑CNN+MLP）
  - 支持 `--mode {regress, classify}`、`--topB`、`--window`、`--seed`
- [ ] 模型导出与推理形态：
  - 选项 A：TorchScript（推理时仍需要 torch）
  - 选项 B：导出为 numpy 权重（纯 numpy MLP，最轻量）
- [ ] 训练质量检查：
  - held‑out loss/accuracy
  - 关键特征消融（无窗口纹理/无 HARQ 特征）

---

## Phase 4：集成推理（NS‑GBS scheduler）

- [ ] 新增推理模块（建议 `code/nsgbs.py`）：
  - `load_model(path)`, `extract_features(...)`, `score_actions(...)`
- [ ] 在 `pf_schedule_radiomap_blocks` 的打分处接入：
  - `if scheduler_kind=="nsgbs": metric = score_net(...) else: metric = heuristic(...)`
- [ ] 确保默认配置不受影响（不开 `nsgbs` 时结果完全一致）。

---

## Phase 5：实验与论文材料（最小闭环到可写）

- [ ] 对比实验矩阵（建议至少）：
  - Heuristic vs NS‑GBS（同样的观测链路）
  - `radiomap_est_error_db`、`baseline_csi_delay_ttis` 扫描（体现鲁棒性优势）
  - Top‑`B`、窗口大小消融（复杂度‑性能权衡）
- [ ] 报告复杂度：每 TTI 打分次数约 `U*(B+2)`，额外推理耗时（ms/TTI）。
- [ ] 产出论文图表：
  - SE/goodput 提升曲线（误差/延迟扫描）
  - Jain fairness（如需要）
  - 复杂度‑性能折中图（B、window）
- [ ] 在文档中记录复现步骤：
  - 生成数据集命令、训练命令、跑测试场景命令（`run_test.py`）

---

## 里程碑（建议）

- [ ] M1：仅重构 + 保持启发式结果不变（回归测试通过）
- [ ] M2：跑通数据集生成（小规模 T/N_UE），能产出可训练数据
- [ ] M3：训练出一个可推理模型，接入后能在小规模场景上看到提升
- [ ] M4：完成多场景对比 + 消融 + 论文图表

