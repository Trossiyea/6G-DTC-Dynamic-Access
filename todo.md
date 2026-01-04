# NS‑GBS（Neural‑Scored Greedy Block Scheduler）实施 TODO

目标：在不改动现有吞吐/链路评估管线的前提下，将 `pf_schedule_radiomap_blocks` 的“动作打分”从启发式 `Δ/Rbar` 升级为小型神经网络打分，并完成可复现实验用于论文写作。

---

## Phase 0：范围与对齐（先定边界）

- [x] 指标对齐：主指标 SE（`avg_se_radiomap`），次要指标 goodput（若启用 HARQ）。
- [x] 训练标签口径：选项 A（用真实 `snr_true` + 同款 `EESM+MCS` 计算候选动作的 `Δ_true`）。
- [x] 训练目标形式：分类 `argmax Δ_true`（每个状态选最优动作）。
- [x] 候选集规模：seed Top‑`B=4`（消融做 `B∈{2,4,8}`），grow 仅 L/R（与现有一致）。
- [x] 训练/验证数据规模：先小规模跑通（`T=200, N_UE=50`），再扩到论文规模（建议 `T>=1000, N_UE=100`，与默认配置一致）。

---

## Phase 1：代码结构调整（为接入 NN 做“最小重构”）

- [x] 在 `code/main.py` 中把“动作枚举/打分/选择”拆出清晰函数（不改现有默认行为）：
  - `iter_actions()` 生成候选动作
  - `score_action_heuristic(...)` 复用原始 `Δ/Rbar` 逻辑
  - `apply_assign(...)` 保持原有硬更新
- [x] 增加配置开关（放在 `code/config.py`，test configs 可覆盖）：
  - `scheduler_kind: {"heuristic","nsgbs"}`
  - `nsgbs_model_path`（可为空表示不用 NN）
  - `nsgbs_topB`, `nsgbs_window`, `nsgbs_use_harq_features`, `nsgbs_score_mode`
- [x] 保持输出与现有一致：`winners`、`record_assignments`、HARQ 逻辑与 water‑filling 后处理不变。

---

## Phase 2：数据采集（构造 NS‑GBS 训练集）

- [x] 在调度循环中记录“训练样本”（建议先做离线 dump，后续再考虑在线）：
  - 状态：`se_obs`（观测）、`Rbar`、`k_assigned/l_idx/r_idx`、剩余 PRB、HARQ gating/retx 标记（可选）
  - 候选动作集合 `A`（seed/grow 的 `(u,z,type)`）
  - 标签：每个候选动作的 `Δ_true` 或最优动作 `a*`
- [x] 实现 `Δ_true` 的计算函数（尽量复用仓库已有 block‑SE 逻辑）：
  - 输入：`snr_true[u, li:ri]`（真实链路），输出：block 吞吐/SE
  - 计算：`Δ_true = thr(new_block) - thr(old_block)`；seed 对应 old=0
- [x] 采样/降成本策略（可配开关）：
  - 每个 TTI 只采样部分 step（例如每 2/4 步采一次）
  - 对 seed 只保留 Top‑B（暂不加随机负样本）
- [x] 新增脚本（建议放 `tools/`）：
  - `tools/generate_nsgbs_dataset.py`：运行指定场景/配置，输出 `output/datasets/*.npz`
- [x] 输出数据格式建议（便于训练）：NPZ + json sidecar（记录 config、特征归一化参数、版本信息、随机种子）。

---

## Phase 3：训练（小模型、可复现）

- [x] 训练依赖策略：
  - 选项 A：新增 `environment_nn.yml`（含 PyTorch），训练与仿真环境分离；
  - 选项 B：本仓库只保留推理代码，训练在外部环境完成后导出权重。
- [x] 新增训练脚本：
  - `train_nsgbs.py`（读取 NPZ，训练 MLP）
  - 支持 `--seed`、`--epochs`、`--batch-size`、`--hidden`、`--depth`、`--dropout`
- [x] 模型导出与推理形态：
  - 选项 A：`*.pt`（state_dict）+ 可选 `*.ts`（TorchScript）
  - 选项 B：导出为 numpy 权重（纯 numpy MLP，最轻量）
- [x] 训练质量检查：
  - held‑out loss/accuracy（训练脚本已输出）
  - 关键特征消融（`--drop-window` / `--drop-harq`）

---

## Phase 4：集成推理（NS‑GBS scheduler）

- [x] 新增推理模块（`code/nsgbs.py`）：
  - `load_nsgbs_scorer(path)` + `score(features)`
- [x] 在 `pf_schedule_radiomap_blocks` 的打分处接入：
  - `if scheduler_kind=="nsgbs": metric = score_net(...) else: metric = heuristic(...)`
- [x] 确保默认配置不受影响（不开 `nsgbs` 时结果完全一致）。

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
