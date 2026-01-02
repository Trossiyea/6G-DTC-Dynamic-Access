# MADU‑Net（适中方案）: Neural‑Scored Greedy Block Scheduler (NS‑GBS)

本文档给出一个**适中复杂度**、适合硕士论文落地的神经网络调度方案：在保留当前仓库启发式调度器的“硬约束/硬更新”框架下，引入一个小型神经网络**替换启发式打分函数**，从而在 CQI‑free（地图先验）条件下获得更强性能，而无需深度展开、teacher 搜索或 RL。

> 核心思想：不让网络“学会怎么构造块”（那会变复杂），而是让网络“学会怎么给候选动作打分”，其余结构复用 `code/main.py` 的 `pf_schedule_radiomap_blocks`。

---

## 1. 与当前 Repo 对齐的背景与接口

### 1.1 现有 RadioMap 块调度器做了什么

本仓库的 RadioMap 调度器（函数 `pf_schedule_radiomap_blocks`）在每个 TTI 内执行：

- 使用 CQI‑free 观测构造每 UE/PRB 的预测指标矩阵（实现上为 `se_pred_k1` 或外部传入的 `se_metric_time[t]`）。
- 在满足 OFDMA 正交 + 单 UE 单连续块等约束下，通过“seed + grow（左/右扩展）”的贪心过程输出 `winners[z]`。
- 用真实信道（`snr_true`）做 `EESM + MCS (+HARQ)` 评估，得到 goodput/SE 等最终指标。

### 1.2 NS‑GBS 的定位（最小侵入）

NS‑GBS 不改变：

- 输出形式：仍为每 TTI 的 `winners[z]`（PRB→UE 分配）。
- 约束实现：仍由现有的块维护逻辑保证（连续性、PRB cap、HARQ gating、重传预留等）。
- 吞吐评估：仍走现有 `EESM+MCS(+HARQ)` 管线。

NS‑GBS 只改变一件事：

- **把启发式 `metric = Δ / Rbar (+ bonus)` 替换为神经网络 `score = ScoreNetθ(features)`**，用 `score` 决定每一步选哪个动作。

这类方法论文上通常被称为“Neural heuristic / Learning‑augmented greedy”。

---

## 2. 基线启发式（论文可用的“先提出启发式”）

为了论文结构更 solid，建议在仓库现有启发式基础上，明确写出一个“可解释的贪心块调度基线”：

- **候选动作**：
  - Seed：对尚未分配任何 PRB 的 UE，从其未占用 PRB 中选一个作为起始点；
  - Grow：对已有块的 UE，尝试向左或向右扩展 1 个 PRB（保持连续）。
- **启发式打分**（示例）：
  - Seed：$\Delta \approx \hat{s}_{u,z}$
  - Grow：$\Delta \approx (k{+}1)\cdot \overline{\hat{s}}_{u,[l..r\pm 1]} - k\cdot \overline{\hat{s}}_{u,[l..r]}$
  - PF：$\text{metric}=\Delta/(Rbar_u+\epsilon)$（可叠加 HARQ 优先项）

然后引出：该启发式的瓶颈在于 $\hat{s}$ 是 CQI‑free 估计，且存在地图误差/模糊/CSI delay，使得手工的 $\Delta$ 估计偏差大，容易在频域纹理（连续干净子带 vs 孤立尖峰）上做出非最优选择。

---

## 3. NS‑GBS：神经网络增强的贪心块调度

### 3.1 关键设计：网络只负责“动作价值评估”

在每一步（每分配一个 PRB 的决策点），对所有可行候选动作 $a\in\mathcal{A}$ 计算分数：

\[
\text{score}(a)=\mathrm{ScoreNet}_\theta(\mathrm{feat}(a))
\]

选择分数最大的动作执行，并通过现有逻辑更新 `winners/l_idx/r_idx/k_assigned`。

这样做的优势：

- **可行性结构化保证**：C3/C4 等组合约束由算法结构保证，不需要 soft constraints。
- **工程侵入小**：不改评估链路，不改 HARQ。
- **训练难度可控**：可以用“单步监督”训练，不需要序列 RL。

### 3.2 候选动作集合（与实现一致）

为了控制复杂度，建议保持与现有实现一致的候选：

- Seed：对每个未开块 UE，只取 Top‑B 个 seed PRB（按观测 $\hat{s}_{u,z}$ 排序）。
- Grow：对已开块 UE，最多两个动作（grow‑left / grow‑right），前提是相邻 PRB 未被占用。
- 约束：HARQ gating / PRB cap / retx 预留保持不变。

候选动作数大约为 $U\cdot(B+2)$，对硕士论文实现/复现很友好。

### 3.3 动作特征设计（最重要，论文可写）

对一个候选动作 $a$（类型为 seed 或 grow），构造低维特征向量，强调“地图纹理 + PF + 块结构”：

**(A) UE‑局部频域纹理（来自 CQI‑free 观测）**
- $\hat{s}_{u,z}$ 及其邻域窗口：$[\hat{s}_{u,z-w},\dots,\hat{s}_{u,z+w}]$
- 邻域统计：均值/方差、左右差分、局部梯度、局部最小值（用于识别“干净连续子带”）

**(B) 当前块结构（如果 grow）**
- 当前块长度 $k$
- 当前块均值/方差：$\overline{\hat{s}}_{u,[l..r]}$
- 扩展后块均值变化：$\overline{\hat{s}}_{u,[l'..r']}-\overline{\hat{s}}_{u,[l..r]}$

**(C) PF 与系统状态**
- PF 记忆状态：$Rbar_u$ 或 $1/(Rbar_u+\epsilon)$
- 剩余 PRB cap：$K_{\max}-k$
- 可选 HARQ：是否可调度、是否重传 UE 标记、是否有 retx block 需求

**(D) 全局轻量上下文（可选，但不必复杂）**
- 当前已分配 PRB 数/剩余 PRB 比例
- 当前步骤 index（已分配比例），帮助网络理解“早期抢占 vs 后期填充”的不同策略

> 实现上可以把 (A)+(B)+(C) 拼成一个向量喂给 MLP；如果想更“神经网络味道”，(A) 的窗口可以先过一层小 1D‑CNN 再拼接。

### 3.4 网络结构（简单但足够写论文）

推荐两种都“适中”：

- **MLP‑ScoreNet（最稳）**：输入为拼接特征向量，2–3 层 MLP 输出标量 score。
- **Window‑CNN + MLP（更贴合“地图纹理”）**：对 (A) 的窗口序列做 1D‑CNN 得到 embedding，再与 (B)(C) 拼接过 MLP 输出 score。

网络输出可以解释为：

- 直接输出“动作价值”score（用于 argmax）
- 或输出“预测边际吞吐 $\widehat{\Delta}$”，然后再做 PF 归一化：$\widehat{\Delta}/(Rbar+\epsilon)$

后者更易与 PF 目标对齐、也更好解释。

### 3.5 NS‑GBS 推理流程（与现有贪心相同）

1. 构造观测矩阵 $\hat{\mathbf{s}}$（保持仓库现有机制：地图误差/blur/CSI delay）
2. 初始化 `winners/l_idx/r_idx/k_assigned/Rbar`
3. 循环直到分配完所有 PRB：
   - 构造候选动作集合 $\mathcal{A}$
   - 对每个候选动作抽取特征并用 ScoreNet 打分
   - 选最大分数动作，调用现有 assign/边界更新逻辑
4. 输出 `winners`，进入现有的 `EESM+MCS(+HARQ)` 评估

---

## 4. 训练：单步监督（不需要 teacher search / RL）

### 4.1 数据集怎么构造（离线采集，易复现）

在仿真运行时，对每个 TTI、每个分配步骤记录：

- 状态（观测 $\hat{\mathbf{s}}$、Rbar、当前 winners/块边界、HARQ 状态等）
- 候选动作集合 $\mathcal{A}$
- 每个候选动作在“真值链路”下的边际收益 $\Delta_{\text{true}}(a)$

其中 $\Delta_{\text{true}}(a)$ 的计算建议**与仓库一致**：

- 使用真实 `snr_true` 取出候选动作对应的块（若是 grow 则是扩展后的块），构造该块的 `sinr_vec_db`；
- 用仓库同款 `EESM + MCS` 近似得到该块的有效 SE/吞吐；
- 计算与当前块吞吐差值作为 $\Delta_{\text{true}}$。

> 这样训练目标与最终评估一致，避免“训练学的是 Shannon，测试跑的是 MCS/HARQ”带来的 gap。

### 4.2 两种训练目标（任选其一）

- **回归（更简单）**：最小化
  \[
  \mathcal{L}=\mathrm{Huber}(\widehat{\Delta}(a),\Delta_{\text{true}}(a))
  \]
  推理时用 $\widehat{\Delta}/(Rbar+\epsilon)$ 做 argmax。
- **分类（更稳健）**：在每个状态上找 $a^\*=\arg\max_a \Delta_{\text{true}}(a)$，最小化交叉熵：
  \[
  \mathcal{L}=-\log \pi_\theta(a^\*|\mathrm{state})
  \]
  推理时用 argmax 分数。

回归更易解释“网络在做 value approximation”，分类更易训练稳定。

### 4.3 训练数据的“鲁棒性增强”

为了突出 NN 优势，可在训练采集阶段开启/随机化：

- 地图误差 `radiomap_est_error_db`
- 地图 blur `radiomap_blur_sigma`
- CSI delay（baseline / RM path）
- 时间变化 flicker/drift、轨道动态（本仓库已有）

这样论文里可以明确展示：在 CQI‑free 观测失配更强时，NS‑GBS 比纯启发式更鲁棒。

---

## 5. 论文实验设计建议（不需要“做差”，也能凸显优势）

建议至少包含：

1. **对比**：启发式（原始） vs NS‑GBS（神经增强）
2. **误差扫描**：随着 `radiomap_est_error_db` / `baseline_csi_delay_ttis` 增大，性能差距变化
3. **消融**：
   - 仅用点值特征 vs 加入局部窗口纹理
   - 仅 seed 打分 vs seed+grow 全打分
   - 不含 HARQ 特征 vs 含 HARQ 特征
4. **复杂度**：报告每 TTI 平均打分次数（约 $U(B+2)$）与额外耗时

指标优先用仓库已有输出：

- `avg_se_radiomap`、相对增益、（若启用 HARQ）goodput/ACK 相关统计
- Jain fairness（若报告中已有）

---

## 6. 实施工作量评估（适中、硕士可控）

- **推理接入**：中低（只需要在现有动作打分位置调用网络；其余代码不动）
- **数据采集**：中（需要把“每步候选动作与真值 Δ”记录下来）
- **训练脚本与依赖**：中（引入 PyTorch 并写一个小训练脚本；模型很小）
- **复现实验**：中（固定随机种子与配置即可；与现有 `run_test.py` 对齐）

---

## 7. 一句话总结（论文写法）

NS‑GBS 将“连续块 PF 贪心调度”的可行性结构保留在启发式框架内，仅用神经网络学习更准确的候选动作价值评估，从而在 CQI‑free、地图误差与 CSI 老化条件下获得显著优于手工打分的调度性能，同时保持实现与复现难度可控。

