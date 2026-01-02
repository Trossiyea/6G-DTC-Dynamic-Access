# MADU-Net: Map-Aware Deep Unfolded **Block Scheduler** (Aligned with This Repo)

本文档给出一套**与当前仓库实现严格对齐**的 MADU‑Net 设计，用于手机直连卫星（DtC）下行 OFDMA 场景的 **CQI‑free 细粒度调度**。  
与旧版本（以连续功率矩阵 $\mathbf{P}\in\mathbb{R}^{U\times K}$ 为变量）不同，本仓库的调度核心是 **PRB→UE 分配** + **每 UE 单连续块** 约束（见 `code/main.py` 的 `pf_schedule_radiomap_blocks`），因此 MADU‑Net 也以“块调度”为中心建模。

---

## 1. 与仓库对齐的问题定义 (Problem Formulation)

### 1.1 资源与时隙

- UE 集合：$\mathcal{U}=\{1,\dots,U\}$
- PRB 集合：$\mathcal{Z}=\{1,\dots,Z\}$（本仓库用 `Z` 表示 PRB 数）
- TTI：$t\in\{1,\dots,T\}$

每个 TTI 输出一个分配向量：
\[
\mathbf{a}_t \in \mathcal{U}^{Z},\quad a_{t,z}\in\mathcal{U}
\]
其中 $a_{t,z}$ 表示 PRB $z$ 分配给哪个 UE（对应代码中的 `winners[z]`）。

### 1.2 CQI‑free 观测与真实评估（Repo 对齐）

本仓库把“调度可见的信息”和“真实性能评估”分开：

- **Observation（调度可见）**：$\hat{\mathbf{s}}_t\in\mathbb{R}^{U\times Z}$，每个 UE/PRB 的预测指标（推荐用 *预测 SE* 或由预测 SINR 映射得到的 SE）。
  - 来源：UE 位置 + 几何链路预算 + radiomap 干扰查询（可叠加地图误差/blur 与 CSI delay）。
  - 对应实现：`se_pred_k1` 或 `se_metric_time[t]`（`code/main.py` 内部构造，调度器实际使用的就是这个矩阵）。
- **Truth（真实性能）**：$\mathbf{\gamma}_t\in\mathbb{R}^{U\times Z}$ 或 $\mathbf{snr}_t$，包含真实 shadowing/fast fading/真实干扰扰动等。
  - 对应实现：`snr_true`，用于 `EESM + MCS (+HARQ)` 的吞吐/goodput 结算。

> 论文叙述建议：MADU‑Net 仅使用 $\hat{\mathbf{s}}_t$（CQI‑free），并在真实信道上评估，从而突出“观测失配鲁棒性”。

### 1.3 约束（与仓库一致）

- **C3（OFDMA 正交）**：每个 PRB 只能分配给一个 UE（$\mathbf{a}_t$ 本身即满足）。
- **C4（连续块）**：每个 UE 在一个 TTI 内至多获得一个连续 PRB 区间 $[l_{t,u},r_{t,u}]$（可为空）：
  \[
  \{z: a_{t,z}=u\}\in\{\varnothing,\{l_{t,u},\dots,r_{t,u}\}\}
  \]
- 可选工程约束（同仓库参数）：每 UE PRB 上限 $r_{t,u}-l_{t,u}+1\le K_{\max}$；HARQ gating（`can_schedule(u)`）；重传块预留（`retx_requirements`）。

### 1.4 目标：PF‑goodput 最大化（与实现一致）

定义每 UE 在 TTI $t$ 的有效吞吐（或等价 SE 和）：
\[
\mathrm{thr}_{t,u}=\sum_{z:a_{t,z}=u} \tilde{s}_{t,u,z}\cdot \eta
\]
- $\tilde{s}_{t,u,z}$：真实链路映射后的有效 SE（`EESM + MCS (+HARQ)` 后的 goodput 归因）
- $\eta$：开销因子（`overhead_eff`）

PF 记忆状态（指数平均）：
\[
R_{t+1,u}=(1-\beta)R_{t,u}+\beta\cdot \mathrm{thr}_{t,u}
\]
目标是提升长期 PF‑goodput（等价理解为最大化 $\sum_u \log R_{t,u}$ 的增量）。

---

## 2. 启发式基线（当前仓库）(Heuristic Baseline)

当前 RadioMap 调度器 `pf_schedule_radiomap_blocks` 采用“seed + grow”的贪心块构造：

1. 未开块 UE 选择一个未分配 PRB 作为 seed；
2. 已开块 UE 只允许向左/向右扩展（保证连续）；
3. 每一步选择最大化 **PF 边际增益** 的动作：
   \[
   \text{metric}(a)\approx \frac{\Delta \widehat{\mathrm{thr}}(a)}{R_{t,u}}
   \]
   其中 $\Delta \widehat{\mathrm{thr}}$ 由 $\hat{\mathbf{s}}_t$（或其块均值/单 MCS 的 EESM 近似）计算。
4. 可选：对最终 UE‑块执行 group water‑filling（同一块内每 PRB 等功率）。

该启发式强在“可行性与工程细节”，弱在“全局性”（纯贪心易受局部峰值/纹理误判影响）。MADU‑Net 的设计目标是：**保留硬约束更新**，仅学习“每一步如何选动作”，从而系统性超过贪心。

---

## 3. MADU‑Net（Repo 对齐版）架构：Deep Unfolded Block Scheduler

### 3.1 深度展开视角 (Deep Unfolding)

把一个 TTI 的块构造过程展开成 $L$ 层（通常 $L=Z$，也可提前停止以控复杂度）。第 $l$ 层维护状态：

- 已分配向量 $\mathbf{a}^{(l)}$（部分 PRB 已被占用）
- 每 UE 的块边界 $(l^{(l)}_u,r^{(l)}_u)$、已分配 PRB 数 $k^{(l)}_u$
- PF 状态 $R_{t,u}$
- CQI‑free 观测矩阵 $\hat{\mathbf{s}}_t$ 与其纹理编码
- 可选 HARQ 状态（`can_schedule`、`retx_mask`、`retx_requirements`）

每层选择一个离散动作 $a^{(l)}$ 并做**硬更新**：

- **硬更新（Hard Projection/Update）**：复用仓库同构的“assign + 边界更新”逻辑，结构上保证 C3/C4（无需 TV/Softmax 去软逼近约束）。

### 3.2 Map Encoder：把“点值”变成“纹理特征”

对每个 UE，把沿 PRB 的观测序列编码为局部纹理特征：

\[
\mathbf{F}_{t,u}=\mathrm{MapEnc}_\phi(\hat{\mathbf{s}}_{t,u}, \hat{\mathbf{i}}_{t,u}) \in \mathbb{R}^{Z\times C}
\]

- 推荐实现：1D‑CNN/TCN（对频域纹理更强、更轻量）；也可用小型 Transformer。
- 直觉：连续块调度关心“某个 PRB 周围是否也干净”，纹理特征可区分“孤立低干扰点”与“宽带干净子带”。

UE 级上下文向量：
\[
\mathbf{c}_{t,u}=\mathrm{UEEnc}_\psi([R_{t,u},\text{HARQ}_u,\text{geometry}_u,\dots])\in\mathbb{R}^{d}
\]

### 3.3 候选动作集：对齐实现、控制动作空间

为了与仓库流程一致并避免 $U\times Z$ 的巨大动作空间，在第 $l$ 层构造有限候选动作：

- **Seed 候选**：对每个未开块 UE，从其未分配 PRB 中取 Top‑$B$ 个（按 $\hat{s}_{t,u,z}$）作为候选 seed：$(u,z)$。
- **Grow 候选**：对已开块 UE，最多两个候选：向左扩展/向右扩展（若边界相邻 PRB 未分配）。
- **Retx 约束候选**：若某 UE 有重传块需求，则把满足需求的动作作为硬约束（必须选/必须预留）。

候选规模约为 $|\mathcal{A}^{(l)}|\approx U\cdot(B+2)$，易于批量打分并保持实时性。

### 3.4 MADU‑Net Scorer：学习替换“贪心 PF metric”

对每个候选动作 $a$ 构造动作特征（示例）：

- UE 上下文：$\mathbf{c}_{t,u}$
- 动作位置纹理：$\mathbf{F}_{t,u}[z]$（seed）或 $\mathbf{F}_{t,u}[l_u\!-\!1], \mathbf{F}_{t,u}[r_u\!+\!1]$（grow）
- 观测边际增益：$\Delta\hat{g}(a)$（seed 用 $\hat{s}_{t,u,z}$；grow 用“块均值变化 + 边界增益”等）
- PF 因子：$1/(R_{t,u}+\epsilon)$
- 约束相关：剩余 PRB 配额、HARQ retx 标记等

用一个轻量 MLP 输出动作 logit：
\[
\ell(a)=f_\theta(\mathrm{feat}(a))
\]

训练时用 softmax 形成策略 $\pi_\theta(a)\propto \exp(\ell(a)/\tau)$ 并采样（用于探索与 RL），推理时用 argmax 得到确定性决策。

### 3.5 输出与功率分配（保持评估一致性）

MADU‑Net 的**主输出**是每 TTI 的 `winners[z]`，完全复用当前评估管线：

- 真实链路：`snr_true` → `EESM + MCS`（单块单 MCS）→（可选）HARQ goodput；
- 可选后处理功率分配：复用 `equal_prb` 或 group `waterfill`（同仓库）。

若论文需要更强增益，可让网络额外输出块级权重 $\alpha_{t,u}$（作为 water‑filling 的权重或鲁棒 margin），但**不改变**调度变量与约束。

### 3.6 算法流程伪代码（单个 TTI）

```python
def madu_net_schedule_one_tti(se_obs, Rbar, harq_state, B=4, L=None):
    # se_obs: [U, Z]  (CQI-free 观测指标，等价于仓库调度器的 se_pred_k1 / se_metric_time[t])
    # Rbar:   [U]     (PF 记忆状态)
    # 输出: winners[z] (每个 PRB 分配给哪个 UE), 且每 UE 至多 1 个连续块

    U, Z = se_obs.shape
    L = Z if L is None else min(L, Z)

    # 0) 预处理：Map Encoder（逐 UE 沿 PRB 编码纹理）
    F = MapEnc(se_obs)  # F[u, z, C]

    # 1) 初始化（与仓库同构）
    winners = [-1] * Z
    l_idx = [-1] * U
    r_idx = [-1] * U
    k_assigned = [0] * U

    # 2) 可选：按 HARQ 重传需求预留块（与仓库逻辑一致）
    preassign_retx_blocks(winners, l_idx, r_idx, k_assigned, harq_state)

    for step in range(L):
        if all(z >= 0 for z in winners):
            break

        # 3) 构造候选动作集合（seed top-B + grow L/R）
        A = []
        for u in range(U):
            if not can_schedule(u, harq_state):
                continue
            if k_assigned[u] == 0:
                A += seed_candidates_topB(u, se_obs[u], winners, B)
            else:
                A += grow_candidates_LR(u, winners, l_idx[u], r_idx[u])

        # 4) 学习打分：用 scorer 替换启发式 delta/Rbar
        logits = []
        for a in A:
            feat = build_action_features(a, se_obs, F, Rbar, l_idx, r_idx, k_assigned, harq_state)
            logits.append(Scorer(feat))  # f_theta

        # 5) 选择并硬更新（保证 C3/C4）
        a_star = A[argmax(logits)]      # 推理：argmax；训练：softmax 采样
        apply_assign(a_star, winners, l_idx, r_idx, k_assigned)

    return winners, l_idx, r_idx
```

---

## 4. 训练策略（面向“强于启发式”的论文目标）

离散动作的端到端无监督（直接最大化 goodput）方差大、收敛慢。为了稳定获得**超过贪心启发式**的结果，推荐“两阶段训练”：

### 4.1 阶段 A：Imitation Learning（从更强 Teacher 蒸馏）

构造一个比当前贪心更强的 teacher，用于生成动作序列标签：

- 在相同观测 $\hat{\mathbf{s}}_t$ 下，用 **beam search / 多随机重启 / 局部交换改进** 搜索更优序列（严格满足 C3/C4/HARQ/PRB 上限）。
- 用与仓库一致的 reward 做序列评估：
  - 快速版：`EESM + MCS` 的吞吐 surrogate；
  - 完整版：包含 HARQ goodput（更贴近最终指标）。

学生网络最小化交叉熵：
\[
\mathcal{L}_{\text{IL}}=-\sum_{l=1}^{L}\log \pi_\theta(a^{(l)}_{\text{teacher}}|\mathrm{state}^{(l)})
\]

优势：收敛稳定、样本效率高，并且可在论文中自然解释“为何能系统性超越贪心”（因为 teacher 本身就是“非贪心更强基线”）。

### 4.2 阶段 B：RL Fine‑tuning（对齐最终 goodput）

在 imitation 初始化基础上，用策略梯度/actor‑critic 直接优化最终指标：

- 回报 $G$：可选总 goodput、或 PF 加权 goodput（与 PF 目标一致）；
- 加入 entropy 正则提升探索、降低局部最优风险；
- 训练时覆盖地图误差/blur、CSI delay、flicker、orbit dynamics 等扰动（对应仓库已有 config），增强泛化鲁棒性。

### 4.3 论文可写的消融点（建议）

- 无 MapEnc（仅点值 $\hat{s}$）vs 有 MapEnc（纹理特征）。
- 启发式 PF metric vs MADU‑Net learned scorer。
- imitation only vs imitation + RL。
- 不同 Top‑$B$ 与展开层数 $L$（复杂度‑性能折中）。

---

## 5. 推理与集成（与仓库评估接口一致）

建议的集成方式是把 MADU‑Net 作为 `pf_schedule_radiomap_blocks` 的“可学习动作打分器”替换件：

- 输入复用调度器已有的 `se_pred_k1 / se_metric_time[t]`、`Rbar`、HARQ 状态；
- 输出仍是 `winners` 与块边界（与当前记录/可视化接口一致）；
- 其余吞吐、MCS、HARQ 评估全部复用现有实现；
- 用 `run_test.py` 的多场景基准与现有启发式做对比即可形成论文实验。

---

## 6. 论文贡献表述（推荐写法）

1. **CQI‑free 的地图感知深度展开块调度**：在不依赖 UE CQI 的前提下，利用 radiomap 频域纹理提升连续 PRB 块的选择质量。
2. **结构化可行性保证**：通过“展开 + 硬更新”天然满足 OFDMA 正交与单块连续约束，无需软正交/TV 去近似组合约束。
3. **超越贪心的训练范式**：以“更强 teacher 搜索”蒸馏 + RL goodput 微调，使网络系统性超过现有启发式，并在地图误差/CSI delay 下保持鲁棒。
