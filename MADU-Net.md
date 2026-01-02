# MADU-Net: Map-Aware Deep Unfolded Network (Simplified OFDMA Model)

## 1. 系统模型与问题定义 (System Model & Formulation)

我们考虑一个采用 **OFDMA** 下行链路的手机直连卫星（DtC）系统。由于子载波正交性，波束内不存在用户间干扰。主要干扰源为地面网络产生的背景干扰。

### 1.1 物理量定义

*   **集合：** 用户集 $\mathcal{U} = \{1, ..., U\}$，资源块（PRB）集 $\mathcal{K} = \{1, ..., K\}$。
*   **优化变量：** 功率分配矩阵 $\mathbf{P} \in \mathbb{R}^{U \times K}$，其中 $p_{u,k}$ 表示分配给用户 $u$ 在 PRB $k$ 上的功率。
*   **信道状态 (CSI):**
    *   $h_{u,k}$: 大尺度信道增益（包含路径损耗和阴影衰落）。由于小尺度衰落变化过快，调度主要依据大尺度特征。
*   **环境状态 (Environment):**
    *   $I_{map}(u,k)$: 从三维电磁地图 $\mathcal{M}$ 采样得到的地面干扰功率。
    *   $\sigma^2$: 热噪声功率（包含残留的波束间干扰）。

### 1.2 信干噪比 (SINR)

在单波束 OFDMA 模型中，SINR 的分母**不包含**其他用户的发射功率。
$$
\gamma_{u,k}(p_{u,k}) = \frac{h_{u,k} p_{u,k}}{I_{map}(u,k) + \sigma^2}
$$
*注：这是一个完全解耦的表达式，每个 RB 的质量仅取决于自身功率和环境。*

### 1.3 优化问题 (Optimization Problem)

目标是最大化加权和速率（Weighted Sum Rate），同时满足功率预算和连续性约束。

$$
\begin{aligned}
\max_{\mathbf{P}} \quad & J(\mathbf{P}) = \sum_{u=1}^{U} w_u \sum_{k=1}^{K} \log_2 \left( 1 + \gamma_{u,k}(p_{u,k}) \right) \\
\text{s.t.} \quad & \text{C1: } p_{u,k} \ge 0, \quad \forall u, k \\
& \text{C2: } \sum_{u=1}^{U} \sum_{k=1}^{K} p_{u,k} \le P_{total} \quad (\text{卫星总功率约束}) \\
& \text{C3: } \sum_{u=1}^{U} \mathbb{I}(p_{u,k} > 0) \le 1, \quad \forall k \quad (\text{RB正交性约束}) \\
& \text{C4: } \text{Adjacency Constraint} \quad (\text{RB连续性约束})
\end{aligned}
$$

*   **难点分析：** 尽管目标函数 $J(\mathbf{P})$ 是凸的，但 **C3 (整数约束)** 和 **C4 (组合约束)** 将该问题变成了 NP-hard 的混合整数非线性规划（MINLP）问题。传统的注水算法（Water-filling）无法处理 C4。

---

## 1.4 CQI-free 观测模型（与仿真实现对齐）

本项目的目标是 **取代 UE CQI 反馈**：调度器不依赖 UE 上报 CQI/CSI，而是基于网络侧可得信息进行资源分配与链路自适应。

- **Truth（用于环境/解码评估）**：$(H_{true}, I_{true})$ 由物理引擎产生（3GPP+TLE+radiomap，可包含 shadowing / fast fading）。
- **Observation（用于调度/MCS 选择）**：$(\hat{H}, \hat{I})$ 在网络侧构建：
  - $\hat{H}$：由 UE 位置 + 卫星几何/链路预算得到（例如 FSPL + antenna gain；不依赖 UE CQI）。
  - $\hat{I}$：由静态电磁地图（长期统计 radiomap）查询得到。

在代码中，CQI 老化/观测失配通过评估脚本 `python madu.py evaluate --config configs/env_s_band.yaml` 体现：`stale` / `geom(+map)` 使用不同的观测来做调度与 MCS 选择，并在真实信道下计算 **goodput + outage**。

---

## 2. MADU-Net 算法架构 (Algorithm Architecture)

MADU-Net 通过深度展开（Deep Unfolding）将**梯度上升法（Gradient Ascent）**与**卷积神经网络（CNN）**结合，利用神经网络的非线性能力来“软化”并解决 C3 和 C4 约束。

### 2.1 基础迭代逻辑 (The Backbone)

我们展开 **投影梯度上升 (Projected Gradient Ascent)** 算法。
目标函数的理论梯度（关于 $p_{u,k}$）为：
$$
\nabla_{u,k} = \frac{\partial J}{\partial p_{u,k}} = \frac{w_u}{(\ln 2)} \cdot \frac{h_{u,k}}{I_{map}(u,k) + \sigma^2 + h_{u,k} p_{u,k}}
$$
*物理意义：* 该梯度指示了在忽略连续性约束时，哪些 RB 性价比最高（信道好、干扰小）。

### 2.2 网络详细设计

网络输入为 $(I_{map}, \hat{\mathbf{H}}, \mathbf{w})$，输出为 $\mathbf{P}$。网络包含 **Map Encoder** 和 **$L$ 层 Unfolded Solver**。

#### **模块 A: 地图特征提取器 (Map Encoder)**

*   **功能：** 提取干扰的频域平滑度特征（识别哪些频段是成片干净的，适合连续分配）。
*   **输入：** $I_{map}$ 张量。
*   **结构：** 1D-CNN (沿频域卷积) 或 2D-CNN (如果考虑多波束空间关联)。
*   **输出：** 特征矩阵 $\mathbf{F} \in \mathbb{R}^{U \times K \times C}$。

#### **模块 B: 深度展开层 (Unfolded Layer $l$)**

第 $l$ 层的更新规则如下：

**1. 物理梯度计算 (Physics Awareness):**
$$ \mathbf{G}^{(l)} = \text{CalculateGradient}(\mathbf{P}^{(l)}, \mathbf{H}, I_{map}) $$
*这保留了物理模型的最优性方向。*

**2. 神经梯度修正 (AI Refinement):**
利用 CNN 修正梯度，使其倾向于选择连续的块。
$$ \Delta \mathbf{P}^{(l)} = \text{CNN}_{refine}\left( \text{Concat}[\mathbf{G}^{(l)}, \mathbf{P}^{(l)}, \mathbf{F}] \right) $$
$$ \hat{\mathbf{P}}^{(l)} = \mathbf{P}^{(l)} + \mu \cdot \Delta \mathbf{P}^{(l)} $$

**3. 连续性平滑 (Smoothness Enforcement - 处理 C4):**
在频域应用 1D 卷积（类似于平均滤波），迫使孤立的功率尖峰被抑制，相邻的功率被拉平。
$$ \tilde{\mathbf{P}}^{(l)} = \text{Conv1D}_{smooth}(\hat{\mathbf{P}}^{(l)}) $$

**4. 正交性与功率投影 (Orthogonality & Projection - 处理 C2, C3):**
这是 OFDMA 特有的步骤。我们需要保证一个 RB 主要分给一个用户。

*   **Softmax 操作 (Across Users):**
    $$ \mathbf{S}_{u,k} = \text{Softmax}_{\text{dim}=u}(\tilde{\mathbf{P}}^{(l)} \cdot \beta) $$
    *这会让某个 RB 上的功率分配趋向于 One-hot（即只给一个用户）。*
*   **总功率归一化:**
    $$ \mathbf{P}^{(l+1)} = \mathbf{S} \cdot P_{total} / \sum \mathbf{S} $$

---

## 3. 训练策略 (Training Strategy)

采用无监督训练。由于移除了用户间干扰，训练会更加稳定。

### 3.1 损失函数

$$ \mathcal{L} = \mathcal{L}_{rate} + \lambda_{cont} \mathcal{L}_{cont} + \lambda_{orth} \mathcal{L}_{orth} $$

1.  **Rate Loss:** $\mathcal{L}_{rate} = - \sum w_u \log(1 + \text{SINR})$
2.  **Continuity Loss (全变分 TV):**
    鼓励同一用户的功率谱是平滑的阶梯状。
    $$ \mathcal{L}_{cont} = \sum_u \sum_k | p_{u,k+1} - p_{u,k} |^2 $$
3.  **Orthogonality Loss (正交性惩罚):**
    虽然 Softmax 提供了软正交，但为了确保输出也是正交的，加入惩罚项：
    $$ \mathcal{L}_{orth} = \sum_k \left( (\sum_u p_{u,k})^2 - \sum_u (p_{u,k})^2 \right) $$
    *当且仅当每个 RB 只有一个用户有功率时，该项为 0。*

---

## 4. 算法流程伪代码

```python
def MADU_Net_Forward(I_map, H, w):
    # 1. 初始化
    # 均匀分配或基于比例公平权重的启发式初始化
    P = Initialize_Power(P_total, w) 
    
    # 2. 提取地图特征
    Map_Feats = Map_Encoder(I_map) # [Batch, U, K, 16]

    # 3. 迭代展开 (假设 10 层)
    for i in range(10):
        # A. 计算物理梯度 (Decoupled Gradient)
        # grad[u,k]只与该用户在该RB的状态有关，计算极快
        Num = w * H
        Denom = (I_map + Noise + H * P) * np.log(2)
        Grad = Num / Denom
        
        # B. 神经网络修正 (注入连续性偏好)
        # 输入: 梯度, 当前功率, 地图特征
        # 输出: 修正量 Delta_P
        Delta_P = ResNet_Block(cat([Grad, P, Map_Feats]))
        
        P_temp = P + learning_rate * Delta_P
        
        # C. 强制平滑 (1D Convolution along K dim)
        # kernel 如 [0.2, 0.6, 0.2]，平滑频域毛刺
        P_smooth = Conv1D(P_temp, kernel_size=3)
        
        # D. 投影 (Projection)
        # D1: 保证非负
        P_pos = ReLU(P_smooth)
        
        # D2: 软正交 (Softmax across users)
        # 温度系数 temp 越小，分配越趋向于独占
        Scores = Softmax(P_pos / temp, dim=Users)
        
        # D3: 恢复功率幅度并归一化
        Magnitude = Sum(P_pos, dim=Users)
        P = Scores * Magnitude
        P = P * (P_total / Sum(P))

    return P
```

## 5. 方案优势总结 (For Paper)

1.  **复杂问题简化求解：** 将包含离散约束（正交性）和组合约束（连续性）的 MINLP 问题，转化为一个端到端的平滑优化过程。
2.  **物理可解释性：** 网络每一层都在执行“梯度上升”，保证了算法是在试图最大化和速率，而不是黑盒盲猜。
3.  **电磁地图价值最大化：** 相比于传统算法只能看到当前的 $I_{map}$ 数值，MADU-Net 的 Encoder 能看到 $I_{map}$ 的**纹理特征**，从而自动避开那些“虽然当前点干扰低，但周围干扰高，不适合建立连续块”的区域。
