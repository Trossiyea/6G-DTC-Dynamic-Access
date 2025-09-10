1) 现实性增强：把“NR-NTN 的坑”都模拟起来

目标：在不把模型复杂度炸穿的前提下，引入对结果最敏感的 NTN 要素。

A. 轨道与波束几何（决定多普勒与时延）
	•	轨道层：加一个 orbit.py（SGP4 或简化近圆轨道），按 TTI 推进卫星位置与速度 r_s(t), v_s(t)；给每个 UE 求视线向量与仰角。由此得到
	•	单向传播时延 τ(t) = ||r_s- r_ue|| / c（LEO 量级 2–7 ms，GEO 量级百毫秒），这会推高 ACK/CSI 时序。
	•	多普勒 f_d(t) = (v_rel/c)·f_c，S 波段（~2 GHz）量级可达数十 kHz，必须考虑预补偿后的残差。
	•	波束层：为每颗星配置多点波束（earth-fixed 或 satellite-fixed），用 G(θ)=G0·cos^m θ + 附带旁瓣/边缘衰落；UE 随时间经历波束内渐变和波束切换。
实现钩子：你的 simple_beam_gain_db 已有雏形，扩成“多波束 + 波束指向随时间变化”，在 compute_caps 前注入 G_rx_db(ue,t)。

B. 频率/时间同步误差与预补偿
	•	频率预补偿（gNB/UE 基于星历与 GNSS）：按 3GPP 的做法，在网侧/终端做多普勒预补偿与TA 预补偿，但留**残差 ε_f（Hz）与 ε_τ（ns）**进入物理层；在你现有链路预算里，把残差映射为
	•	EVM/ICI 损失：对 OFDM，残差频偏约化成 SNR 折损 Δγ ≈ (2π·ε_f·T_sym)^2 的一阶近似；
	•	定时偏差：转成 CP 超限概率或额外符间干扰罚分。
依据：NTN 规范要求采用 GNSS/星历辅助进行定时与频率预补偿（含 Additional TA/ATA 机制），并允许较长时延场景下的特殊时序。

C. RTT 导致的 HARQ/反馈时序
	•	HARQ-ACK 延迟/延后（deferral）：把 ACK 期望时刻按 τ_UL+τ_DL+处理时延平移，并实现ACK 延后与可能禁用 HARQ、改走 RLC ARQ两种选项（LEO 倾向“增加进程数/延后”，GEO 倾向“禁 HARQ”）。
	•	CSI/CQI 时滞：引入 Δ_csi（几个到几十 ms）→ 调度使用过期的 CQI；这点在高速多普勒+波束移动下非常要命。
实现钩子：在你的 pf_schedule_* 里读取 cqi[t-Δ_csi]；在链路自适应里用 OLLA 把目标 BLER 锁到 10% 左右（见 38.214 MCS/BLER 流程）。

D. 上行功率控制（Fractional-PC）
	•	用 NR 规范化公式计算 PUSCH 发射功率：
P_PUSCH(dBm) = min(P_CMAX, 10·log10(M·12) + P0 + α·PL + Δ_TF + f(ΔTPC) + …)
其中 α∈{0,0.4,0.5,0.6,0.7,0.8,0.9,1}，P0 基于小区；把它作为 compute_caps 的输入功率而不是常数 23 dBm。
实现钩子：写一个 pusch_power_control(ue,t)，循环闭环（ΔTPC）每 TTI 更新。

E. 物理层细节（更贴近“手机”）
	•	UL 波形：选择 DFT-s-OFDM（PAPR 低、终端友好），保留 CP-OFDM 作为对照。
	•	MCS/CQI：改用 38.214 的标准 MCS 表（64QAM/256QAM/低 SE 三套）；SINR→CQI 的映射非规范性，采用 AWGN 目标 BLER 10% 的行业通用映射即可，并用 OLLA 收敛。
	•	频段/数值：默认选 n256 S-band（UL 1980–2010 MHz / DL 2170–2200 MHz），SCS 15/30 kHz 可切换；UE 最大发射功率与 RF 要求参考 38.101-5（Satellite access 专篇）。

F. 阻塞/体遮挡与环境损耗
	•	加入人体遮挡（1–6 dB）、树叶/室内穿透（>5 dB）、低仰角额外损耗（随仰角线性或分段增加），以及 S-band 的小雨衰可忽略、暴雨衰轻微（可选）。
实现钩子：在 compute_caps 里把这些做成独立的随机项或仰角的函数项 L_extra(θ, env)。

G. 干扰模型从“噪声地图”到“星座”
	•	现有 Radio Map 只建同频外部干扰，可新增“相邻波束泄漏/同星座邻星”两类干扰，并让其随时间（轨道/波束切换）缓慢漂移；这比纯静态地图更接近实际星座复用。

以上 A–G 都能直接挂接到你现有的 compute_caps / pf_schedule_* / run_* 框架里：把“几何/时序/功控/反馈”的状态在每个 TTI 更新，再把残差频偏/时偏转成 SNR/EVM 罚分，最后走规范化的 MCS/BLER/调度。

⸻

2) “最贴 3GPP 的 baseline”怎么设？

3GPP 不规定“要用 PF/MT/… 哪种调度算法”，但严格规定物理/控制流程、MCS 表、功控、时序等。下面给出一套可执行的 baseline 清单，把你模拟的“基线”尽量做成“规范流程的简化实现”。

(a) 体系与频段
	•	NR-NTN，透明或再生架构二选一；频段 n256（S-band） 或 n255（L-band）；FR1-NTN 的 UE RF/性能要求走 TS 38.101-5。
	•	Numerology：SCS 15 kHz 或 30 kHz（两档切换用来对比多普勒残差鲁棒性），常规 CP。

(b) 上行物理信道与功控
	•	UL 波形：PUSCH 选 DFT-s-OFDM（标准允许 CP-OFDM/DFT-s-OFDM；手机上行更偏向后者）。
	•	Fractional Power Control：严格按 TS 38.213 的 PUSCH 公式（含 P0, α, ΔTPC），α 取 0.8/1.0 做两档。

(c) 定时与频率同步（NTN 特化）
	•	Additional Timing Advance (ATA)：按 TS 38.211 的 NTN TA 机制，对 UL 发射做预补偿；并允许网侧在 RA 后下发/更新 TA。
	•	频率预补偿：按 TR 38.821 的方案使用星历/GNSS 做频偏预补偿；baseline 中保留一个 残差频偏 ε_f（例如 200–1000 Hz）参数化。

(d) 链路自适应与 MCS
	•	MCS 表：用 TS 38.214 的三套表（Table 5.1.3.1-1/-2/-3：64QAM/256QAM/低 SE）。首选 64QAM 表作为手机直连卫星默认。
	•	CQI/CSI：CQI 值→MCS 的映射遵循 38.214 的定义，但 SINR→CQI 属于厂商实现，baseline 采用 AWGN 目标 BLER≈10% 的常用曲线 + OLLA（外环 0.1 dB 步长）。
	•	CSI 时滞：设置 Δ_csi 与 Δ_sched（DCI 下发到生效的时差），按真实 RTT 调大。

(e) HARQ/时序
	•	ACK 延后（deferral）：按照 TS 38.214 的 HARQ-ACK deferral 描述，计算 ACK 期望时刻；LEO 场景把 HARQ 进程数增到 16 并启用 deferral，GEO 场景提供禁用 HARQ、走 RLC的开关。
	•	RLC/Timer：把重传/丢包相关的 RLC 定时器适当放宽以覆盖较大 RTT（规范在 38.300/高层定义，baseline 用放大系数实现）。

(f) 调度“基线”
	•	调度算法本身不被 3GPP 规定，因此“最合规的 baseline”应体现在输入/反馈/限制条件合规：
	1.	使用标准化的 CQI/MCS/功控；
	2.	遵循 NTN 时序（ACK 延后、CSI 时滞、TA 预补偿）；
	3.	资源分配遵循 TS 38.214 的资源映射与 TBS 计算；
	4.	UL 端功率/PRB 选择受 P_CMAX 与 α 约束。
	•	你之前的“宽带-PF vs PRB-PF”对比仍可以保留，但请把链路自适应、反馈时滞与功控都按上面规范“喂进去”，这样这个 baseline 就是“3GPP 流程 + 任意调度”的标准化对照。

(g) 频段与 UE 约束
	•	频段/带宽：n255/n256（FR1-NTN），支持 5/10/15/20/30 MHz 等；UE 发射功率/EIRP、带外指标、灵敏度等参照 TS 38.101-5。

⸻

放到你代码里的“最短改造路径”
	•	新模块：orbit.py（星历/多普勒/时延）、pc.py（PUSCH Fractional-PC）、harq.py（ACK deferral/进程管理）、csi.py（CQI 计算 + OLLA + 时滞）。
	•	现有接口最小侵入：
	•	compute_caps(...) → 增加 (t) 与 residuals={eps_f, eps_tau}；把 P_tx_dbm 改成来自 pc.pusch_power_control(ue,t)；
	•	pf_schedule_* → 读取 csi.get_cqi(ue, t-Δ_csi)；
	•	run_once/ run_many → 初始化轨道与 HARQ 状态机，并在每 TTI 更新 τ(t), f_d(t) 与 ACK 事件。

⸻

参考（规范为主，版本尽量取 R18/R17 的最新公开稿）
	•	物理信道/调制（含 NTN TA/时序基础）：3GPP TS 38.211（ETSI TS 138 211 v18.x）。
	•	控制层过程（功控、DCI、TA 细节）：3GPP TS 38.213（ETSI TS 138 213 v18.x / v17.x）。
	•	数据信道过程（MCS 表、HARQ-ACK 延后）：3GPP TS 38.214（ETSI TS 138 214 v18.x）。
	•	总体架构/NTN 适配：3GPP TS 38.300（Stage-2，总览与 NTN 架构）。
	•	UE RF/性能（卫星接入专篇）：3GPP TS 38.101-5（FR1-NTN，含 n255/n256 等）。
	•	NR-NTN 解决方案（GNSS/星历辅助、预补偿等）：3GPP TR 38.821。

