# Radio Map 调度算法的 SINR 计算详解

本文档详细说明 `code/main.py` 中基于 Radio Map 的调度算法如何计算每个 UE 在每个 PRB 上的 SINR。

## 概述

Radio Map 是一个三维张量 `R_xyz_dbm[X, Y, Z]`，表示在不同地理位置 (X, Y) 和不同频率资源块 (Z=PRB) 上接收到的**地面干扰功率**（单位：dBm）。

SINR 计算的核心思想是：
- **信号功率**：来自卫星的下行信号（考虑路径损耗、天线增益、信道衰落）
- **干扰+噪声功率**：Radio Map 中的地面干扰 + 热噪声

## 完整计算流程

### 第 1 步：加载 Radio Map

```python
# 位置：run_once() 函数
R_xyz_dbm, X, Y, Z = select_radio_map(config)
# R_xyz_dbm 维度: [X, Y, Z] = [空间x, 空间y, PRB编号]
# 单位: dBm（地面干扰功率）
```

**Radio Map 含义**：
- `R_xyz_dbm[x, y, z]` 表示在地理位置 (x, y) 的 PRB z 上，UE 接收到的**地面基站干扰功率**（dBm）
- 这是预先测量或仿真得到的，代表地面蜂窝网络的上行干扰

### 第 2 步：计算信号接收功率

在 `compute_caps()` 函数中（line 454-544）：

#### 2.1 基本链路预算

```python
# 发射功率（每 PRB）
P_tx_dbm = config["P_tx_dbm"]  # 例如: 30 dBm

# 自由空间路径损耗（FSPL）
L_fs_db = compute_geometry_and_beam(...)  # 基于距离和频率
# FSPL公式: 20*log10(d) + 20*log10(f) + 32.45
# d: 距离(km), f: 频率(MHz)

# 接收天线增益（波束形成增益）
G_rx_db = simple_beam_gain_db(...)  # 基于指向角度
# 通常: 38 dB（波束中心）到 35 dB（波束边缘）

# 大尺度衰落（3GPP NTN 信道模型）
large_scale_db, fading_lin = sample_3gpp_ntn_fading(
    rng, elevation_deg, Z, 
    profile_name="s_band_handheld_urban"
)
# large_scale_db: 包括阴影衰落、LoS/NLoS状态损耗
# fading_lin: 小尺度衰落（每PRB不同）

# 平均接收功率（dBm，不含小尺度衰落）
P_rx_dbm[ue] = P_tx - L_fs[ue] + G_rx[ue] + large_scale_db[ue]
```

#### 2.2 每 PRB 接收功率

```python
# 添加小尺度衰落（频率选择性）
# fading_lin[ue, z] 是线性功率增益
P_rx_prb_dbm[ue, z] = P_rx_dbm[ue] + 10*log10(fading_lin[ue, z])
```

**关键点**：
- `P_rx_dbm[ue]`: UE 的平均接收功率（标量）
- `fading_lin[ue, z]`: 每个 PRB 的小尺度衰落系数（线性，≈1.0附近）
- `P_rx_prb_dbm[ue, z]`: 每个 PRB 的实际接收功率

### 第 3 步：提取地面干扰功率

```python
# 根据 UE 位置提取 Radio Map 中的干扰
# ue_pos[ue] = [x_idx, y_idx]
I_uez_dbm[ue, z] = R_xyz_dbm[x_idx[ue], y_idx[ue], z]
# 维度: [N_UE, Z]
# 单位: dBm（地面干扰功率）
```

**说明**：
- 每个 UE 根据其地理位置 (x_idx, y_idx) 从 Radio Map 中提取对应的干扰功率
- 不同 PRB (z) 的干扰可能不同（频率选择性干扰）

### 第 4 步：计算总干扰+噪声功率

```python
# 热噪声功率（每PRB）
N0_dbm = thermal_noise_dbm(bw_hz=360e3, temp_K=290)
# 例如: -121.45 dBm（30 kHz SCS，12子载波 = 360 kHz）

# 接收机噪声系数和实现损耗
N0_eff_dbm = N0_dbm + rx_nf_db + impl_loss_db
# rx_nf_db: 接收机噪声系数，例如 7 dB
# impl_loss_db: 实现损耗，例如 1 dB
# N0_eff_dbm ≈ -113.45 dBm

# 总干扰+噪声（线性域相加，然后转回dB）
I_total_mw[ue, z] = 10^(I_uez_dbm[ue,z]/10) + 10^(N0_eff_dbm/10)
I_total_dbm[ue, z] = 10*log10(I_total_mw[ue, z])
```

**重要**：
- 干扰和噪声在**线性域**（mW）相加
- 然后转换回对数域（dBm）

### 第 5 步：计算 SINR

```python
# SINR（每UE每PRB）
gamma_db[ue, z] = P_rx_prb_dbm[ue, z] - I_total_dbm[ue, z]
snr_lin[ue, z] = 10^(gamma_db[ue, z] / 10)
```

**公式展开**：
```
SINR(dB) = 信号功率(dBm) - 干扰+噪声(dBm)
         = [P_tx - L_fs + G_rx + large_scale + 10*log10(fading)]
           - [10*log10(I_ground + N0)]

线性SINR = 10^(SINR(dB)/10)
```

### 第 6 步：计算频谱效率

```python
# Shannon容量（bits/s/Hz）
cap[ue, z] = log2(1 + snr_lin[ue, z])

# 或使用MCS表（更真实）
if use_mcs:
    sinr_db = 10*log10(snr_lin[ue, z])
    cap[ue, z] = sinr_to_se_mcs(sinr_db, table="3gpp_table_2")
```

## 完整公式总结

### Per-PRB SINR

```
信号功率 S[ue, z] (dBm) = 
    P_tx                    # 发射功率
    - L_fs[ue]              # 自由空间损耗
    + G_rx[ue]              # 接收天线增益
    + large_scale_db[ue]    # 大尺度衰落（阴影+LoS/NLoS）
    + 10*log10(fading_lin[ue, z])  # 小尺度衰落（瑞利/莱斯）

干扰+噪声功率 I[ue, z] (dBm) = 
    10*log10(
        10^(R_xyz_dbm[x[ue], y[ue], z] / 10)  # 地面干扰（线性）
        + 10^(N0_eff / 10)                     # 热噪声（线性）
    )

SINR[ue, z] (dB) = S[ue, z] - I[ue, z]

SINR[ue, z] (线性) = 10^(SINR[ue,z](dB) / 10)
```

### Wideband SINR（用于基线调度器）

```python
# 宽带干扰（所有PRB平均）
I_wb_mw[ue] = mean(I_total_mw[ue, :])  # 线性平均
I_wb_dbm[ue] = 10*log10(I_wb_mw[ue])

# 宽带信号（平均衰落）
mean_fading[ue] = mean(fading_lin[ue, :])
P_rx_wb_mw[ue] = 10^(P_rx_dbm[ue]/10) * mean_fading[ue]

# 宽带SINR
snr_lin_wb[ue] = P_rx_wb_mw[ue] / I_wb_mw[ue]
```

## Radio Map 的作用

Radio Map 在计算中的核心作用：

1. **空间感知**：不同位置的 UE 面临不同的地面干扰
   ```python
   # UE1在城市中心 -> 高干扰
   I_uez_dbm[ue1, z] = R_xyz_dbm[城市中心x, 城市中心y, z] = -50 dBm
   
   # UE2在郊区 -> 低干扰
   I_uez_dbm[ue2, z] = R_xyz_dbm[郊区x, 郊区y, z] = -80 dBm
   ```

2. **频率感知**：不同 PRB 的干扰不同
   ```python
   # 某些频段干扰高
   I_uez_dbm[ue, prb_10] = -55 dBm  # 高干扰PRB
   I_uez_dbm[ue, prb_30] = -75 dBm  # 低干扰PRB
   ```

3. **调度决策**：
   - **基线调度器**：不使用 per-PRB 信息，所有 PRB 视为相同
   - **Radio Map 调度器**：优先分配干扰小的 PRB 给 UE

## 代码流程图

```
┌─────────────────────────────────────────────────┐
│ 1. 加载 Radio Map: R_xyz_dbm[X, Y, Z]          │
│    (地面干扰功率 dBm)                            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 2. 生成 UE 位置: ue_pos[N_UE] = [(x,y), ...]   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 3. 计算几何参数（每UE）                         │
│    - L_fs[ue]: 自由空间损耗                     │
│    - G_rx[ue]: 接收天线增益                     │
│    - elevation[ue]: 仰角                        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 4. 3GPP NTN 信道建模（每UE每PRB）               │
│    - large_scale_db[ue]: 大尺度衰落             │
│    - fading_lin[ue, z]: 小尺度衰落              │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 5. 计算接收功率（每UE每PRB）                    │
│    P_rx_prb_dbm[ue,z] =                         │
│        P_tx - L_fs + G_rx + large_scale         │
│        + 10*log10(fading_lin[ue,z])             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 6. 从 Radio Map 提取干扰（每UE每PRB）           │
│    I_uez_dbm[ue,z] =                            │
│        R_xyz_dbm[x_idx[ue], y_idx[ue], z]       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 7. 计算总干扰+噪声（每UE每PRB）                 │
│    I_total_mw[ue,z] =                           │
│        10^(I_uez_dbm[ue,z]/10)                  │
│        + 10^(N0_eff/10)                         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 8. 计算 SINR（每UE每PRB）                       │
│    SINR_db[ue,z] =                              │
│        P_rx_prb_dbm[ue,z] - I_total_dbm[ue,z]   │
│    snr_lin[ue,z] = 10^(SINR_db[ue,z]/10)       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 9. 计算频谱效率（每UE每PRB）                    │
│    SE[ue,z] = log2(1 + snr_lin[ue,z])          │
│    或使用 MCS 表                                 │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ 10. Radio Map 调度器决策                        │
│     - 为每个 UE 选择最佳 PRB 组合               │
│     - 考虑 per-PRB SINR 差异                    │
│     - 使用连续 PRB 块分配                        │
└─────────────────────────────────────────────────┘
```

## 关键代码位置

### 主要函数

1. **`compute_caps()`** (line 454-544)
   - 核心 SINR 计算函数
   - 输入：Radio Map, UE位置, 链路参数
   - 输出：`snr_lin[UE, Z]`, `cap[UE, Z]`

2. **`run_once()`** (line 1360+)
   - 主仿真流程
   - 调用 `compute_caps()` 计算 SINR
   - 调用调度器进行资源分配

3. **`pf_schedule_radiomap_blocks()`** (line 725+)
   - Radio Map 感知调度器
   - 使用 per-PRB SINR 进行智能调度

### 关键代码片段

```python
# Line 523: 从 Radio Map 提取干扰
I_uez_dbm = R_xyz_dbm[x_idx, y_idx, :]  # [UE,Z]

# Line 525-527: 干扰+噪声（线性相加）
N0_eff_dbm = N0_dbm + rx_nf_db + impl_loss_db
I_total_mw = dbm_to_mw(I_uez_dbm) + dbm_to_mw(N0_eff_dbm)
I_total_dbm = mw_to_dbm(I_total_mw)

# Line 530-531: 计算 SINR
gamma_db = P_rx_prb_dbm - I_total_dbm    # [UE,Z]
snr_lin = 10.0 ** (gamma_db / 10.0)
```

## 数值示例

### 场景设置
- 卫星：轨道高度 600 km，发射功率 30 dBm/PRB
- UE1：城市中心，干扰 -50 dBm
- UE2：郊区，干扰 -80 dBm
- 频率：2.19 GHz（多伦多）
- 热噪声：-121.45 dBm

### UE1（城市中心）计算

```
# 信号功率
P_tx = 30 dBm
L_fs = 20*log10(600) + 20*log10(2190) + 32.45 = 175.3 dB
G_rx = 38 dB（波束中心）
large_scale = -5 dB（阴影衰落）
fading = 0 dB（平均）

P_rx_prb = 30 - 175.3 + 38 - 5 + 0 = -112.3 dBm

# 干扰+噪声
I_ground = -50 dBm = 0.00001 mW
N0_eff = -113.45 dBm = 0.00000045 mW
I_total = 0.00001 + 0.00000045 = 0.0000104 mW = -49.8 dBm

# SINR
SINR = -112.3 - (-49.8) = -62.5 dB
snr_lin = 10^(-62.5/10) = 5.6e-7 ≈ 0（很差！）

# 频谱效率
SE = log2(1 + 5.6e-7) ≈ 0 bits/s/Hz
```

**结论**：城市中心干扰过强，SINR非常低，几乎无法通信。

### UE2（郊区）计算

```
# 信号功率（相同）
P_rx_prb = -112.3 dBm

# 干扰+噪声
I_ground = -80 dBm = 0.0000001 mW
N0_eff = -113.45 dBm = 0.00000045 mW
I_total = 0.0000001 + 0.00000045 = 0.00000055 mW = -112.6 dBm

# SINR
SINR = -112.3 - (-112.6) = 0.3 dB
snr_lin = 10^(0.3/10) = 1.07

# 频谱效率
SE = log2(1 + 1.07) ≈ 1.05 bits/s/Hz
```

**结论**：郊区干扰低，SINR为正，可以传输数据。

## Radio Map vs Baseline 调度器对比

### Baseline 调度器
- 使用**宽带平均 SINR**：所有 PRB 视为相同
- 不考虑 per-PRB 干扰差异
- 简单快速，但次优

```python
# Baseline使用平均SINR
snr_wb[ue] = mean(P_rx / I_total) over all PRBs
```

### Radio Map 调度器
- 使用**per-PRB SINR**：每个 PRB 独立考虑
- 优先分配低干扰 PRB
- 更复杂，但性能更好

```python
# RadioMap使用per-PRB SINR
for each ue:
    # 找到最佳PRB
    best_prb = argmax(snr_lin[ue, :])
    # 分配连续块
    allocate_block(ue, best_prb, block_size)
```

**性能提升示例**：
- Baseline：所有 UE 平均分配所有 PRB → 城市UE得到差PRB
- RadioMap：城市UE得到相对较好的PRB → 12-15% 吞吐量提升

## 时变特性

Radio Map 支持时变（可选）：

```python
# 每个TTI更新Radio Map
if enable_time_varying:
    # 漂移（模拟UE移动）
    R_t = np.roll(R_xyz_dbm, shift=(vx, vy, 0), axis=(0,1,2))
    
    # 闪烁（模拟干扰波动）
    R_t += rng.normal(0, flicker_std, size=R_t.shape)
```

## 总结

Radio Map 调度算法的 SINR 计算是一个完整的链路预算过程：

1. ✅ **信号**：卫星发射 → 路径损耗 → 天线增益 → 信道衰落
2. ✅ **干扰**：Radio Map 提供地面干扰 + 热噪声
3. ✅ **SINR**：信号功率 / （干扰+噪声）功率
4. ✅ **调度**：利用 per-PRB SINR 差异优化资源分配

关键优势：
- 🎯 **空间感知**：不同位置不同干扰
- 🎯 **频率感知**：不同 PRB 不同干扰  
- 🎯 **智能调度**：优先分配好的 PRB

## 相关文件

- `code/main.py` - 主实现（SINR计算和调度）
- `code/ntn_channel.py` - 3GPP NTN 信道模型
- `code/orbit.py` - 几何参数和波束增益
- `radio_map/` - Radio Map 数据文件

## 参考公式

### 自由空间损耗（FSPL）
```
L_fs (dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.45
```

### 热噪声功率
```
N0 (dBm) = -174 + 10*log10(BW_Hz) + 10*log10(T/290)
```

### Shannon容量
```
C (bits/s/Hz) = log2(1 + SINR_linear)
```

### dBm ↔ mW 转换
```
P_mW = 10^(P_dBm / 10)
P_dBm = 10 * log10(P_mW)
```
