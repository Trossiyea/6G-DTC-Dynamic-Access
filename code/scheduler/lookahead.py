"""
NTN 轨道感知前瞻调度模块 (OALS - Orbit-Aware Lookahead Scheduling)

专利核心算法实现：
1. 前瞻因子计算 (LookaheadFactor)
2. 度量修正函数 (compute_correction_factor)
3. 前瞻调度器 (OALSScheduler)

核心创新：利用 NTN 卫星轨道的完全可预测性，实现调度时机优化。
- 对于非实时业务，等待最优时刻调度
- 对于实时业务，紧急时立即调度
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============== 配置类 ==============

@dataclass
class OALSConfig:
    """OALS 算法配置参数

    Attributes:
        lookahead_horizon_ttis: 前瞻窗口长度 (TTI 数)
        lookahead_sample_interval: 稀疏采样间隔 (减少计算量)
        alpha_urgent: 紧急阈值，urgency > alpha 时立即调度
        beta_wait: 可等待阈值，urgency < beta 且 Φ < theta 时延迟调度
        theta_lookahead: 前瞻触发阈值
        gamma_decay: 衰减指数，控制等待惩罚强度
        boost_factor: 紧急提升因子
        enable_trend_correction: 是否启用趋势修正
        trend_threshold_db: 趋势判断阈值 (dB/TTI)
        trend_epsilon: 趋势修正幅度
        handover_prep_ttis: 切换准备窗口
        handover_hyst_db: 切换滞后门限
        harq_sinr_aggressive_db: 激进 MCS 的 SINR 偏移
        harq_sinr_conservative_db: 保守 MCS 的 SINR 偏移
        harq_delta_threshold_db: HARQ 前瞻 SINR 变化阈值
    """
    # 前瞻窗口
    lookahead_horizon_ttis: int = 200
    lookahead_sample_interval: int = 5
    lookahead_update_interval: int = 10  # 每隔多少 TTI 更新一次预测缓存

    # 度量修正参数
    alpha_urgent: float = 0.8
    beta_wait: float = 0.3
    theta_lookahead: float = 0.7
    gamma_decay: float = 2.0
    boost_factor: float = 2.0

    # 趋势修正
    enable_trend_correction: bool = True
    trend_threshold_db: float = 0.5
    trend_epsilon: float = 0.1

    # 切换预测
    handover_prep_ttis: int = 100
    handover_hyst_db: float = 3.0

    # HARQ 前瞻
    harq_sinr_aggressive_db: float = 1.5
    harq_sinr_conservative_db: float = 1.5
    harq_delta_threshold_db: float = 3.0

    # 调试
    enable_logging: bool = False

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> 'OALSConfig':
        """从配置字典创建 OALSConfig 实例"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)


# ============== 前瞻因子计算 ==============

@dataclass
class LookaheadCache:
    """前瞻预测缓存"""
    values: np.ndarray          # [N_UE, n_samples] 卫星链路增益
    sample_ttis: List[int]      # 采样时刻列表
    base_tti: int               # 起始 TTI
    n_ue: int
    n_samples: int


class LookaheadFactor:
    """前瞻因子计算器

    利用轨道预测计算每个 UE 在当前时刻相对于未来窗口内最优时刻的前瞻因子 Φ。

    Φ ∈ [0, 1]:
    - Φ = 1: 当前是窗口内链路最优时刻
    - Φ = 0: 当前是窗口内链路最差时刻
    """

    def __init__(self, orbit_model, config: OALSConfig):
        """
        Args:
            orbit_model: OrbitModel 或 ConstellationOrbit 实例
            config: OALS 配置
        """
        self.orbit = orbit_model
        self.cfg = config

        # 缓存
        self._cache: Optional[LookaheadCache] = None
        self._last_update_tti: int = -1

    @property
    def is_cached(self) -> bool:
        """检查缓存是否有效"""
        return self._cache is not None

    def needs_update(self, t: int) -> bool:
        """检查是否需要更新缓存"""
        if self._cache is None:
            return True
        return (t - self._last_update_tti) >= self.cfg.lookahead_update_interval

    def update(self, ue_pos: np.ndarray, t: int, sat_idx: Optional[int] = None) -> None:
        """更新前瞻预测缓存

        Args:
            ue_pos: UE 位置 [N_UE, 2]
            t: 当前 TTI
            sat_idx: 卫星索引 (用于星座模式，单星模式为 None)
        """
        H = self.cfg.lookahead_horizon_ttis
        interval = self.cfg.lookahead_sample_interval
        n_ue = ue_pos.shape[0]

        # 采样点
        sample_ttis = list(range(t, t + H, interval))
        n_samples = len(sample_ttis)

        # 批量预测卫星几何
        g_sat = np.zeros((n_ue, n_samples))

        for i, tau in enumerate(sample_ttis):
            if sat_idx is not None and hasattr(self.orbit, 'geometry_for_sat'):
                # 星座模式
                L_fs, G_rx, _, _, _ = self.orbit.geometry_for_sat(ue_pos, sat_idx, tau)
            else:
                # 单星模式
                L_fs, G_rx, _, _, _ = self.orbit.get_geometry(ue_pos, t=tau)

            g_sat[:, i] = G_rx - L_fs  # 卫星链路增益 (dB)

        # 更新缓存
        self._cache = LookaheadCache(
            values=g_sat,
            sample_ttis=sample_ttis,
            base_tti=t,
            n_ue=n_ue,
            n_samples=n_samples,
        )
        self._last_update_tti = t

    def compute_phi(self, ue_id: int) -> float:
        """计算单个 UE 的前瞻因子 Φ

        公式: Φ = (G_sat(t) - G_min) / (G_max - G_min)

        Args:
            ue_id: UE 索引

        Returns:
            Φ ∈ [0, 1], 1 表示当前是窗口内最优时刻
        """
        if self._cache is None:
            return 1.0  # 无缓存时默认返回 1 (立即调度)

        g_sat = self._cache.values[ue_id]
        g_current = g_sat[0]  # 当前时刻
        g_max = np.max(g_sat)
        g_min = np.min(g_sat)

        if g_max - g_min < 1e-6:
            return 1.0  # 无变化时返回 1

        return float((g_current - g_min) / (g_max - g_min))

    def compute_phi_batch(self) -> np.ndarray:
        """批量计算所有 UE 的前瞻因子

        Returns:
            Φ [N_UE] 数组
        """
        if self._cache is None:
            return np.ones(1)  # 无缓存时返回全 1

        g_sat = self._cache.values  # [N_UE, n_samples]
        g_current = g_sat[:, 0]
        g_max = np.max(g_sat, axis=1)
        g_min = np.min(g_sat, axis=1)

        denom = np.maximum(g_max - g_min, 1e-6)
        return (g_current - g_min) / denom

    def compute_trend(self, ue_id: int) -> float:
        """计算信道变化趋势 Δ

        公式: Δ ≈ (G_sat[t+1] - G_sat[t]) / interval

        Args:
            ue_id: UE 索引

        Returns:
            Δ (dB/TTI), >0 表示改善, <0 表示恶化
        """
        if self._cache is None or self._cache.n_samples < 2:
            return 0.0

        g_sat = self._cache.values[ue_id]
        interval = self.cfg.lookahead_sample_interval

        return float((g_sat[1] - g_sat[0]) / interval)

    def compute_trend_batch(self) -> np.ndarray:
        """批量计算所有 UE 的变化趋势

        Returns:
            Δ [N_UE] 数组 (dB/TTI)
        """
        if self._cache is None or self._cache.n_samples < 2:
            return np.zeros(1)

        g_sat = self._cache.values
        interval = self.cfg.lookahead_sample_interval

        return (g_sat[:, 1] - g_sat[:, 0]) / interval

    def predict_best_time(self, ue_id: int) -> Tuple[int, float]:
        """预测最优调度时刻

        Args:
            ue_id: UE 索引

        Returns:
            (t*, G_sat_max): 最优时刻 TTI 和对应的链路增益 (dB)
        """
        if self._cache is None:
            return 0, 0.0

        g_sat = self._cache.values[ue_id]
        sample_ttis = self._cache.sample_ttis

        best_idx = int(np.argmax(g_sat))
        return sample_ttis[best_idx], float(g_sat[best_idx])

    def get_g_sat_at_time(self, ue_id: int, t_offset: int) -> float:
        """获取指定时刻的预测 G_sat

        Args:
            ue_id: UE 索引
            t_offset: 相对于 base_tti 的偏移 (TTI 数)

        Returns:
            G_sat (dB)
        """
        if self._cache is None:
            return 0.0

        interval = self.cfg.lookahead_sample_interval
        idx = min(t_offset // interval, self._cache.n_samples - 1)
        idx = max(0, idx)

        return float(self._cache.values[ue_id, idx])

    def get_current_g_sat(self, ue_id: int) -> float:
        """获取当前时刻的 G_sat"""
        if self._cache is None:
            return 0.0
        return float(self._cache.values[ue_id, 0])


# ============== 度量修正函数 ==============

def compute_correction_factor(
    phi: float,
    urgency: float,
    trend: float,
    config: OALSConfig
) -> float:
    """计算度量修正因子 f(Φ, u, Δ)

    分段函数:
    - urgency > α: f = 1 + boost × (u - α) / (1 - α)  (紧急区)
    - urgency < β 且 Φ < θ: f = Φ^γ                    (可等待区)
    - 其他: f = 1                                      (正常区)

    Args:
        phi: 前瞻因子 ∈ [0, 1]
        urgency: 紧迫度 ∈ [0, 1+]
        trend: 变化趋势 (dB/TTI)
        config: OALS 配置

    Returns:
        修正因子 f
    """
    alpha = config.alpha_urgent
    beta = config.beta_wait
    theta = config.theta_lookahead
    gamma = config.gamma_decay
    boost = config.boost_factor

    # 基础修正
    if urgency > alpha:
        # 紧急区: 提升优先级
        f = 1.0 + boost * (urgency - alpha) / max(1.0 - alpha, 1e-6)
    elif urgency < beta and phi < theta:
        # 可等待区: 降低优先级
        f = max(phi, 1e-6) ** gamma
    else:
        # 正常区
        f = 1.0

    # 趋势修正
    if config.enable_trend_correction:
        eps = config.trend_epsilon
        th = config.trend_threshold_db

        if trend > th:
            f *= (1.0 - eps)  # 正在改善，略降优先级
        elif trend < -th:
            f *= (1.0 + eps)  # 正在恶化，略升优先级

    return f


def compute_correction_factor_batch(
    phi: np.ndarray,
    urgency: np.ndarray,
    trend: np.ndarray,
    config: OALSConfig
) -> np.ndarray:
    """批量计算修正因子 (向量化版本)

    Args:
        phi: 前瞻因子 [N_UE]
        urgency: 紧迫度 [N_UE]
        trend: 变化趋势 [N_UE]
        config: OALS 配置

    Returns:
        修正因子 [N_UE]
    """
    alpha = config.alpha_urgent
    beta = config.beta_wait
    theta = config.theta_lookahead
    gamma = config.gamma_decay
    boost = config.boost_factor

    n_ue = len(phi)
    f = np.ones(n_ue)

    # 紧急区
    urgent_mask = urgency > alpha
    if np.any(urgent_mask):
        f[urgent_mask] = 1.0 + boost * (urgency[urgent_mask] - alpha) / max(1.0 - alpha, 1e-6)

    # 可等待区
    wait_mask = (urgency < beta) & (phi < theta)
    if np.any(wait_mask):
        f[wait_mask] = np.maximum(phi[wait_mask], 1e-6) ** gamma

    # 趋势修正
    if config.enable_trend_correction:
        eps = config.trend_epsilon
        th = config.trend_threshold_db

        improving = trend > th
        degrading = trend < -th

        f[improving] *= (1.0 - eps)
        f[degrading] *= (1.0 + eps)

    return f


# ============== 前瞻调度器 ==============

@dataclass
class SchedulingRecommendation:
    """调度建议"""
    ue_id: int
    phi: float                      # 前瞻因子
    urgency: float                  # 紧迫度
    trend: float                    # 变化趋势
    correction: float               # 修正因子
    best_time: int                  # 预测最优时刻
    recommendation: str             # 'schedule_now' / 'delay' / 'urgent'
    g_sat_current: float           # 当前 G_sat
    g_sat_best: float              # 最优 G_sat


class OALSScheduler:
    """轨道感知前瞻调度器

    核心功能：
    1. 维护前瞻因子计算器
    2. 根据业务紧迫度和前瞻因子修正调度度量
    3. 提供调度建议
    """

    def __init__(
        self,
        n_ue: int,
        orbit_model,
        config: Optional[OALSConfig] = None
    ):
        """
        Args:
            n_ue: UE 数量
            orbit_model: 轨道模型实例
            config: OALS 配置，None 时使用默认值
        """
        self.n_ue = n_ue
        self.orbit = orbit_model
        self.cfg = config or OALSConfig()

        self.lookahead = LookaheadFactor(orbit_model, self.cfg)

        # 状态
        self.current_tti = 0
        self.urgency = np.zeros(n_ue)  # 由外部更新

        # 统计
        self._stats = {
            'updates': 0,
            'schedule_now': 0,
            'delay': 0,
            'urgent': 0,
        }

    def update_lookahead(
        self,
        ue_pos: np.ndarray,
        t: int,
        sat_idx: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """更新前瞻预测

        Args:
            ue_pos: UE 位置 [N_UE, 2]
            t: 当前 TTI
            sat_idx: 卫星索引 (星座模式)
            force: 强制更新

        Returns:
            是否执行了更新
        """
        self.current_tti = t

        if force or self.lookahead.needs_update(t):
            self.lookahead.update(ue_pos, t, sat_idx)
            self._stats['updates'] += 1
            return True

        return False

    def set_urgency(self, urgency: np.ndarray) -> None:
        """设置业务紧迫度

        Args:
            urgency: 紧迫度 [N_UE]，范围 [0, 1+]
        """
        self.urgency = np.asarray(urgency).flatten()

    def set_urgency_from_buffer(
        self,
        hol_delay_ms: np.ndarray,
        pdb_ms: np.ndarray
    ) -> None:
        """从缓冲区延迟计算紧迫度

        公式: urgency = hol_delay / pdb

        Args:
            hol_delay_ms: 队首延迟 [N_UE] (ms)
            pdb_ms: 延迟预算 [N_UE] (ms)
        """
        self.urgency = hol_delay_ms / np.maximum(pdb_ms, 1e-6)

    def compute_oals_metric(
        self,
        pf_metric: np.ndarray,
        qos_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算 OALS 调度度量

        公式: M = PF × W_qos × f(Φ, urgency)

        Args:
            pf_metric: PF 度量 [N_UE] 或 [N_UE, Z]
            qos_weight: QoS 权重 [N_UE]，None 时使用全 1

        Returns:
            OALS 度量，形状与 pf_metric 相同
        """
        n_ue = pf_metric.shape[0]

        # 默认 QoS 权重
        if qos_weight is None:
            qos_weight = np.ones(n_ue)

        # 确保 urgency 维度正确
        if len(self.urgency) != n_ue:
            self.urgency = np.zeros(n_ue)

        # 计算前瞻因子
        phi = self.lookahead.compute_phi_batch()
        if len(phi) != n_ue:
            phi = np.ones(n_ue)

        # 计算趋势
        trend = self.lookahead.compute_trend_batch()
        if len(trend) != n_ue:
            trend = np.zeros(n_ue)

        # 计算修正因子
        correction = compute_correction_factor_batch(
            phi, self.urgency, trend, self.cfg
        )

        # 应用修正
        if pf_metric.ndim == 1:
            return pf_metric * qos_weight * correction
        else:
            # per-PRB 度量
            return pf_metric * qos_weight[:, np.newaxis] * correction[:, np.newaxis]

    def get_scheduling_recommendation(self, ue_id: int) -> SchedulingRecommendation:
        """获取单个 UE 的调度建议

        Args:
            ue_id: UE 索引

        Returns:
            SchedulingRecommendation
        """
        phi = self.lookahead.compute_phi(ue_id)
        trend = self.lookahead.compute_trend(ue_id)
        urgency = self.urgency[ue_id] if ue_id < len(self.urgency) else 0.0
        best_time, g_sat_best = self.lookahead.predict_best_time(ue_id)
        g_sat_current = self.lookahead.get_current_g_sat(ue_id)

        correction = compute_correction_factor(phi, urgency, trend, self.cfg)

        # 生成建议
        if urgency > self.cfg.alpha_urgent:
            recommendation = 'urgent'
            self._stats['urgent'] += 1
        elif urgency < self.cfg.beta_wait and phi < self.cfg.theta_lookahead:
            recommendation = 'delay'
            self._stats['delay'] += 1
        else:
            recommendation = 'schedule_now'
            self._stats['schedule_now'] += 1

        return SchedulingRecommendation(
            ue_id=ue_id,
            phi=phi,
            urgency=urgency,
            trend=trend,
            correction=correction,
            best_time=best_time,
            recommendation=recommendation,
            g_sat_current=g_sat_current,
            g_sat_best=g_sat_best,
        )

    def get_all_recommendations(self) -> List[SchedulingRecommendation]:
        """获取所有 UE 的调度建议"""
        return [self.get_scheduling_recommendation(u) for u in range(self.n_ue)]

    def get_phi_array(self) -> np.ndarray:
        """获取所有 UE 的前瞻因子"""
        return self.lookahead.compute_phi_batch()

    def get_trend_array(self) -> np.ndarray:
        """获取所有 UE 的变化趋势"""
        return self.lookahead.compute_trend_batch()

    def get_statistics(self) -> Dict[str, int]:
        """获取调度统计"""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """重置统计"""
        self._stats = {
            'updates': 0,
            'schedule_now': 0,
            'delay': 0,
            'urgent': 0,
        }


# ============== 工厂函数 ==============

def create_oals_scheduler(
    n_ue: int,
    orbit_model,
    config: Optional[Dict[str, Any]] = None
) -> OALSScheduler:
    """创建 OALS 调度器

    Args:
        n_ue: UE 数量
        orbit_model: 轨道模型实例
        config: 配置字典，None 时使用默认值

    Returns:
        OALSScheduler 实例
    """
    oals_config = OALSConfig.from_config_dict(config) if config else OALSConfig()
    return OALSScheduler(n_ue, orbit_model, oals_config)


# ============== 导出 ==============

__all__ = [
    'OALSConfig',
    'LookaheadFactor',
    'LookaheadCache',
    'OALSScheduler',
    'SchedulingRecommendation',
    'compute_correction_factor',
    'compute_correction_factor_batch',
    'create_oals_scheduler',
]
