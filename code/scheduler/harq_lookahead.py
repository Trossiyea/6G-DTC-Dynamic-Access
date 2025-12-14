"""
HARQ 前瞻 MCS 选择模块

专利核心创新：
1. 基于轨道预测的前瞻性 MCS 选择
2. 考虑重传时刻的预测信道状态
3. 切换感知的 HARQ 策略

核心思想：
- 如果预测重传时信道更好 → 激进首传 MCS
- 如果预测重传时信道更差 → 保守首传 MCS
- 切换准备期 → 尽量完成传输，避免跨卫星重传
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .lookahead import LookaheadFactor


# ============== 数据类型定义 ==============

class MCSStrategy(Enum):
    """MCS 选择策略"""
    STANDARD = "standard"           # 标准选择
    AGGRESSIVE = "aggressive"       # 激进（预测未来更好）
    CONSERVATIVE = "conservative"   # 保守（预测未来更差）
    URGENT = "urgent"              # 紧急（切换准备期）


@dataclass
class MCSAdjustment:
    """MCS 调整建议"""
    ue_id: int
    sinr_adjustment_db: float       # SINR 调整量 (dB)
    strategy: MCSStrategy           # 选择策略
    rationale: str                  # 决策原因
    predicted_delta_db: float       # 预测的 SINR 变化 (dB)
    k1_slots: int                   # ACK 延迟
    confidence: float               # 决策置信度


@dataclass
class HARQLookaheadConfig:
    """HARQ 前瞻配置"""
    # MCS 调整参数
    aggressive_sinr_boost_db: float = 1.5       # 激进策略的 SINR 提升
    conservative_sinr_margin_db: float = 1.5    # 保守策略的 SINR 余量
    delta_threshold_db: float = 3.0             # SINR 变化阈值

    # 紧急模式参数
    urgent_sinr_margin_db: float = 2.0          # 紧急模式额外余量
    handover_prep_window_ttis: int = 100        # 切换准备窗口

    # 提前重传参数
    early_retx_trend_threshold: float = -1.0    # 趋势阈值 (dB/TTI)
    early_retx_max_rv: int = 2                  # 最大 RV 索引

    # 调试
    enable_logging: bool = False

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> 'HARQLookaheadConfig':
        """从配置字典创建"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)


# ============== HARQ 前瞻 MCS 选择器 ==============

class HARQLookaheadMCS:
    """HARQ 前瞻 MCS 选择器

    利用轨道预测信息优化 MCS 选择：
    1. 预测 ACK/NACK 时刻的信道状态
    2. 根据预测调整首传 MCS
    3. 在切换准备期采用保守策略
    """

    def __init__(
        self,
        lookahead_factor: 'LookaheadFactor',
        config: Optional[HARQLookaheadConfig] = None
    ):
        """
        Args:
            lookahead_factor: LookaheadFactor 实例（用于获取预测信息）
            config: HARQ 前瞻配置
        """
        self.lookahead = lookahead_factor
        self.cfg = config or HARQLookaheadConfig()

        # 状态
        self._handover_prep_ues: set = set()  # 处于切换准备期的 UE

        # 统计
        self._stats = {
            'standard': 0,
            'aggressive': 0,
            'conservative': 0,
            'urgent': 0,
        }

    def set_handover_prep_ues(self, ue_ids: set) -> None:
        """设置处于切换准备期的 UE

        Args:
            ue_ids: UE ID 集合
        """
        self._handover_prep_ues = set(ue_ids)

    def get_mcs_adjustment(
        self,
        ue_id: int,
        sinr_now_db: float,
        k1_slots: int,
        k2_slots: int = 4,
    ) -> MCSAdjustment:
        """获取 MCS 选择调整

        Args:
            ue_id: UE ID
            sinr_now_db: 当前 SINR (dB)
            k1_slots: ACK 延迟 (slots)
            k2_slots: 重传延迟 (slots)

        Returns:
            MCSAdjustment
        """
        # 检查是否在切换准备期
        if ue_id in self._handover_prep_ues:
            self._stats['urgent'] += 1
            return MCSAdjustment(
                ue_id=ue_id,
                sinr_adjustment_db=-self.cfg.urgent_sinr_margin_db,
                strategy=MCSStrategy.URGENT,
                rationale="Handover preparation - conservative MCS",
                predicted_delta_db=0.0,
                k1_slots=k1_slots,
                confidence=0.9,
            )

        # 预测 ACK 时刻的 G_sat
        t_ack = k1_slots + k2_slots  # 简化：重传可能发生的时刻

        g_sat_now = self.lookahead.get_current_g_sat(ue_id)
        g_sat_ack = self.lookahead.get_g_sat_at_time(ue_id, t_ack)

        delta_sinr = g_sat_ack - g_sat_now

        # 根据预测调整
        if delta_sinr > self.cfg.delta_threshold_db:
            # 预测重传时信道更好 → 激进首传
            self._stats['aggressive'] += 1
            return MCSAdjustment(
                ue_id=ue_id,
                sinr_adjustment_db=self.cfg.aggressive_sinr_boost_db,
                strategy=MCSStrategy.AGGRESSIVE,
                rationale=f"Future channel better by {delta_sinr:.1f} dB",
                predicted_delta_db=delta_sinr,
                k1_slots=k1_slots,
                confidence=0.7,
            )

        elif delta_sinr < -self.cfg.delta_threshold_db:
            # 预测重传时信道更差 → 保守首传
            self._stats['conservative'] += 1
            return MCSAdjustment(
                ue_id=ue_id,
                sinr_adjustment_db=-self.cfg.conservative_sinr_margin_db,
                strategy=MCSStrategy.CONSERVATIVE,
                rationale=f"Future channel worse by {abs(delta_sinr):.1f} dB",
                predicted_delta_db=delta_sinr,
                k1_slots=k1_slots,
                confidence=0.7,
            )

        else:
            # 信道稳定 → 标准选择
            self._stats['standard'] += 1
            return MCSAdjustment(
                ue_id=ue_id,
                sinr_adjustment_db=0.0,
                strategy=MCSStrategy.STANDARD,
                rationale="Channel stable",
                predicted_delta_db=delta_sinr,
                k1_slots=k1_slots,
                confidence=0.9,
            )

    def should_trigger_early_retx(
        self,
        ue_id: int,
        current_rv: int,
        sinr_accumulated_db: float,
    ) -> bool:
        """判断是否应提前触发重传

        在以下情况下触发提前重传：
        1. 信道正在快速恶化
        2. HARQ 未达到最大 RV
        3. UE 处于切换准备期

        Args:
            ue_id: UE ID
            current_rv: 当前 RV 索引
            sinr_accumulated_db: 累积 SINR (dB)

        Returns:
            是否应提前重传
        """
        # 切换准备期总是提前重传（如果可能）
        if ue_id in self._handover_prep_ues and current_rv < self.cfg.early_retx_max_rv:
            return True

        # 检查趋势
        trend = self.lookahead.compute_trend(ue_id)

        # 信道正在快速恶化且 HARQ 未完成
        if trend < self.cfg.early_retx_trend_threshold and current_rv < self.cfg.early_retx_max_rv:
            return True

        return False

    def get_adjusted_sinr_for_mcs(
        self,
        ue_id: int,
        sinr_eff_db: float,
        k1_slots: int,
        olla_offset_db: float = 0.0,
    ) -> float:
        """获取用于 MCS 选择的调整后 SINR

        Args:
            ue_id: UE ID
            sinr_eff_db: 有效 SINR (dB)
            k1_slots: ACK 延迟
            olla_offset_db: OLLA 偏移 (dB)

        Returns:
            调整后的 SINR (dB)
        """
        adj = self.get_mcs_adjustment(ue_id, sinr_eff_db, k1_slots)
        return sinr_eff_db + olla_offset_db + adj.sinr_adjustment_db

    def get_statistics(self) -> Dict[str, int]:
        """获取统计"""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """重置统计"""
        self._stats = {
            'standard': 0,
            'aggressive': 0,
            'conservative': 0,
            'urgent': 0,
        }


# ============== 批量处理工具 ==============

def compute_harq_sinr_adjustments(
    lookahead_factor: 'LookaheadFactor',
    k1_slots: np.ndarray,
    k2_slots: int,
    config: HARQLookaheadConfig,
    handover_prep_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """批量计算 HARQ SINR 调整

    Args:
        lookahead_factor: 前瞻因子计算器
        k1_slots: 每 UE 的 ACK 延迟 [N_UE]
        k2_slots: 重传延迟 (标量)
        config: 配置
        handover_prep_mask: 切换准备期掩码 [N_UE]

    Returns:
        SINR 调整量 [N_UE] (dB)
    """
    if not lookahead_factor.is_cached:
        return np.zeros(len(k1_slots))

    n_ue = len(k1_slots)
    adjustments = np.zeros(n_ue)

    # 获取当前 G_sat
    cache = lookahead_factor._cache
    if cache is None:
        return adjustments

    g_sat_now = cache.values[:, 0]

    # 计算每个 UE 的预测 G_sat
    sample_interval = lookahead_factor.cfg.lookahead_sample_interval
    n_samples = cache.n_samples

    for ue in range(n_ue):
        t_ack = int(k1_slots[ue]) + k2_slots
        idx = min(t_ack // sample_interval, n_samples - 1)
        idx = max(0, idx)

        g_sat_ack = cache.values[ue, idx]
        delta = g_sat_ack - g_sat_now[ue]

        # 切换准备期
        if handover_prep_mask is not None and handover_prep_mask[ue]:
            adjustments[ue] = -config.urgent_sinr_margin_db
            continue

        # 根据预测调整
        if delta > config.delta_threshold_db:
            adjustments[ue] = config.aggressive_sinr_boost_db
        elif delta < -config.delta_threshold_db:
            adjustments[ue] = -config.conservative_sinr_margin_db
        else:
            adjustments[ue] = 0.0

    return adjustments


# ============== 工厂函数 ==============

def create_harq_lookahead_mcs(
    lookahead_factor: 'LookaheadFactor',
    config: Optional[Dict[str, Any]] = None
) -> HARQLookaheadMCS:
    """创建 HARQ 前瞻 MCS 选择器

    Args:
        lookahead_factor: LookaheadFactor 实例
        config: 配置字典

    Returns:
        HARQLookaheadMCS 实例
    """
    harq_config = HARQLookaheadConfig.from_config_dict(config) if config else HARQLookaheadConfig()
    return HARQLookaheadMCS(lookahead_factor, harq_config)


# ============== 导出 ==============

__all__ = [
    'MCSStrategy',
    'MCSAdjustment',
    'HARQLookaheadConfig',
    'HARQLookaheadMCS',
    'compute_harq_sinr_adjustments',
    'create_harq_lookahead_mcs',
]
