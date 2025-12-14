"""
NTN 预测性切换优化模块

专利核心创新：
1. 基于轨道预测的切换时刻预测
2. 切换感知调度策略
3. 切换准备期资源优化

核心思想：利用轨道的完全可预测性，精确预测切换时刻，
并在切换前后采取针对性的调度策略，减少切换中断影响。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============== 数据类型定义 ==============

class HandoverReason(Enum):
    """切换原因"""
    NONE = "none"               # 无切换
    ELEVATION = "elevation"     # 仰角过低
    SNR = "snr"                 # SNR 更优
    COVERAGE = "coverage"       # 覆盖丢失


class SchedulingPhase(Enum):
    """调度阶段"""
    NORMAL = "normal"           # 正常调度
    PREPARATION = "preparation" # 切换准备期
    HANDOVER = "handover"       # 切换中
    RECOVERY = "recovery"       # 切换恢复期


@dataclass
class HandoverPrediction:
    """切换预测结果"""
    ue_id: int
    will_handover: bool
    predicted_time: Optional[int]           # 预测切换时刻 (TTI)
    target_satellite: Optional[int]         # 目标卫星 ID
    source_satellite: int                   # 源卫星 ID
    reason: HandoverReason                  # 切换原因
    time_to_handover: Optional[int]         # 距离切换的 TTI 数
    confidence: float                       # 置信度 [0, 1]
    snr_improvement_db: float              # 预期 SNR 改善 (dB)


@dataclass
class SchedulingAdjustment:
    """调度调整建议"""
    ue_id: int
    phase: SchedulingPhase
    defer_non_realtime: bool                # 是否延迟非实时业务
    accelerate_realtime: bool               # 是否加速实时业务
    reduce_new_harq: bool                   # 是否减少新 HARQ 传输
    target_satellite: Optional[int]         # 目标卫星 (切换中时)
    priority_boost: float                   # 优先级提升因子
    reason: str                             # 调整原因


# ============== 配置 ==============

@dataclass
class PredictiveHOConfig:
    """预测性切换配置"""
    # 切换检测参数
    min_elev_deg: float = 10.0              # 最小仰角 (度)
    ho_hyst_db: float = 3.0                 # 切换滞后门限 (dB)
    ho_ttt_ttis: int = 20                   # Time-to-Trigger (TTI 数)

    # 前瞻参数
    lookahead_horizon_ttis: int = 200       # 前瞻窗口
    prep_window_ttis: int = 100             # 切换准备窗口
    recovery_window_ttis: int = 50          # 切换恢复窗口

    # 调度调整参数
    realtime_priority_boost: float = 1.5    # 实时业务优先级提升
    new_harq_reduction: float = 0.5         # 新 HARQ 传输减少比例

    # 调试
    enable_logging: bool = False

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> 'PredictiveHOConfig':
        """从配置字典创建"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)


# ============== 预测性切换管理器 ==============

class PredictiveHandoverManager:
    """预测性切换管理器

    核心功能：
    1. 预测切换时刻
    2. 确定调度阶段
    3. 提供调度调整建议
    """

    def __init__(
        self,
        n_ue: int,
        orbit,
        config: Optional[PredictiveHOConfig] = None
    ):
        """
        Args:
            n_ue: UE 数量
            orbit: ConstellationOrbit 实例
            config: 配置
        """
        self.n_ue = n_ue
        self.orbit = orbit
        self.cfg = config or PredictiveHOConfig()

        # 状态
        self.serving_sat = np.full(n_ue, -1, dtype=int)
        self.predictions: Dict[int, HandoverPrediction] = {}
        self.current_tti = 0

        # 统计
        self._stats = {
            'predictions_made': 0,
            'handovers_predicted': 0,
            'handovers_by_elevation': 0,
            'handovers_by_snr': 0,
        }

    def set_serving(self, serving: np.ndarray) -> None:
        """设置当前服务卫星

        Args:
            serving: 每个 UE 的服务卫星 ID [N_UE]
        """
        self.serving_sat = np.asarray(serving).flatten().copy()

    def update_time(self, t: int) -> None:
        """更新当前时间

        Args:
            t: 当前 TTI
        """
        self.current_tti = t

    def predict_handover(
        self,
        ue_id: int,
        ue_pos: np.ndarray,
        t: Optional[int] = None
    ) -> HandoverPrediction:
        """预测单个 UE 的切换

        搜索未来 H 个 TTI，检测：
        1. 仰角条件：服务卫星仰角低于阈值
        2. SNR 条件：候选卫星 SNR 超过服务卫星 + 滞后

        Args:
            ue_id: UE ID
            ue_pos: 该 UE 的位置 [2]
            t: 当前 TTI，None 时使用内部时间

        Returns:
            HandoverPrediction
        """
        t = t if t is not None else self.current_tti
        self._stats['predictions_made'] += 1

        serving = self.serving_sat[ue_id]
        if serving < 0:
            # 无服务卫星
            return HandoverPrediction(
                ue_id=ue_id,
                will_handover=False,
                predicted_time=None,
                target_satellite=None,
                source_satellite=-1,
                reason=HandoverReason.NONE,
                time_to_handover=None,
                confidence=1.0,
                snr_improvement_db=0.0,
            )

        # 确保 ue_pos 是 2D
        ue_pos_2d = np.asarray(ue_pos).reshape(1, -1)

        # 搜索切换触发点
        ttt_counter = 0
        ttt_candidate = -1

        for tau in range(t, t + self.cfg.lookahead_horizon_ttis):
            # 获取当前服务卫星的几何
            try:
                _, _, _, _, elev = self.orbit.geometry_for_sat(ue_pos_2d, serving, tau)
                elev_serving = float(elev[0])
            except Exception:
                elev_serving = 0.0

            # 检查仰角条件
            if elev_serving < self.cfg.min_elev_deg:
                # 找替代卫星
                candidates = self._get_candidates_at(tau)
                best_sat, best_snr, _ = self._find_best_sat(
                    ue_pos_2d, candidates, tau, exclude=serving
                )

                self._stats['handovers_predicted'] += 1
                self._stats['handovers_by_elevation'] += 1

                pred = HandoverPrediction(
                    ue_id=ue_id,
                    will_handover=True,
                    predicted_time=tau,
                    target_satellite=best_sat,
                    source_satellite=serving,
                    reason=HandoverReason.ELEVATION,
                    time_to_handover=tau - t,
                    confidence=0.95,
                    snr_improvement_db=0.0,  # 被迫切换，不一定改善
                )
                self.predictions[ue_id] = pred
                return pred

            # 检查 SNR 条件
            snr_serving = self._compute_snr(ue_pos_2d, serving, tau)
            candidates = self._get_candidates_at(tau)

            for cand in candidates:
                if cand == serving:
                    continue

                snr_cand = self._compute_snr(ue_pos_2d, cand, tau)

                if snr_cand > snr_serving + self.cfg.ho_hyst_db:
                    if ttt_candidate != cand:
                        ttt_counter = 0
                        ttt_candidate = cand

                    ttt_counter += 1

                    if ttt_counter >= self.cfg.ho_ttt_ttis:
                        self._stats['handovers_predicted'] += 1
                        self._stats['handovers_by_snr'] += 1

                        pred = HandoverPrediction(
                            ue_id=ue_id,
                            will_handover=True,
                            predicted_time=tau,
                            target_satellite=cand,
                            source_satellite=serving,
                            reason=HandoverReason.SNR,
                            time_to_handover=tau - t,
                            confidence=0.8,
                            snr_improvement_db=snr_cand - snr_serving,
                        )
                        self.predictions[ue_id] = pred
                        return pred
                else:
                    if ttt_candidate == cand:
                        ttt_counter = 0

        # 无切换预测
        pred = HandoverPrediction(
            ue_id=ue_id,
            will_handover=False,
            predicted_time=None,
            target_satellite=None,
            source_satellite=serving,
            reason=HandoverReason.NONE,
            time_to_handover=None,
            confidence=1.0,
            snr_improvement_db=0.0,
        )
        self.predictions[ue_id] = pred
        return pred

    def predict_all_handovers(
        self,
        ue_pos: np.ndarray,
        t: Optional[int] = None
    ) -> List[HandoverPrediction]:
        """预测所有 UE 的切换

        Args:
            ue_pos: UE 位置 [N_UE, 2]
            t: 当前 TTI

        Returns:
            预测列表
        """
        t = t if t is not None else self.current_tti
        predictions = []

        for ue_id in range(self.n_ue):
            pred = self.predict_handover(ue_id, ue_pos[ue_id], t)
            predictions.append(pred)

        return predictions

    def get_scheduling_phase(
        self,
        ue_id: int,
        t: Optional[int] = None
    ) -> SchedulingPhase:
        """获取调度阶段

        Args:
            ue_id: UE ID
            t: 当前 TTI

        Returns:
            SchedulingPhase
        """
        t = t if t is not None else self.current_tti
        pred = self.predictions.get(ue_id)

        if pred is None or not pred.will_handover:
            return SchedulingPhase.NORMAL

        t_ho = pred.predicted_time
        prep = self.cfg.prep_window_ttis
        recovery = self.cfg.recovery_window_ttis

        if t < t_ho - prep:
            return SchedulingPhase.NORMAL
        elif t < t_ho:
            return SchedulingPhase.PREPARATION
        elif t == t_ho:
            return SchedulingPhase.HANDOVER
        elif t < t_ho + recovery:
            return SchedulingPhase.RECOVERY
        else:
            return SchedulingPhase.NORMAL

    def get_scheduling_adjustment(
        self,
        ue_id: int,
        t: Optional[int] = None
    ) -> SchedulingAdjustment:
        """获取调度调整建议

        Args:
            ue_id: UE ID
            t: 当前 TTI

        Returns:
            SchedulingAdjustment
        """
        t = t if t is not None else self.current_tti
        phase = self.get_scheduling_phase(ue_id, t)
        pred = self.predictions.get(ue_id)

        if phase == SchedulingPhase.NORMAL:
            return SchedulingAdjustment(
                ue_id=ue_id,
                phase=phase,
                defer_non_realtime=False,
                accelerate_realtime=False,
                reduce_new_harq=False,
                target_satellite=None,
                priority_boost=1.0,
                reason="Normal scheduling",
            )

        elif phase == SchedulingPhase.PREPARATION:
            # 切换准备期策略
            target_better = (pred.reason == HandoverReason.SNR and
                           pred.snr_improvement_db > self.cfg.ho_hyst_db)

            return SchedulingAdjustment(
                ue_id=ue_id,
                phase=phase,
                defer_non_realtime=target_better,
                accelerate_realtime=True,
                reduce_new_harq=True,
                target_satellite=pred.target_satellite,
                priority_boost=self.cfg.realtime_priority_boost,
                reason=f"Handover prep: {pred.time_to_handover} TTIs to HO",
            )

        elif phase == SchedulingPhase.HANDOVER:
            return SchedulingAdjustment(
                ue_id=ue_id,
                phase=phase,
                defer_non_realtime=False,
                accelerate_realtime=True,
                reduce_new_harq=True,
                target_satellite=pred.target_satellite,
                priority_boost=self.cfg.realtime_priority_boost * 1.5,
                reason="Handover in progress",
            )

        else:  # RECOVERY
            return SchedulingAdjustment(
                ue_id=ue_id,
                phase=phase,
                defer_non_realtime=False,
                accelerate_realtime=False,
                reduce_new_harq=False,
                target_satellite=None,
                priority_boost=1.0,
                reason="Post-handover recovery",
            )

    def get_all_adjustments(
        self,
        t: Optional[int] = None
    ) -> List[SchedulingAdjustment]:
        """获取所有 UE 的调度调整"""
        return [self.get_scheduling_adjustment(u, t) for u in range(self.n_ue)]

    def get_ue_in_preparation(
        self,
        t: Optional[int] = None
    ) -> List[int]:
        """获取处于切换准备期的 UE 列表"""
        t = t if t is not None else self.current_tti
        result = []
        for ue_id in range(self.n_ue):
            if self.get_scheduling_phase(ue_id, t) == SchedulingPhase.PREPARATION:
                result.append(ue_id)
        return result

    def get_ue_in_handover(
        self,
        t: Optional[int] = None
    ) -> List[int]:
        """获取正在切换的 UE 列表"""
        t = t if t is not None else self.current_tti
        result = []
        for ue_id in range(self.n_ue):
            if self.get_scheduling_phase(ue_id, t) == SchedulingPhase.HANDOVER:
                result.append(ue_id)
        return result

    def get_statistics(self) -> Dict[str, int]:
        """获取统计"""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """重置统计"""
        self._stats = {
            'predictions_made': 0,
            'handovers_predicted': 0,
            'handovers_by_elevation': 0,
            'handovers_by_snr': 0,
        }

    # ============== 内部方法 ==============

    def _get_candidates_at(self, t: int) -> List[int]:
        """获取指定时刻的候选卫星"""
        if hasattr(self.orbit, 'candidate_indices_at'):
            return self.orbit.candidate_indices_at(t)
        return []

    def _compute_snr(self, ue_pos: np.ndarray, sat_id: int, t: int) -> float:
        """计算 SNR (简化版：使用 G_sat 作为代理)

        Args:
            ue_pos: UE 位置 [1, 2]
            sat_id: 卫星 ID
            t: TTI

        Returns:
            SNR 代理值 (dB)
        """
        try:
            L_fs, G_rx, _, _, _ = self.orbit.geometry_for_sat(ue_pos, sat_id, t)
            return float(G_rx[0] - L_fs[0])
        except Exception:
            return -np.inf

    def _find_best_sat(
        self,
        ue_pos: np.ndarray,
        candidates: List[int],
        t: int,
        exclude: int = -1
    ) -> Tuple[int, float, float]:
        """找最佳卫星

        Returns:
            (best_sat_id, best_snr, best_elev)
        """
        best_sat = -1
        best_snr = -np.inf
        best_elev = 0.0

        for cand in candidates:
            if cand == exclude:
                continue

            snr = self._compute_snr(ue_pos, cand, t)

            if snr > best_snr:
                best_snr = snr
                best_sat = cand

                try:
                    _, _, _, _, elev = self.orbit.geometry_for_sat(ue_pos, cand, t)
                    best_elev = float(elev[0])
                except Exception:
                    best_elev = 0.0

        return best_sat, best_snr, best_elev


# ============== 切换感知调度度量修正 ==============

def apply_handover_aware_correction(
    metric: np.ndarray,
    adjustments: List[SchedulingAdjustment],
    is_realtime: np.ndarray,
) -> np.ndarray:
    """应用切换感知调度度量修正

    Args:
        metric: 原始调度度量 [N_UE] 或 [N_UE, Z]
        adjustments: 调度调整列表
        is_realtime: 是否实时业务 [N_UE]

    Returns:
        修正后的度量
    """
    corrected = metric.copy()
    n_ue = len(adjustments)

    for adj in adjustments:
        ue = adj.ue_id
        if ue >= n_ue:
            continue

        # 应用优先级提升
        if is_realtime[ue] and adj.accelerate_realtime:
            if corrected.ndim == 1:
                corrected[ue] *= adj.priority_boost
            else:
                corrected[ue, :] *= adj.priority_boost

        # 延迟非实时业务
        if not is_realtime[ue] and adj.defer_non_realtime:
            if corrected.ndim == 1:
                corrected[ue] *= 0.1  # 大幅降低优先级
            else:
                corrected[ue, :] *= 0.1

    return corrected


# ============== 工厂函数 ==============

def create_predictive_ho_manager(
    n_ue: int,
    orbit,
    config: Optional[Dict[str, Any]] = None
) -> PredictiveHandoverManager:
    """创建预测性切换管理器

    Args:
        n_ue: UE 数量
        orbit: ConstellationOrbit 实例
        config: 配置字典

    Returns:
        PredictiveHandoverManager 实例
    """
    ho_config = PredictiveHOConfig.from_config_dict(config) if config else PredictiveHOConfig()
    return PredictiveHandoverManager(n_ue, orbit, ho_config)


# ============== 导出 ==============

__all__ = [
    'HandoverReason',
    'SchedulingPhase',
    'HandoverPrediction',
    'SchedulingAdjustment',
    'PredictiveHOConfig',
    'PredictiveHandoverManager',
    'apply_handover_aware_correction',
    'create_predictive_ho_manager',
]
