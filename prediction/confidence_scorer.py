"""
置信度评分模块 - 给每场比赛推荐分配0-100的置信度分

评分因子:
  1. EV得分 (30%): EV越高分越高
  2. 赔率共识度 (20%): 各公司赔率分歧越小分越高
  3. 泊松模型拟合度 (25%): 模型预测概率 vs 市场概率的差异
  4. 联赛进球趋势 (15%): 联赛平均进球数的参考价值
  5. 边际消除稳定性 (10%): 不同方法结果一致性
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import CONFIDENCE_WEIGHTS

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """置信度评分器"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or CONFIDENCE_WEIGHTS

    def score(self, factors: Dict[str, float]) -> int:
        """
        基于多因子计算最终置信度 (0-100)

        Args:
            factors: {
                "ev": EV值,
                "odds_variance": 赔率方差,
                "poisson_decisiveness": 泊松模型决定性,
                "league_goals": 联赛平均进球,
                "margin_stability": 边际消除稳定性,
            }

        Returns:
            置信度分数 (0-100的整数)
        """
        scores = {
            "ev_score": self._score_ev(factors.get("ev", 0)),
            "odds_consensus": self._score_consensus(factors.get("odds_variance", 1)),
            "poisson_fit": self._score_poisson(factors.get("poisson_decisiveness", 0)),
            "league_trend": self._score_league(factors.get("league_goals", 2.5)),
            "margin_stability": self._score_stability(factors.get("margin_stability", 0.5)),
        }

        total = 0.0
        for key, score in scores.items():
            weight = self.weights.get(key, 0.2)
            total += score * weight

        return min(100, max(0, int(round(total))))

    def _score_ev(self, ev: float) -> float:
        """EV得分 (0-100)"""
        if ev <= 0:
            return 0
        # EV=0.05 → 40分, EV=0.15 → 80分, EV>=0.25 → 100分
        score = ev / 0.25 * 100
        return min(100, max(0, score))

    def _score_consensus(self, variance: float) -> float:
        """赔率共识度得分 (0-100), 方差越小越好"""
        if variance is None or variance < 0:
            return 50
        # 方差<0.01 → 100分, 方差>0.1 → 0分
        if variance <= 0.01:
            return 100
        if variance >= 0.1:
            return 0
        # 线性插值
        return max(0, min(100, 100 - (variance - 0.01) / 0.09 * 100))

    def _score_poisson(self, decisiveness: float) -> float:
        """
        泊松模型决定性得分 (0-100)

        decisiveness: 模型预测的主胜概率偏离0.33的程度
        偏离越大 (越有把握) 分数越高
        """
        if decisiveness is None:
            return 50
        # decisiveness = |predicted_home - 0.33|, 范围0~0.67
        # 0 → 0分, 0.3 → 100分
        score = decisiveness / 0.3 * 100
        return min(100, max(0, score))

    def _score_league(self, avg_goals: float) -> float:
        """联赛进球趋势得分 (0-100)"""
        if avg_goals is None:
            return 50
        # 2.0~3.0进球最利于预测 → 100分
        # <1.5 或 >3.5 → 50分
        if 2.0 <= avg_goals <= 3.0:
            return 100
        if avg_goals < 1.5 or avg_goals > 3.5:
            return 50
        # 线性过渡
        if avg_goals < 2.0:
            return 50 + (avg_goals - 1.5) / 0.5 * 50
        else:  # avg_goals > 3.0
            return 100 - (avg_goals - 3.0) / 0.5 * 50

    def _score_stability(self, stability: float) -> float:
        """
        边际消除稳定性得分 (0-100)

        stability: 不同方法消除边际后的概率标准差, 越小越稳定
        取 1 - std * 2, 截断到0~100
        """
        if stability is None:
            return 50
        # std=0 → 100分, std>=0.5 → 0分
        score = (1.0 - stability * 2.0) * 100
        return max(0, min(100, score))

    def compute_factors(
        self,
        ev: float,
        odds_list: List[float],
        predicted_home_prob: float,
        market_home_prob: float,
        league_avg_goals: Optional[float] = None,
        margin_probs_list: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        计算所有评分因子

        Args:
            ev: 期望值
            odds_list: 各公司主胜赔率列表
            predicted_home_prob: 模型预测的主胜概率
            market_home_prob: 市场隐含主胜概率
            league_avg_goals: 联赛平均总进球
            margin_probs_list: 不同边际消除方法得到的概率列表

        Returns:
            {ev, odds_variance, poisson_decisiveness, league_goals, margin_stability}
        """
        # 赔率方差
        odds_variance = float(np.var(odds_list)) if len(odds_list) > 1 else 0.001

        # 泊松决定性: 模型与市场的偏差程度
        decisiveness = abs(predicted_home_prob - 0.33)

        # 联赛进球
        league_goals = league_avg_goals if league_avg_goals else 2.5

        # 边际消除稳定性
        margin_stability = 0.5
        if margin_probs_list and len(margin_probs_list) >= 2:
            home_probs = [p.get("home_prob", 0) for p in margin_probs_list if p]
            if home_probs:
                margin_stability = min(1.0, float(np.std(home_probs)) * 2)

        return {
            "ev": ev,
            "odds_variance": odds_variance,
            "poisson_decisiveness": decisiveness,
            "league_goals": league_goals,
            "margin_stability": margin_stability,
        }
