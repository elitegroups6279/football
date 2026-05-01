"""
泊松进球预测模型

基于泊松分布预测足球比赛进球数和赛果:
  - P(X=k) = λ^k * e^(-λ) / k!
  - 假设主客队进球独立服从泊松分布
  - 构建比分概率矩阵, 推导胜平负、大小球概率

参考:
  - Maher (1982). "Modelling association football scores"
  - Dixon & Coles (1997). "Modelling association football scores and inefficiencies"
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PoissonPredictor:
    """泊松进球预测器"""

    # 默认联赛进球参数 (竞彩常见联赛)
    DEFAULT_LEAGUE_PARAMS = {
        "attack_home_avg": 1.45,
        "attack_away_avg": 1.15,
        "defense_home_avg": 1.10,
        "defense_away_avg": 1.40,
    }

    def __init__(self, max_goals: int = 6):
        """
        Args:
            max_goals: 每队最多预测进球数 (用于构建概率矩阵)
        """
        self.max_goals = max_goals
        # 缓存阶乘
        self._factorial_cache = [math.factorial(i) for i in range(max_goals + 1)]

    def poisson_pmf(self, lam: float, k: int) -> float:
        """
        泊松概率质量函数 P(X=k)

        Args:
            lam: 泊松分布的λ参数 (平均进球数)
            k: 进球数

        Returns:
            P(X=k)
        """
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        if k > self.max_goals:
            k = self.max_goals
        return (lam ** k) * math.exp(-lam) / self._factorial_cache[k]

    def build_goal_grid(self, home_lambda: float, away_lambda: float) -> np.ndarray:
        """
        构建比分概率矩阵 (7x7)

        Args:
            home_lambda: 主队期望进球
            away_lambda: 客队期望进球

        Returns:
            (max_goals+1) x (max_goals+1) 矩阵, [i,j] = P(home=i, away=j)
        """
        max_g = self.max_goals
        grid = np.zeros((max_g + 1, max_g + 1))

        home_probs = [self.poisson_pmf(home_lambda, i) for i in range(max_g + 1)]
        away_probs = [self.poisson_pmf(away_lambda, j) for j in range(max_g + 1)]

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                grid[i, j] = home_probs[i] * away_probs[j]

        return grid

    def predict_outcome(self, home_lambda: float, away_lambda: float) -> Dict[str, float]:
        """
        预测比赛胜平负概率

        Args:
            home_lambda: 主队期望进球
            away_lambda: 客队期望进球

        Returns:
            {home_win, draw, away_win} 概率
        """
        grid = self.build_goal_grid(home_lambda, away_lambda)
        max_g = self.max_goals

        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                prob = grid[i, j]
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob

        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total

        return {
            "home_win": round(home_win, 6),
            "draw": round(draw, 6),
            "away_win": round(away_win, 6),
        }

    def predict_over_under(self, home_lambda: float, away_lambda: float,
                           threshold: float = 2.5) -> Dict[str, float]:
        """
        预测大小球概率

        Args:
            home_lambda: 主队期望进球
            away_lambda: 客队期望进球
            threshold: 大小球阈值 (默认2.5)

        Returns:
            {over, under} 概率
        """
        grid = self.build_goal_grid(home_lambda, away_lambda)
        max_g = self.max_goals

        over = 0.0
        under = 0.0

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                total_goals = i + j
                prob = grid[i, j]
                if total_goals > threshold:
                    over += prob
                else:
                    under += prob

        total = over + under
        if total > 0:
            over /= total
            under /= total

        return {
            "over": round(over, 6),
            "under": round(under, 6),
            "expected_goals": round(home_lambda + away_lambda, 4),
        }

    def predict_exact_score(self, home_lambda: float, away_lambda: float,
                            top_n: int = 5) -> List[Dict]:
        """
        预测最可能的比分

        Args:
            home_lambda: 主队期望进球
            away_lambda: 客队期望进球
            top_n: 返回前N个最可能比分

        Returns:
            [{score, probability, cumulative}] 按概率降序
        """
        grid = self.build_goal_grid(home_lambda, away_lambda)
        max_g = self.max_goals

        results = []
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                prob = grid[i, j]
                if prob > 0.001:  # 过滤极低概率
                    results.append({
                        "score": f"{i}:{j}",
                        "home_goals": i,
                        "away_goals": j,
                        "probability": round(prob, 6),
                    })

        results.sort(key=lambda x: x["probability"], reverse=True)

        # 累计概率
        cumulative = 0.0
        for r in results[:top_n]:
            cumulative += r["probability"]
            r["cumulative"] = round(cumulative, 4)

        return results[:top_n]

    def estimate_lambda_from_odds(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        method: str = "shin",
    ) -> Dict[str, float]:
        """
        从赔率反推泊松λ参数

        使用:
          1. 从赔率中消除边际得到真实概率
          2. 用经验公式估算主客队λ

        Args:
            home_odds: 主胜赔率
            draw_odds: 平局赔率
            away_odds: 客胜赔率
            method: 边际消除方法

        Returns:
            {home_lambda, away_lambda}
        """
        from prediction.margin_removal import MarginRemover

        probs = MarginRemover.remove_margin(home_odds, draw_odds, away_odds, method=method)
        p_home = probs["home_prob"]
        p_draw = probs["draw_prob"]
        p_away = probs["away_prob"]

        # 从胜平负概率估算λ的经验方法:
        # 用泊松概率的比值来反推
        # P(home_win) = sum_{i>j} Pois(i|λ_h) * Pois(j|λ_a)
        # 这里用简化的近似公式:
        total_p = p_home + p_draw + p_away
        if total_p == 0:
            total_p = 1

        # 期望进球 = 通过赔率隐含概率估算
        # 参考: 主队λ ≈ -ln(1 - p_home - 0.5*p_draw) 的简化
        # 更精确: 使用查表法或迭代求解

        # 简化近似: 用总进球期望 ≈ 2.5 (典型足球比赛)
        # 主队进球 = 2.5 * p_home / (p_home + p_away) + 0.5  (主场优势修正)
        home_attack_ratio = p_home / (p_home + p_away + 0.001)

        # 竞彩足球平均总进球约 2.5, 分配主客
        total_expected = 2.5
        home_lambda = total_expected * home_attack_ratio + 0.15  # 主场优势
        away_lambda = total_expected - home_lambda

        home_lambda = max(0.3, min(3.5, home_lambda))
        away_lambda = max(0.2, min(3.0, away_lambda))

        return {
            "home_lambda": round(home_lambda, 4),
            "away_lambda": round(away_lambda, 4),
        }

    def extract_league_strength(
        self, df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        从历史数据提取联赛级别的攻防强度参数

        Args:
            df: 包含 league_id, home_score, away_score, home_team, away_team 的DataFrame

        Returns:
            {league_id: {attack_home_avg, defense_home_avg, attack_away_avg, defense_away_avg}}
        """
        if df.empty or not all(c in df.columns for c in ["home_score", "away_score"]):
            logger.warning("历史数据不足, 使用默认参数")
            return {}

        strengths = {}
        for league_id, group in df.groupby("league_id") if "league_id" in df.columns else [(0, df)]:
            group = group.dropna(subset=["home_score", "away_score"])
            if len(group) < 5:
                continue

            strengths[int(league_id)] = {
                "attack_home_avg": round(group["home_score"].mean(), 4),
                "defense_home_avg": round(group["away_score"].mean(), 4),
                "attack_away_avg": round(group["away_score"].mean(), 4),
                "defense_away_avg": round(group["home_score"].mean(), 4),
            }

        return strengths

    def predict_from_odds(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        margin_method: str = "shin",
    ) -> Dict:
        """
        从赔率直接生成完整预测

        Args:
            home_odds: 主胜赔率
            draw_odds: 平局赔率
            away_odds: 客胜赔率
            margin_method: 边际消除方法

        Returns:
            {outcome, over_under, exact_scores, lambdas, fair_odds}
        """
        # 1. 从赔率估算λ
        lambdas = self.estimate_lambda_from_odds(home_odds, draw_odds, away_odds, margin_method)
        home_lambda = lambdas["home_lambda"]
        away_lambda = lambdas["away_lambda"]

        # 2. 泊松预测胜平负
        outcome = self.predict_outcome(home_lambda, away_lambda)

        # 3. 大小球
        over_under = self.predict_over_under(home_lambda, away_lambda)

        # 4. 最可能比分
        scores = self.predict_exact_score(home_lambda, away_lambda, top_n=5)

        # 5. 公平赔率
        fair_odds = {
            "fair_home": round(1.0 / outcome["home_win"], 4) if outcome["home_win"] > 0 else 999,
            "fair_draw": round(1.0 / outcome["draw"], 4) if outcome["draw"] > 0 else 999,
            "fair_away": round(1.0 / outcome["away_win"], 4) if outcome["away_win"] > 0 else 999,
        }

        return {
            "outcome": outcome,
            "over_under": over_under,
            "exact_scores": scores,
            "lambdas": lambdas,
            "fair_odds": fair_odds,
        }
