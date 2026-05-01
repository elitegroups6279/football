"""
边际消除算法 - 将博彩公司赔率转换为真实概率

三种算法:
  1. Basic proportionality (基本比例法): true = implied / sum_implied
  2. Power method (幂方法): 对赔率取1/k次幂后归一化 (k > 1)
  3. Shin's method (Shin方法): 迭代求解, 考虑市场低效

参考:
  - Clarke, S. (2008). "An analysis of the efficacy of Shin's method"
  - Wikipedia: https://en.wikipedia.org/wiki/Shin%27s_method
"""

import math
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class MarginRemover:
    """博彩公司边际消除器"""

    @staticmethod
    def remove_margin(
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        method: str = "shin",
        power_k: float = 1.2,
        shin_tol: float = 1e-8,
        shin_max_iter: int = 1000,
    ) -> Dict[str, float]:
        """
        将赔率转换为真实概率 (去除博彩公司边际)

        Args:
            home_odds: 主胜赔率
            draw_odds: 平局赔率
            away_odds: 客胜赔率
            method: 'basic' | 'power' | 'shin'
            power_k: 幂方法的k值 (默认1.2)
            shin_tol: Shin方法收敛容差
            shin_max_iter: Shin方法最大迭代次数

        Returns:
            {home_prob, draw_prob, away_prob} 真实概率, 和为1
        """
        if method == "basic":
            return MarginRemover._basic(home_odds, draw_odds, away_odds)
        elif method == "power":
            return MarginRemover._power(home_odds, draw_odds, away_odds, k=power_k)
        elif method == "shin":
            return MarginRemover._shin(home_odds, draw_odds, away_odds,
                                       tol=shin_tol, max_iter=shin_max_iter)
        else:
            logger.warning(f"未知边际消除方法 '{method}', 使用basic")
            return MarginRemover._basic(home_odds, draw_odds, away_odds)

    @staticmethod
    def _basic(home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """基本比例法: 隐含概率之和 > 1, 归一化即可"""
        implied_home = 1.0 / home_odds
        implied_draw = 1.0 / draw_odds
        implied_away = 1.0 / away_odds
        total = implied_home + implied_draw + implied_away

        return {
            "home_prob": round(implied_home / total, 6),
            "draw_prob": round(implied_draw / total, 6),
            "away_prob": round(implied_away / total, 6),
        }

    @staticmethod
    def _power(home_odds: float, draw_odds: float, away_odds: float,
               k: float = 1.2) -> Dict[str, float]:
        """
        幂方法: 对赔率取1/k次方

        true_prob_i = (1/odds_i)^(1/k) / sum((1/odds_j)^(1/k))

        k=1 等价于 basic; k>1 会压缩极端概率
        """
        implied_home = (1.0 / home_odds) ** (1.0 / k)
        implied_draw = (1.0 / draw_odds) ** (1.0 / k)
        implied_away = (1.0 / away_odds) ** (1.0 / k)
        total = implied_home + implied_draw + implied_away

        return {
            "home_prob": round(implied_home / total, 6),
            "draw_prob": round(implied_draw / total, 6),
            "away_prob": round(implied_away / total, 6),
        }

    @staticmethod
    def _shin(home_odds: float, draw_odds: float, away_odds: float,
              tol: float = 1e-8, max_iter: int = 1000) -> Dict[str, float]:
        """
        Shin方法: 假设博彩市场存在一定比例的低效投注(z)

        通过迭代求解z使得真实概率之和为1
        初始z=0, 迭代更新直到收敛
        """
        # 隐含概率
        imp_h = 1.0 / home_odds
        imp_d = 1.0 / draw_odds
        imp_a = 1.0 / away_odds
        imp_total = imp_h + imp_d + imp_a
        margin = imp_total - 1.0

        # 初始值: z从margin开始
        z = margin
        for _ in range(max_iter):
            # 计算真实概率: p_i = sqrt(z^2 + 4*(1-z)*imp_i^2) - z) / (2*(1-z))
            # 修正版公式, 避免除以0
            z_prev = z

            denom = 2.0 * (1.0 - z) if z < 1.0 else 2.0 * 1e-10

            p_h = (math.sqrt(z * z + 4.0 * (1.0 - z) * imp_h * imp_h) - z) / denom
            p_d = (math.sqrt(z * z + 4.0 * (1.0 - z) * imp_d * imp_d) - z) / denom
            p_a = (math.sqrt(z * z + 4.0 * (1.0 - z) * imp_a * imp_a) - z) / denom

            p_sum = p_h + p_d + p_a

            # 更新z: z = 1 - sum(p_i^2) / (sum(p_i) - margin) ... 简化近似
            # 实际上直接用二分法求使得sum(p)=1的z
            if abs(p_sum - 1.0) < tol:
                break

            # 牛顿法更新z: z_{n+1} = z_n - (sum(p) - 1) / d(sum(p))/dz
            # 近似: z = z + (1 - p_sum) / 2
            z = z + (1.0 - p_sum) * 0.5
            z = max(0.0, min(1.0, z))

            if abs(z - z_prev) < tol:
                break

        # 最终归一化保证和为1
        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total
            p_d /= total
            p_a /= total

        return {
            "home_prob": round(p_h, 6),
            "draw_prob": round(p_d, 6),
            "away_prob": round(p_a, 6),
        }

    @staticmethod
    def get_payout_rate(home_odds: float, draw_odds: float, away_odds: float) -> float:
        """计算博彩公司赔付率(返还率)"""
        implied = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds
        return round(1.0 / implied, 6)

    @staticmethod
    def get_overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
        """计算博彩公司边际(overround)"""
        implied = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds
        return round(implied - 1.0, 6)
