"""
价值投注模块 - 计算期望值和凯利公式投注额

核心公式:
  - Expected Value: EV = P(win) * odds - 1
  - Kelly Criterion: f* = (p * b - q) / b  =  (p * odds - 1) / (odds - 1)
  - Fractional Kelly: f_fraction = f* * fraction (默认1/4, 控制风险)
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION, MIN_EV_THRESHOLD

logger = logging.getLogger(__name__)


class ValueBetting:
    """价值投注计算器"""

    @staticmethod
    def calculate_ev(probability: float, odds: float) -> float:
        """
        计算期望值

        EV = probability * odds - 1
        EV > 0 意味着有投注价值

        Args:
            probability: 真实概率 (0~1)
            odds: 博彩公司赔率

        Returns:
            期望值 (小数)
        """
        if probability <= 0 or odds <= 1:
            return 0.0
        return round(probability * odds - 1.0, 6)

    @staticmethod
    def calculate_kelly_stake(
        probability: float,
        odds: float,
        bankroll: float = DEFAULT_BANKROLL,
        fraction: float = DEFAULT_KELLY_FRACTION,
    ) -> Dict[str, float]:
        """
        计算凯利公式建议投注额

        f* = (p * odds - 1) / (odds - 1)
        stake = bankroll * f* * fraction

        Args:
            probability: 真实概率 (0~1)
            odds: 博彩公司赔率
            bankroll: 总本金
            fraction: 凯利系数 (默认0.25, 即1/4凯利)

        Returns:
            {full_kelly, fractional_kelly, stake_amount, ev}
        """
        if probability <= 0 or odds <= 1:
            return {"full_kelly": 0, "fractional_kelly": 0, "stake_amount": 0, "ev": 0}

        ev = probability * odds - 1.0
        if ev <= 0:
            return {"full_kelly": 0, "fractional_kelly": 0, "stake_amount": 0, "ev": round(ev, 6)}

        # 全凯利
        full_kelly = (probability * odds - 1.0) / (odds - 1.0)
        full_kelly = max(0.0, min(1.0, full_kelly))  # 限制在0~1

        # 分数凯利
        frac_kelly = full_kelly * fraction

        # 投注金额
        stake = bankroll * frac_kelly

        return {
            "full_kelly": round(full_kelly, 6),
            "fractional_kelly": round(frac_kelly, 6),
            "stake_amount": round(stake, 2),
            "ev": round(ev, 6),
        }

    @staticmethod
    def calculate_edge(probability: float, odds: float) -> float:
        """
        计算"优势" / 边际

        edge = probability - (1/odds)
        正值表示你的概率估计高于博彩公司隐含概率

        Args:
            probability: 你的预测概率
            odds: 市场赔率

        Returns:
            优势值 (小数)
        """
        market_implied = 1.0 / odds
        return round(probability - market_implied, 6)

    @staticmethod
    def find_best_bet(
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
    ) -> Dict:
        """
        在胜平负三个选项中找最佳投注

        Args:
            home_prob: 预测主胜概率
            draw_prob: 预测平局概率
            away_prob: 预测客胜概率
            home_odds: 主胜赔率
            draw_odds: 平局赔率
            away_odds: 客胜赔率

        Returns:
            最佳投注建议 (包含选中的选项、EV、建议投注额等)
        """
        outcomes = [
            {"pick": "主胜", "prob": home_prob, "odds": home_odds},
            {"pick": "平局", "prob": draw_prob, "odds": draw_odds},
            {"pick": "客胜", "prob": away_prob, "odds": away_odds},
        ]

        best = None
        best_ev = -999

        for o in outcomes:
            ev = ValueBetting.calculate_ev(o["prob"], o["odds"])
            if ev > best_ev:
                best_ev = ev
                best = o

        if best is None:
            return {"pick": "无", "ev": 0, "stake": 0}

        kelly = ValueBetting.calculate_kelly_stake(best["prob"], best["odds"])
        return {
            "pick": best["pick"],
            "predicted_prob": best["prob"],
            "market_odds": best["odds"],
            "fair_odds": round(1.0 / best["prob"], 4) if best["prob"] > 0 else 999,
            "ev": best_ev,
            "edge": ValueBetting.calculate_edge(best["prob"], best["odds"]),
            "stake": kelly["stake_amount"],
            "kelly_fraction": kelly["fractional_kelly"],
        }

    @staticmethod
    def filter_value_bets(
        predictions: List[Dict],
        min_ev: float = MIN_EV_THRESHOLD,
        max_rec: int = 10,
    ) -> pd.DataFrame:
        """
        过滤出有价值的投注, 按EV排序

        Args:
            predictions: 每场比赛的预测结果列表
            min_ev: 最小EV阈值 (默认0.05)
            max_rec: 最大推荐数量

        Returns:
            DataFrame, 按EV降序排列
        """
        value_bets = []
        for p in predictions:
            bet = p.get("best_bet", {})
            ev = bet.get("ev", 0)
            if ev >= min_ev:
                value_bets.append({
                    **p.get("match_info", {}),
                    **bet,
                })

        if not value_bets:
            return pd.DataFrame()

        df = pd.DataFrame(value_bets)
        df = df.sort_values("ev", ascending=False).head(max_rec)
        return df.reset_index(drop=True)
