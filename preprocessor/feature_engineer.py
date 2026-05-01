"""
特征工程模块

从原始赔率数据中提取以下特征：
  - 主胜、平局、客胜的即时赔率
  - 平均赔率和赔率方差
  - 赔率变化趋势分析
  - 亚盘的盘口和水位信息
  - 凯利指数计算
  - 不同博彩公司间的赔率差异度量
  - 球队近期状态特征

目标变量构建：
  - 胜平负结果（三分类）
  - 让球盘输赢预测
  - 具体比分预测
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class KellyCalculator:
    """凯利指数计算器"""

    @staticmethod
    def calculate_kelly_index(
        home_win_odds: float,
        draw_odds: float,
        away_win_odds: float,
        home_win_prob: float = None,
        draw_prob: float = None,
        away_win_prob: float = None,
    ) -> Dict[str, float]:
        """
        计算凯利指数

        Kelly = 赔率 * 概率 - 1 / (赔率 - 1)

        Args:
            home_win_odds: 主胜赔率
            draw_odds: 平局赔率
            away_win_odds: 客胜赔率
            home_win_prob: 主胜概率（如果为None，使用1/odds估算）
            draw_prob: 平局概率
            away_win_prob: 客胜概率

        Returns:
            {"home_kelly": ..., "draw_kelly": ..., "away_kelly": ...}
        """
        # 如果没有提供概率，使用赔率的倒数估算
        if home_win_prob is None:
            total = 1 / home_win_odds + 1 / draw_odds + 1 / away_win_odds
            home_win_prob = (1 / home_win_odds) / total
            draw_prob = (1 / draw_odds) / total
            away_win_prob = (1 / away_win_odds) / total

        home_kelly = home_win_odds * home_win_prob - (1 - home_win_prob)
        draw_kelly = draw_odds * draw_prob - (1 - draw_prob)
        away_kelly = away_win_odds * away_win_prob - (1 - away_win_prob)

        return {
            "home_kelly": round(home_kelly, 4),
            "draw_kelly": round(draw_kelly, 4),
            "away_kelly": round(away_kelly, 4),
        }

    @staticmethod
    def calculate_expected_value(
        odds: float,
        probability: float,
    ) -> float:
        """
        计算期望值

        EV = 赔率 * 概率 - 1

        Args:
            odds: 赔率
            probability: 获胜概率

        Returns:
            期望值
        """
        return odds * probability - 1


class OddsFeatureExtractor:
    """赔率特征提取器"""

    @staticmethod
    def extract_europe_features(
        odds_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从欧赔数据中提取特征

        Args:
            odds_list: 欧赔数据列表

        Returns:
            特征字典
        """
        features = {
            "company_count": len(odds_list),
            "avg_home_win": None,
            "avg_draw": None,
            "avg_away_win": None,
            "var_home_win": None,
            "var_draw": None,
            "var_away_win": None,
            "min_home_win": None,
            "max_home_win": None,
            "min_away_win": None,
            "max_away_win": None,
            "home_draw_diff": None,
            "home_away_diff": None,
            "draw_away_diff": None,
            "max_company_diff": None,  # 最大公司间分歧
            "implied_home_prob": None,  # 隐含主胜概率
            "implied_draw_prob": None,
            "implied_away_prob": None,
            "kelly_home": None,
            "kelly_draw": None,
            "kelly_away": None,
        }

        if not odds_list:
            return features

        home_wins = [o["home_win"] for o in odds_list if o.get("home_win")]
        draws = [o["draw"] for o in odds_list if o.get("draw")]
        away_wins = [o["away_win"] for o in odds_list if o.get("away_win")]

        if not home_wins or not draws or not away_wins:
            return features

        # 平均赔率
        avg_home = np.mean(home_wins)
        avg_draw = np.mean(draws)
        avg_away = np.mean(away_wins)

        features["avg_home_win"] = round(avg_home, 4)
        features["avg_draw"] = round(avg_draw, 4)
        features["avg_away_win"] = round(avg_away, 4)

        # 赔率方差（分歧度）
        features["var_home_win"] = round(np.var(home_wins), 6)
        features["var_draw"] = round(np.var(draws), 6)
        features["var_away_win"] = round(np.var(away_wins), 6)

        # 极值
        features["min_home_win"] = min(home_wins)
        features["max_home_win"] = max(home_wins)
        features["min_away_win"] = min(away_wins)
        features["max_away_win"] = max(away_wins)

        # 赔率差异
        features["home_draw_diff"] = round(avg_home - avg_draw, 4)
        features["home_away_diff"] = round(avg_home - avg_away, 4)
        features["draw_away_diff"] = round(avg_draw - avg_away, 4)

        # 公司间最大分歧
        home_range = max(home_wins) - min(home_wins)
        away_range = max(away_wins) - min(away_wins)
        features["max_company_diff"] = round(max(home_range, away_range), 4)

        # 隐含概率（使用平均赔率）
        total_implied = 1/avg_home + 1/avg_draw + 1/avg_away
        features["implied_home_prob"] = round((1/avg_home) / total_implied, 4)
        features["implied_draw_prob"] = round((1/avg_draw) / total_implied, 4)
        features["implied_away_prob"] = round((1/avg_away) / total_implied, 4)

        # 凯利指数（使用平均赔率和隐含概率）
        kelly = KellyCalculator.calculate_kelly_index(
            avg_home, avg_draw, avg_away,
            features["implied_home_prob"],
            features["implied_draw_prob"],
            features["implied_away_prob"],
        )
        features["kelly_home"] = kelly["home_kelly"]
        features["kelly_draw"] = kelly["draw_kelly"]
        features["kelly_away"] = kelly["away_kelly"]

        return features

    @staticmethod
    def extract_asia_features(
        odds_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从亚盘数据中提取特征

        Args:
            odds_list: 亚盘数据列表

        Returns:
            特征字典
        """
        features = {
            "asia_company_count": len(odds_list),
            "avg_handicap": None,
            "avg_home_odds": None,
            "avg_away_odds": None,
            "handicap_consensus": None,  # 盘口一致性
        }

        if not odds_list:
            return features

        handicaps = [o["handicap"] for o in odds_list if o.get("handicap")]
        home_odds = [o["home_odds"] for o in odds_list if o.get("home_odds")]
        away_odds = [o["away_odds"] for o in odds_list if o.get("away_odds")]

        if handicaps:
            features["avg_handicap"] = round(np.mean(handicaps), 4)
            # 盘口一致性：如果所有公司盘口相同则为1，否则为0
            features["handicap_consensus"] = 1 if len(set(handicaps)) == 1 else 0

        if home_odds:
            features["avg_home_odds"] = round(np.mean(home_odds), 4)
        if away_odds:
            features["avg_away_odds"] = round(np.mean(away_odds), 4)

        return features

    @staticmethod
    def extract_bigsmall_features(
        odds_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从大小球数据中提取特征

        Args:
            odds_list: 大小球数据列表

        Returns:
            特征字典
        """
        features = {
            "bigsmall_company_count": len(odds_list),
            "avg_total_goals": None,
            "avg_over_odds": None,
            "avg_under_odds": None,
            "over_implied_prob": None,
        }

        if not odds_list:
            return features

        totals = [o["total_goals"] for o in odds_list if o.get("total_goals")]
        over_odds = [o["over_odds"] for o in odds_list if o.get("over_odds")]
        under_odds = [o["under_odds"] for o in odds_list if o.get("under_odds")]

        if totals:
            features["avg_total_goals"] = round(np.mean(totals), 4)
        if over_odds:
            avg_over = np.mean(over_odds)
            features["avg_over_odds"] = round(avg_over, 4)
            features["over_implied_prob"] = round(1 / avg_over, 4)
        if under_odds:
            features["avg_under_odds"] = round(np.mean(under_odds), 4)

        return features

    @staticmethod
    def extract_odds_change_features(
        odds_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        提取赔率变化趋势特征（初盘 vs 即时盘对比）

        Args:
            odds_list: 包含初盘和即时盘的赔率数据列表

        Returns:
            变化趋势特征
        """
        features = {
            "home_odds_trend": 0,   # 1=上升, -1=下降, 0=不变
            "draw_odds_trend": 0,
            "away_odds_trend": 0,
            "home_odds_change_pct": 0,
            "draw_odds_change_pct": 0,
            "away_odds_change_pct": 0,
        }

        if not odds_list:
            return features

        # 使用第一家公司的数据判断趋势
        first = odds_list[0]
        home_current = first.get("home_win")
        home_initial = first.get("initial_home_win")
        draw_current = first.get("draw")
        draw_initial = first.get("initial_draw")
        away_current = first.get("away_win")
        away_initial = first.get("initial_away_win")

        if home_current and home_initial:
            change = home_current - home_initial
            features["home_odds_trend"] = 1 if change > 0 else (-1 if change < 0 else 0)
            features["home_odds_change_pct"] = round(change / home_initial * 100, 4)

        if draw_current and draw_initial:
            change = draw_current - draw_initial
            features["draw_odds_trend"] = 1 if change > 0 else (-1 if change < 0 else 0)
            features["draw_odds_change_pct"] = round(change / draw_initial * 100, 4)

        if away_current and away_initial:
            change = away_current - away_initial
            features["away_odds_trend"] = 1 if change > 0 else (-1 if change < 0 else 0)
            features["away_odds_change_pct"] = round(change / away_initial * 100, 4)

        return features


class FeatureBuilder:
    """特征构建器 - 整合所有特征"""

    def __init__(self):
        self.odds_extractor = OddsFeatureExtractor()

    def build_match_features(
        self,
        match_df: pd.DataFrame,
        odds_dir: str = None,
    ) -> pd.DataFrame:
        """
    为比赛数据构建完整特征集

        Args:
            match_df: 比赛数据DataFrame
            odds_dir: 赔率数据目录

        Returns:
            包含特征的数据DataFrame
        """
        from pathlib import Path
        import json

        if odds_dir is None:
            from config import ODDS_DIR
            odds_dir = ODDS_DIR

        features_list = []

        for _, match in match_df.iterrows():
            gid = match.get("gid")
            if pd.isna(gid):
                continue

            gid = int(gid)
            feature = {"gid": gid}

            # 基本比赛特征
            feature["league_id"] = match.get("league_id")
            feature["home_team"] = match.get("home_team")
            feature["away_team"] = match.get("away_team")

            # 主客场因子
            feature["is_home"] = 1

            # 加载赔率数据
            odds_file = Path(str(odds_dir)) / f"{gid}.json"
            if odds_file.exists():
                with open(odds_file, "r", encoding="utf-8") as f:
                    odds_data = json.load(f)

                # 欧赔特征
                euro_features = self.odds_extractor.extract_europe_features(
                    odds_data.get("europe_odds", [])
                )
                feature.update(euro_features)

                # 亚盘特征
                asia_features = self.odds_extractor.extract_asia_features(
                    odds_data.get("asia_odds", [])
                )
                feature.update(asia_features)

                # 大小球特征
                bigsmall_features = self.odds_extractor.extract_bigsmall_features(
                    odds_data.get("bigsmall_odds", [])
                )
                feature.update(bigsmall_features)

                # 赔率变化趋势
                change_features = self.odds_extractor.extract_odds_change_features(
                    odds_data.get("europe_odds", [])
                )
                feature.update(change_features)

                # 凯利指数分歧（最大-最小）
                if euro_features.get("kelly_home") is not None:
                    kelly_values = [
                        euro_features["kelly_home"],
                        euro_features["kelly_draw"],
                        euro_features["kelly_away"],
                    ]
                    feature["kelly_max_diff"] = round(max(kelly_values) - min(kelly_values), 4)

                # 赔付率（返还率）
                if all([euro_features.get("avg_home_win"),
                        euro_features.get("avg_draw"),
                        euro_features.get("avg_away_win")]):
                    payout = (
                        1 / euro_features["avg_home_win"]
                        + 1 / euro_features["avg_draw"]
                        + 1 / euro_features["avg_away_win"]
                    )
                    feature["payout_rate"] = round(1 / payout, 4) if payout > 0 else None

            features_list.append(feature)

        return pd.DataFrame(features_list) if features_list else pd.DataFrame()


class TargetBuilder:
    """目标变量构建器"""

    @staticmethod
    def build_result_target(df: pd.DataFrame) -> pd.Series:
        """
        构建胜平负目标变量

        Args:
            df: 包含home_score和away_score的DataFrame

        Returns:
            Series: 1=主胜, 0=平局, -1=客胜（或2/1/0）
        """
        def result(row):
            if row["home_score"] > row["away_score"]:
                return 2  # 主胜
            elif row["home_score"] == row["away_score"]:
                return 1  # 平局
            else:
                return 0  # 客胜

        return df.apply(result, axis=1)

    @staticmethod
    def build_handicap_target(
        df: pd.DataFrame,
        handicap: float = 0.0,
    ) -> pd.Series:
        """
        构建让球盘目标变量

        Args:
            df: 包含比分的DataFrame
            handicap: 让球数（正数=主队让球）

        Returns:
            Series: 1=赢盘, 0=走水, -1=输盘
        """
        def result(row):
            adjusted_home = row["home_score"] - handicap
            if adjusted_home > row["away_score"]:
                return 1
            elif adjusted_home == row["away_score"]:
                return 0
            else:
                return -1

        return df.apply(result, axis=1)

    @staticmethod
    def build_total_goals_target(df: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
        """
        构建大小球目标变量

        Args:
            df: 包含比分的DataFrame
            threshold: 进球数阈值

        Returns:
            Series: 1=大球, 0=小球
        """
        total = df["home_score"] + df["away_score"]
        return (total > threshold).astype(int)

    @staticmethod
    def build_score_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建比分预测目标

        Returns:
            DataFrame: 包含home_goals, away_goals, total_goals, goal_diff
        """
        result = pd.DataFrame()
        result["home_goals"] = df["home_score"]
        result["away_goals"] = df["away_score"]
        result["total_goals"] = df["home_score"] + df["away_score"]
        result["goal_diff"] = df["home_score"] - df["away_score"]
        return result
