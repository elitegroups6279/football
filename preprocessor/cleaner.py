"""
数据清洗与预处理模块

实现类似 sta310_pre 函数的数据预处理功能：
  - 处理缺失值和异常值
  - 数据标准化和格式转换
  - 多重校验确保数据质量
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器"""

    @staticmethod
    def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗比赛数据

        Args:
            df: 原始比赛DataFrame

        Returns:
            清洗后的DataFrame
        """
        df = df.copy()

        # 1. 删除完全重复的行
        df = df.drop_duplicates()
        initial_len = len(df)

        # 2. 处理gid
        if "gid" in df.columns:
            df = df.dropna(subset=["gid"])
            df["gid"] = df["gid"].astype(int)

        # 3. 处理日期时间
        date_columns = ["match_date", "match_time", "matchTime"]
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    continue

        # 4. 处理比分
        score_columns = ["home_score", "away_score"]
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
                df[col] = df[col].replace(-1, np.nan)

        # 5. 处理球队名称
        team_columns = ["home_team", "away_team", "league_name"]
        for col in team_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(["nan", "None", ""], np.nan)

        # 6. 填充缺失的联赛名称
        if "league_name" in df.columns:
            df["league_name"] = df["league_name"].fillna("未知联赛")

        cleaned_len = len(df)
        logger.info(f"数据清洗: {initial_len} -> {cleaned_len} 条 ({initial_len - cleaned_len} 条被移除)")

        return df

    @staticmethod
    def clean_odds_data(odds_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗赔率数据

        Args:
            odds_dict: 原始赔率数据字典

        Returns:
            清洗后的赔率数据
        """
        cleaned = odds_dict.copy()

        # 清洗欧赔
        if "europe_odds" in cleaned:
            cleaned["europe_odds"] = [
                o for o in cleaned["europe_odds"]
                if o.get("home_win") is not None
                or o.get("draw") is not None
                or o.get("away_win") is not None
            ]

        # 清洗亚盘
        if "asia_odds" in cleaned:
            cleaned["asia_odds"] = [
                o for o in cleaned["asia_odds"]
                if o.get("handicap") is not None
            ]

        # 清洗大小球
        if "bigsmall_odds" in cleaned:
            cleaned["bigsmall_odds"] = [
                o for o in cleaned["bigsmall_odds"]
                if o.get("total_goals") is not None
            ]

        return cleaned

    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
        """
        使用IQR方法检测异常值

        Args:
            df: 数据DataFrame
            column: 列名
            factor: IQR倍数（默认1.5）

        Returns:
            布尔序列，True表示异常值
        """
        if column not in df.columns:
            return pd.Series([False] * len(df))

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        return (df[column] < lower_bound) | (df[column] > upper_bound)

    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], factor: float = 3.0) -> pd.DataFrame:
        """
        移除异常值

        Args:
            df: 数据DataFrame
            columns: 要检测的列名列表
            factor: Z-score阈值

        Returns:
            移除异常值后的DataFrame
        """
        df = df.copy()
        initial_len = len(df)

        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if pd.isna(std) or std == 0:
                    continue
                z_scores = (df[col] - mean) / std
                df = df[z_scores.abs() <= factor]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"异常值移除: 移除了 {removed} 条记录 (columns={columns})")

        return df

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """
        填充缺失值

        Args:
            df: 数据DataFrame
            strategy: 填充策略（mean/median/mode/zero）

        Returns:
            填充后的DataFrame
        """
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    fill_value = df[col].mean()
                elif strategy == "median":
                    fill_value = df[col].median()
                elif strategy == "mode":
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                elif strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = 0

                df[col] = df[col].fillna(fill_value)
                count = df[col].isnull().sum()
                if count > 0:
                    df[col] = df[col].fillna(0)

        logger.info(f"缺失值填充完成 (strategy={strategy})")
        return df

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名（统一命名规范）

        Returns:
            列名标准化后的DataFrame
        """
        df = df.copy()

        # 常见字段名映射
        column_mapping = {
            "homeTeamName": "home_team",
            "awayTeamName": "away_team",
            "homeTeamScore": "home_score",
            "awayTeamScore": "away_score",
            "leagueName": "league_name",
            "matchTime": "match_time",
            "matchDate": "match_date",
            "matchStatus": "status",
            "letBall": "handicap",
            "homeWin": "home_win",
            "awayWin": "away_win",
        }

        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        return df


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_match_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证比赛数据质量

        Returns:
            数据质量报告
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "duplicate_count": 0,
            "validity": {},
        }

        # 检查缺失值
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report["missing_values"][col] = {
                    "count": int(missing),
                    "percentage": round(missing / len(df) * 100, 2),
                }

        # 检查重复
        if "gid" in df.columns:
            report["duplicate_count"] = int(df["gid"].duplicated().sum())

        # 检查数据有效性
        if "home_score" in df.columns:
            invalid_scores = df[df["home_score"].notna() & (df["home_score"] < 0)].shape[0]
            report["validity"]["negative_scores"] = int(invalid_scores)

        return report
