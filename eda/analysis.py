"""
探索性数据分析(EDA)模块

统计分析功能：
1. 不同星期几的比赛数量分布
2. 比赛进球数分布统计
3. 各联赛的平均进球数统计
4. 不同联赛的胜平负概率分布

赔率相关分析：
5. 不同博彩公司之间赔率差异性分析
6. 强队vs弱队的让球盘口胜率分析
7. 爆冷门比赛的赛前赔率特征分析

球队相关统计：
8. 各球队作为主队和客队的比赛次数统计
9. 识别比赛频率最高的球队
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 尝试导入seaborn（可选）
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from config import REPORTS_DIR

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MatchStatistics:
    """比赛数据统计分析"""

    @staticmethod
    def weekday_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        统计不同星期几的比赛数量分布

        预期结果：周六的比赛数量最多

        Args:
            df: 包含match_date列的DataFrame

        Returns:
            星期分布统计表
        """
        if "match_date" not in df.columns:
            logger.warning("缺少match_date列")
            return pd.DataFrame()

        date_col = pd.to_datetime(df["match_date"], errors="coerce")
        weekdays = date_col.dt.dayofweek  # 0=周一

        weekday_names = {
            0: "周一", 1: "周二", 2: "周三", 3: "周四",
            4: "周五", 5: "周六", 6: "周日",
        }
        distribution = weekdays.value_counts().sort_index()
        distribution.index = distribution.index.map(weekday_names)
        distribution.name = "match_count"

        result = distribution.reset_index()
        result.columns = ["weekday", "match_count"]
        result["percentage"] = (result["match_count"] / result["match_count"].sum() * 100).round(2)

        logger.info("星期分布统计完成")
        logger.info(f"  周六比赛数: {result.loc[result['weekday']=='周六', 'match_count'].values}")
        return result

    @staticmethod
    def goal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        统计比赛进球数分布

        预期结果：总进球1球和2球的比赛最多

        Args:
            df: 包含home_score和away_score的DataFrame

        Returns:
            进球数分布表
        """
        if not all(col in df.columns for col in ["home_score", "away_score"]):
            logger.warning("缺少比分列")
            return pd.DataFrame()

        total_goals = df["home_score"] + df["away_score"]
        distribution = total_goals.value_counts().sort_index().reset_index()
        distribution.columns = ["total_goals", "match_count"]
        distribution["percentage"] = (distribution["match_count"] / distribution["match_count"].sum() * 100).round(2)

        logger.info(f"进球数分布统计完成 (0-{int(distribution['total_goals'].max())}球)")
        return distribution

    @staticmethod
    def league_avg_goals(df: pd.DataFrame) -> pd.DataFrame:
        """
        统计各联赛的平均进球数

        Args:
            df: 包含league_name, home_score, away_score的DataFrame

        Returns:
            联赛平均进球数统计表
        """
        if not all(col in df.columns for col in ["league_name", "home_score", "away_score"]):
            logger.warning("缺少联赛名或比分列")
            return pd.DataFrame()

        df = df.dropna(subset=["home_score", "away_score"])

        stats = df.groupby("league_name").agg(
            match_count=("home_score", "count"),
            avg_home_goals=("home_score", "mean"),
            avg_away_goals=("away_score", "mean"),
            avg_total_goals=("home_score", lambda x: (x + df.loc[x.index, "away_score"]).mean()),
            std_total_goals=("home_score", lambda x: (x + df.loc[x.index, "away_score"]).std()),
        ).reset_index()

        stats["avg_home_goals"] = stats["avg_home_goals"].round(2)
        stats["avg_away_goals"] = stats["avg_away_goals"].round(2)
        stats["avg_total_goals"] = stats["avg_total_goals"].round(2)
        stats["std_total_goals"] = stats["std_total_goals"].round(2)

        stats = stats.sort_values("avg_total_goals", ascending=False)
        logger.info(f"联赛平均进球统计完成 ({len(stats)}个联赛)")
        return stats

    @staticmethod
    def league_result_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        统计不同联赛的胜平负概率分布

        Args:
            df: 包含league_name, home_score, away_score的DataFrame

        Returns:
            联赛胜平负概率分布表
        """
        if not all(col in df.columns for col in ["league_name", "home_score", "away_score"]):
            logger.warning("缺少联赛名或比分列")
            return pd.DataFrame()

        def get_result(row):
            if row["home_score"] > row["away_score"]:
                return "主胜"
            elif row["home_score"] == row["away_score"]:
                return "平局"
            else:
                return "客胜"

        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["result"] = df.apply(get_result, axis=1)

        # 交叉统计
        crosstab = pd.crosstab(
            df["league_name"], df["result"],
            normalize="index",
        ).round(4) * 100

        # 添加比赛计数
        counts = df.groupby("league_name").size()
        crosstab["match_count"] = counts

        # 确保三列都存在
        for col in ["主胜", "平局", "客胜"]:
            if col not in crosstab.columns:
                crosstab[col] = 0.0

        crosstab = crosstab[["match_count", "主胜", "平局", "客胜"]].fillna(0)
        crosstab.columns = ["match_count", "home_win_pct", "draw_pct", "away_win_pct"]
        crosstab = crosstab.sort_values("match_count", ascending=False)
        crosstab = crosstab[crosstab["match_count"] >= 5]  # 过滤样本量太少的联赛

        logger.info(f"联赛胜平负分布统计完成 ({len(crosstab)}个联赛)")
        return crosstab.reset_index()


class OddsAnalysis:
    """赔率相关分析"""

    @staticmethod
    def company_odds_comparison(
        odds_dir: str,
        gid_list: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        不同博彩公司之间赔率差异性分析

        Args:
            odds_dir: 赔率数据目录
            gid_list: 比赛ID列表，默认分析所有

        Returns:
            各公司赔率统计对比表
        """
        from config import BOOKMAKER_NAMES

        odds_path = Path(odds_dir)
        data_by_company: Dict[int, List[float]] = {}

        files = list(odds_path.glob("*.json"))
        if gid_list:
            files = [f for f in files if int(f.stem) in gid_list]

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                odds_data = json.load(f)

            for odd in odds_data.get("europe_odds", []):
                cid = odd.get("company_id")
                hw = odd.get("home_win")
                if cid and hw:
                    if cid not in data_by_company:
                        data_by_company[cid] = []
                    data_by_company[cid].append(hw)

        # 构建统计表
        stats = []
        for cid, odds_list in data_by_company.items():
            if len(odds_list) < 3:
                continue
            stats.append({
                "company_id": cid,
                "company_name": BOOKMAKER_NAMES.get(cid, f"公司{cid}"),
                "sample_count": len(odds_list),
                "avg_odds": round(np.mean(odds_list), 4),
                "std_odds": round(np.std(odds_list), 4),
                "min_odds": round(min(odds_list), 4),
                "max_odds": round(max(odds_list), 4),
            })

        result = pd.DataFrame(stats).sort_values("sample_count", ascending=False)
        logger.info(f"公司赔率对比分析完成 ({len(result)}家公司)")
        return result

    @staticmethod
    def upset_odds_analysis(
        df: pd.DataFrame,
        odds_dir: str,
        upset_threshold: float = 2.5,
    ) -> pd.DataFrame:
        """
        爆冷门比赛的赛前赔率特征分析

        爆冷定义：主胜赔率 > upset_threshold 但主队获胜

        Args:
            df: 包含比赛结果的数据
            odds_dir: 赔率数据目录
            upset_threshold: 爆冷赔率阈值

        Returns:
            爆冷比赛特征分析表
        """
        from pathlib import Path

        upset_matches = []

        for _, match in df.iterrows():
            gid = match.get("gid")
            if pd.isna(gid):
                continue

            odds_file = Path(odds_dir) / f"{int(gid)}.json"
            if not odds_file.exists():
                continue

            with open(odds_file, "r", encoding="utf-8") as f:
                odds_data = json.load(f)

            euro = odds_data.get("europe_odds", [])
            if not euro:
                continue

            avg_home_win = np.mean([o["home_win"] for o in euro if o.get("home_win")])

            home_score = match.get("home_score")
            away_score = match.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            is_upset = False
            upset_type = ""

            # 主胜爆冷：高赔率下主队获胜
            if avg_home_win > upset_threshold and home_score > away_score:
                is_upset = True
                upset_type = "主胜爆冷"

            # 客胜爆冷：主胜赔率很低但客队获胜
            if avg_home_win < 1.5 and away_score > home_score:
                is_upset = True
                upset_type = "客胜爆冷"

            if is_upset:
                upset_matches.append({
                    "gid": int(gid),
                    "league": match.get("league_name"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "avg_home_win": round(avg_home_win, 4),
                    "upset_type": upset_type,
                })

        result = pd.DataFrame(upset_matches)
        logger.info(f"爆冷分析完成: 发现 {len(result)} 场爆冷比赛")
        return result

    @staticmethod
    def handicap_win_rate(
        df: pd.DataFrame,
        odds_dir: str,
    ) -> pd.DataFrame:
        """
        强队vs弱队的让球盘口胜率分析

        Args:
            df: 比赛数据
            odds_dir: 赔率目录

        Returns:
            让球盘口胜率统计
        """
        from pathlib import Path

        handicap_stats = []

        for _, match in df.iterrows():
            gid = match.get("gid")
            if pd.isna(gid):
                continue

            odds_file = Path(odds_dir) / f"{int(gid)}.json"
            if not odds_file.exists():
                continue

            with open(odds_file, "r", encoding="utf-8") as f:
                odds_data = json.load(f)

            asia = odds_data.get("asia_odds", [])
            if not asia:
                continue

            avg_handicap = np.mean([o["handicap"] for o in asia if o.get("handicap")])
            home_score = match.get("home_score")
            away_score = match.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            adjusted_home = home_score - avg_handicap
            if adjusted_home > away_score:
                result_against_hcap = "赢盘"
            elif adjusted_home == away_score:
                result_against_hcap = "走水"
            else:
                result_against_hcap = "输盘"

            handicap_stats.append({
                "gid": int(gid),
                "avg_handicap": avg_handicap,
                "handicap_group": round(avg_handicap * 2) / 2,  # 按0.5分组
                "result_vs_handicap": result_against_hcap,
            })

        if not handicap_stats:
            return pd.DataFrame()

        stats_df = pd.DataFrame(handicap_stats)
        summary = stats_df.groupby("handicap_group").agg(
            total=("result_vs_handicap", "count"),
            win_count=("result_vs_handicap", lambda x: (x == "赢盘").sum()),
            push_count=("result_vs_handicap", lambda x: (x == "走水").sum()),
            lose_count=("result_vs_handicap", lambda x: (x == "输盘").sum()),
        ).reset_index()

        summary["win_rate"] = (summary["win_count"] / summary["total"] * 100).round(2)
        summary = summary.sort_values("handicap_group")

        logger.info(f"让球盘口胜率分析完成")
        return summary


class TeamStatistics:
    """球队相关统计分析"""

    @staticmethod
    def team_match_frequency(df: pd.DataFrame) -> pd.DataFrame:
        """
        统计各球队作为主队和客队的比赛次数

        Args:
            df: 包含home_team和away_team的DataFrame

        Returns:
            球队比赛频率统计
        """
        if not all(col in df.columns for col in ["home_team", "away_team"]):
            logger.warning("缺少球队列")
            return pd.DataFrame()

        # 主客场统计
        home_counts = df["home_team"].value_counts().reset_index()
        home_counts.columns = ["team", "home_matches"]

        away_counts = df["away_team"].value_counts().reset_index()
        away_counts.columns = ["team", "away_matches"]

        # 合并
        stats = home_counts.merge(away_counts, on="team", how="outer").fillna(0)
        stats["home_matches"] = stats["home_matches"].astype(int)
        stats["away_matches"] = stats["away_matches"].astype(int)
        stats["total_matches"] = stats["home_matches"] + stats["away_matches"]
        stats = stats.sort_values("total_matches", ascending=False)

        logger.info(f"球队频率统计完成 ({len(stats)}支球队)")
        return stats.reset_index(drop=True)

    @staticmethod
    def most_active_teams(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        识别比赛频率最高的球队

        Args:
            df: 比赛数据
            top_n: 返回前N支球队

        Returns:
            最活跃球队列表
        """
        freq_stats = TeamStatistics.team_match_frequency(df)
        return freq_stats.head(top_n)

    @staticmethod
    def team_home_advantage(df: pd.DataFrame, min_matches: int = 5) -> pd.DataFrame:
        """
        计算各球队的主场优势

        Args:
            df: 比赛数据
            min_matches: 最小比赛数

        Returns:
            球队主场优势统计
        """
        if not all(col in df.columns for col in ["home_team", "away_team", "home_score", "away_score"]):
            return pd.DataFrame()

        df = df.dropna(subset=["home_score", "away_score"]).copy()

        # 主队场均进球
        home_stats = df.groupby("home_team").agg(
            home_matches=("home_score", "count"),
            home_goals=("home_score", "mean"),
            home_conceded=("away_score", "mean"),
            home_wins=("home_score", lambda x: (x > df.loc[x.index, "away_score"]).sum()),
        ).reset_index()

        # 客队场均进球
        away_stats = df.groupby("away_team").agg(
            away_matches=("away_score", "count"),
            away_goals=("away_score", "mean"),
            away_conceded=("home_score", "mean"),
            away_wins=("away_score", lambda x: (x > df.loc[x.index, "home_score"]).sum()),
        ).reset_index()

        merged = home_stats.merge(
            away_stats,
            left_on="home_team",
            right_on="away_team",
            how="inner",
        )
        merged.columns = [
            "team", "home_matches", "home_goals_avg", "home_conceded_avg", "home_wins",
            "_away_team", "away_matches", "away_goals_avg", "away_conceded_avg", "away_wins",
        ]
        merged = merged.drop(columns=["_away_team"])

        merged["total_matches"] = merged["home_matches"] + merged["away_matches"]
        merged = merged[merged["total_matches"] >= min_matches]

        merged["home_advantage_goals"] = (merged["home_goals_avg"] - merged["away_goals_avg"]).round(2)
        merged["home_win_rate"] = (merged["home_wins"] / merged["home_matches"] * 100).round(2)
        merged["away_win_rate"] = (merged["away_wins"] / merged["away_matches"] * 100).round(2)
        merged["home_advantage_wr"] = (merged["home_win_rate"] - merged["away_win_rate"]).round(2)

        merged = merged.sort_values("home_advantage_goals", ascending=False)

        logger.info(f"球队主场优势分析完成 ({len(merged)}支球队)")
        return merged.reset_index(drop=True)


class EDAVisualizer:
    """EDA可视化工具"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_weekday_distribution(self, weekday_stats: pd.DataFrame) -> str:
        """绘制星期分布柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#FF6B6B" if w == "周六" else "#4ECDC4" for w in weekday_stats["weekday"]]
        bars = ax.bar(weekday_stats["weekday"], weekday_stats["match_count"], color=colors)

        for bar, count in zip(bars, weekday_stats["match_count"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(count)), ha="center", va="bottom", fontsize=10)

        ax.set_xlabel("星期", fontsize=12)
        ax.set_ylabel("比赛数量", fontsize=12)
        ax.set_title("不同星期比赛数量分布", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.tight_layout()

        path = self.output_dir / "weekday_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"图表已保存: {path}")
        return str(path)

    def plot_goal_distribution(self, goal_stats: pd.DataFrame) -> str:
        """绘制进球数分布柱状图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        goals = goal_stats["total_goals"]
        counts = goal_stats["match_count"]

        # 只显示0-10球范围
        mask = goals <= 10
        goals = goals[mask]
        counts = counts[mask]

        bars = ax.bar(goals, counts, color="#45B7D1", edgecolor="white")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(count)), ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("总进球数", fontsize=12)
        ax.set_ylabel("比赛数量", fontsize=12)
        ax.set_title("比赛总进球数分布", fontsize=14, fontweight="bold")
        ax.set_xticks(range(int(goals.min()), int(goals.max()) + 1))
        plt.tight_layout()

        path = self.output_dir / "goal_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_league_avg_goals(self, league_stats: pd.DataFrame, top_n: int = 20) -> str:
        """绘制联赛平均进球数（前N名）"""
        top = league_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.4)))
        bars = ax.barh(range(len(top)), top["avg_total_goals"].values, color="#96CEB4")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["league_name"].values)
        ax.set_xlabel("平均总进球数", fontsize=12)
        ax.set_title(f"联赛平均进球数 (Top {top_n})", fontsize=14, fontweight="bold")

        for bar, val in zip(bars, top["avg_total_goals"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="left", va="center", fontsize=9)

        plt.tight_layout()
        path = self.output_dir / "league_avg_goals.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_league_result_distribution(self, result_stats: pd.DataFrame, top_n: int = 15) -> str:
        """绘制联赛胜平负堆叠柱状图"""
        top = result_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(14, max(6, len(top) * 0.5)))

        x = range(len(top))
        width = 0.6

        ax.barh(x, top["home_win_pct"].values, height=width, label="主胜", color="#FF6B6B")
        ax.barh(x, top["draw_pct"].values, height=width, left=top["home_win_pct"].values,
                label="平局", color="#FFE66D")
        ax.barh(x, top["away_win_pct"].values, height=width,
                left=top["home_win_pct"].values + top["draw_pct"].values,
                label="客胜", color="#4ECDC4")

        ax.set_yticks(list(x))
        ax.set_yticklabels(top["league_name"].values)
        ax.set_xlabel("百分比 (%)", fontsize=12)
        ax.set_title("各联赛胜平负分布", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")

        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row["home_win_pct"] / 2, i, f"{row['home_win_pct']:.1f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax.text(row["home_win_pct"] + row["draw_pct"] / 2, i, f"{row['draw_pct']:.1f}%",
                    ha="center", va="center", fontsize=8)
            ax.text(row["home_win_pct"] + row["draw_pct"] + row["away_win_pct"] / 2, i,
                    f"{row['away_win_pct']:.1f}%", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        path = self.output_dir / "league_result_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_company_comparison(self, company_stats: pd.DataFrame, top_n: int = 10) -> str:
        """绘制博彩公司平均赔率对比"""
        top = company_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top)), top["avg_odds"].values, color="#A8D8EA", edgecolor="white")

        for bar, val in zip(bars, top["avg_odds"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top["company_name"].values, rotation=30, ha="right")
        ax.set_ylabel("平均主胜赔率", fontsize=12)
        ax.set_title("各博彩公司平均赔率对比", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / "company_odds_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_handicap_win_rate(self, handicap_stats: pd.DataFrame) -> str:
        """绘制让球盘口胜率图"""
        if handicap_stats.empty:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(handicap_stats["handicap_group"].values,
                handicap_stats["win_rate"].values, "o-", color="#FF6B6B", linewidth=2)

        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%基准线")
        ax.set_xlabel("平均让球数 (正=主队让球)", fontsize=12)
        ax.set_ylabel("赢盘率 (%)", fontsize=12)
        ax.set_title("不同盘口下的赢盘率", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "handicap_win_rate.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_team_activity(self, team_stats: pd.DataFrame, top_n: int = 20) -> str:
        """绘制最活跃球队图"""
        top = team_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(top))
        width = 0.35

        ax.bar([i - width / 2 for i in x], top["home_matches"].values, width,
               label="主场", color="#FF6B6B")
        ax.bar([i + width / 2 for i in x], top["away_matches"].values, width,
               label="客场", color="#4ECDC4")

        ax.set_xticks(list(x))
        ax.set_xticklabels(top["team"].values, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("比赛数量", fontsize=12)
        ax.set_title(f"最活跃球队 (Top {top_n})", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / "team_activity.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def plot_home_advantage(self, team_stats: pd.DataFrame, top_n: int = 15) -> str:
        """绘制主场优势图"""
        top = team_stats.head(top_n)

        fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.4)))
        colors = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in top["home_advantage_goals"]]
        bars = ax.barh(range(len(top)), top["home_advantage_goals"].values, color=colors)

        for bar, val in zip(bars, top["home_advantage_goals"]):
            ax.text(bar.get_width() + 0.02 if val > 0 else bar.get_width() - 0.15,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="left" if val > 0 else "right", va="center", fontsize=9)

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["team"].values)
        ax.set_xlabel("主场进球优势", fontsize=12)
        ax.set_title("球队主场优势 (主场场均进球 - 客场场均进球)", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
        plt.tight_layout()

        path = self.output_dir / "home_advantage.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
