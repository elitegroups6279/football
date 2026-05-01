"""
报告生成器 - 将分析结果汇总为HTML报告
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import REPORTS_DIR

logger = logging.getLogger(__name__)


class ReportGenerator:
    """EDA报告生成器"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else REPORTS_DIR

    def generate_html_report(
        self,
        weekday_stats: pd.DataFrame = None,
        goal_stats: pd.DataFrame = None,
        league_goals: pd.DataFrame = None,
        league_results: pd.DataFrame = None,
        company_stats: pd.DataFrame = None,
        team_freq: pd.DataFrame = None,
        home_advantage: pd.DataFrame = None,
        upset_matches: pd.DataFrame = None,
        handicap_stats: pd.DataFrame = None,
        charts: Optional[dict] = None,
    ) -> str:
        """
        生成HTML格式的EDA分析报告

        Returns:
            HTML文件路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        charts = charts or {}

        # 构建HTML内容
        html_parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>足球数据EDA分析报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; padding-left: 10px; border-left: 4px solid #3498db; }}
        h3 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4f8; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .summary {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; font-size: 12px; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
        .badge-primary {{ background: #3498db; color: white; }}
        .badge-success {{ background: #27ae60; color: white; }}
        .badge-warning {{ background: #f39c12; color: white; }}
        .badge-danger {{ background: #e74c3c; color: white; }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚽ 足球数据探索性分析(EDA)报告</h1>
    <p>生成时间: {timestamp}</p>
"""]

        # ===== 1. 数据概览 =====
        html_parts.append('<h2>一、数据概览</h2><div class="summary">')

        if weekday_stats is not None and not weekday_stats.empty:
            total = int(weekday_stats["match_count"].sum())
            html_parts.append(f"<p>总比赛场次: <strong>{total}</strong></p>")

        if league_goals is not None:
            html_parts.append(f"<p>涉及联赛数: <strong>{len(league_goals)}</strong></p>")

        if team_freq is not None:
            html_parts.append(f"<p>涉及球队数: <strong>{len(team_freq)}</strong></p>")

        html_parts.append('</div>')

        # ===== 2. 统计分析 =====
        html_parts.append('<h2>二、统计分析</h2>')

        # 2.1 星期分布
        if weekday_stats is not None and not weekday_stats.empty:
            html_parts.append('<h3>2.1 比赛星期分布</h3>')
            html_parts.append(weekday_stats.to_html(index=False, classes="display"))
            if "weekday_distribution.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["weekday_distribution.png"]}" alt="星期分布"></div>')

        # 2.2 进球分布
        if goal_stats is not None and not goal_stats.empty:
            html_parts.append('<h3>2.2 进球数分布</h3>')
            display_df = goal_stats[goal_stats["total_goals"] <= 10].copy()
            html_parts.append(display_df.to_html(index=False, classes="display"))
            if "goal_distribution.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["goal_distribution.png"]}" alt="进球分布"></div>')

        # 2.3 联赛平均进球
        if league_goals is not None and not league_goals.empty:
            html_parts.append('<h3>2.3 联赛平均进球数 (Top 20)</h3>')
            html_parts.append(league_goals.head(20).to_html(index=False, classes="display"))
            if "league_avg_goals.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["league_avg_goals.png"]}" alt="联赛平均进球"></div>')

        # 2.4 联赛胜平负分布
        if league_results is not None and not league_results.empty:
            html_parts.append('<h3>2.4 联赛胜平负分布</h3>')
            html_parts.append(league_results.head(20).to_html(index=False, classes="display"))
            if "league_result_distribution.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["league_result_distribution.png"]}" alt="联赛胜平负"></div>')

        # ===== 3. 赔率分析 =====
        html_parts.append('<h2>三、赔率相关分析</h2>')

        if company_stats is not None and not company_stats.empty:
            html_parts.append('<h3>3.1 博彩公司赔率对比</h3>')
            html_parts.append(company_stats.to_html(index=False, classes="display"))
            if "company_odds_comparison.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["company_odds_comparison.png"]}" alt="公司赔率对比"></div>')

        if handicap_stats is not None and not handicap_stats.empty:
            html_parts.append('<h3>3.2 让球盘口胜率</h3>')
            html_parts.append(handicap_stats.to_html(index=False, classes="display"))
            if "handicap_win_rate.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["handicap_win_rate.png"]}" alt="让球胜率"></div>')

        if upset_matches is not None and not upset_matches.empty:
            html_parts.append('<h3>3.3 爆冷比赛分析</h3>')
            html_parts.append(f"<p>共发现 <strong>{len(upset_matches)}</strong> 场爆冷比赛</p>")
            html_parts.append(upset_matches.head(30).to_html(index=False, classes="display"))

        # ===== 4. 球队统计 =====
        html_parts.append('<h2>四、球队相关统计</h2>')

        if team_freq is not None and not team_freq.empty:
            html_parts.append('<h3>4.1 球队比赛频率 (Top 20)</h3>')
            html_parts.append(team_freq.head(20).to_html(index=False, classes="display"))
            if "team_activity.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["team_activity.png"]}" alt="球队活跃度"></div>')

        if home_advantage is not None and not home_advantage.empty:
            html_parts.append('<h3>4.2 球队主场优势 (Top 15)</h3>')
            html_parts.append(home_advantage.head(15).to_html(index=False, classes="display"))
            if "home_advantage.png" in (charts or {}):
                html_parts.append(f'<div class="chart"><img src="{charts["home_advantage.png"]}" alt="主场优势"></div>')

        # 尾部
        html_parts.append(f"""
    <div class="footer">
        <p>Generated by Football Data Analysis System | {timestamp}</p>
    </div>
</div>
</body>
</html>
""")

        # 写入文件
        report_path = self.output_dir / "eda_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info(f"EDA报告已生成: {report_path}")
        return str(report_path)
