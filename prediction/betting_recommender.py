"""
投注推荐引擎 - 核心编排器

Pipeline:
  1. fetch: 从API获取竞彩比赛列表和赔率
  2. predict: 用泊松模型预测每场比赛
  3. evaluate: 计算EV, 凯利投注额
  4. score: 置信度评分
  5. rank: 按EV排名, 过滤阈值
  6. report: 生成HTML报告
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

from config import (
    PREDICTIONS_DIR, HISTORICAL_DIR, ODDS_DIR,
    DEFAULT_BANKROLL, MIN_EV_THRESHOLD, DEFAULT_KELLY_FRACTION,
    MAX_RECOMMENDATIONS,
)
from prediction.jingcai_api import JingcaiAPI
from prediction.margin_removal import MarginRemover
from prediction.poisson_model import PoissonPredictor
from prediction.value_betting import ValueBetting
from prediction.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class BettingRecommender:
    """投注推荐引擎"""

    def __init__(self):
        self.api = JingcaiAPI()
        self.poisson = PoissonPredictor()
        self.value = ValueBetting()
        self.scorer = ConfidenceScorer()
        self.remover = MarginRemover()

    def fetch_and_predict(
        self,
        date_str: Optional[str] = None,
    ) -> List[Dict]:
        """
        获取竞彩数据 -> 泊松预测 -> EV评估

        Args:
            date_str: 日期 YYYY-MM-DD

        Returns:
            每场比赛的完整预测结果列表
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 1. 获取比赛
        matches = self.api.get_jingcai_matches(date_str)
        if not matches:
            logger.warning(f"{date_str} 无竞彩比赛数据")
            return []

        logger.info(f"开始分析 {len(matches)} 场竞彩比赛...")
        predictions = []

        for match in matches:
            gid = match.get("gid")
            if not gid:
                continue

            # 2. 赔率获取 (优先从match自带赔率, 不足则单独请求)
            home_odds = match.get("home_win_odds")
            draw_odds = match.get("draw_odds")
            away_odds = match.get("away_win_odds")

            odds_data = None
            odds_list = []

            # 如果match自带的赔率不完整, 从API获取
            if not all([home_odds, draw_odds, away_odds]):
                odds_data = self.api.get_match_odds(gid)
                if odds_data and odds_data.get("europe_odds"):
                    euro = odds_data["europe_odds"]
                    odds_list = [o["home_win"] for o in euro if o.get("home_win")]
                    # 用平均赔率
                    if odds_list:
                        home_odds = float(np.mean(odds_list))
                    draw_list = [o["draw"] for o in euro if o.get("draw")]
                    draw_odds = float(np.mean(draw_list)) if draw_list else None
                    away_list = [o["away_win"] for o in euro if o.get("away_win")]
                    away_odds = float(np.mean(away_list)) if away_list else None

            if not all([home_odds, draw_odds, away_odds]):
                logger.debug(f"比赛 {gid} 赔率数据不足, 跳过")
                continue

            home_odds = float(home_odds)
            draw_odds = float(draw_odds)
            away_odds = float(away_odds)

            # 3. 边际消除 (用三种方法)
            probs_basic = self.remover.remove_margin(home_odds, draw_odds, away_odds, "basic")
            probs_power = self.remover.remove_margin(home_odds, draw_odds, away_odds, "power")
            probs_shin = self.remover.remove_margin(home_odds, draw_odds, away_odds, "shin")

            # 4. 泊松预测
            poisson_result = self.poisson.predict_from_odds(
                home_odds, draw_odds, away_odds, margin_method="shin",
            )

            # 5. 寻找最佳投注 (用泊松概率作为真实概率)
            outcome = poisson_result["outcome"]
            best_bet = self.value.find_best_bet(
                outcome["home_win"], outcome["draw"], outcome["away_win"],
                home_odds, draw_odds, away_odds,
            )

            # 6. 置信度评分因子
            factors = self.scorer.compute_factors(
                ev=best_bet.get("ev", 0),
                odds_list=odds_list if odds_list else [home_odds],
                predicted_home_prob=outcome["home_win"],
                market_home_prob=probs_shin["home_prob"],
                league_avg_goals=None,
                margin_probs_list=[probs_basic, probs_power, probs_shin],
            )
            confidence = self.scorer.score(factors)

            # 7. 汇总
            prediction = {
                "match_info": {
                    "gid": gid,
                    "league": match.get("league_name", ""),
                    "home_team": match.get("home_team", ""),
                    "away_team": match.get("away_team", ""),
                    "match_time": match.get("match_time", ""),
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                },
                "true_prob": {
                    "home": probs_shin["home_prob"],
                    "draw": probs_shin["draw_prob"],
                    "away": probs_shin["away_prob"],
                },
                "poisson": poisson_result,
                "best_bet": best_bet,
                "confidence": confidence,
            }
            predictions.append(prediction)

        return predictions

    def generate_recommendations(
        self,
        date_str: Optional[str] = None,
        bankroll: float = DEFAULT_BANKROLL,
        min_ev: float = MIN_EV_THRESHOLD,
        max_rec: int = MAX_RECOMMENDATIONS,
    ) -> pd.DataFrame:
        """
        生成投注推荐表

        Args:
            date_str: 日期
            bankroll: 本金
            min_ev: 最小EV
            max_rec: 最多推荐数

        Returns:
            推荐DataFrame
        """
        predictions = self.fetch_and_predict(date_str)
        if not predictions:
            return pd.DataFrame()

        # 计算实际投注额
        for p in predictions:
            bet = p.get("best_bet", {})
            prob = bet.get("predicted_prob", 0)
            odds = bet.get("market_odds", 0)
            kelly = self.value.calculate_kelly_stake(prob, odds, bankroll)
            bet["stake_amount"] = kelly["stake_amount"]

        # 过滤有价值的
        rows = []
        for p in predictions:
            bet = p.get("best_bet", {})
            ev = bet.get("ev", 0)
            if ev >= min_ev:
                info = p["match_info"]
                rows.append({
                    "gid": info["gid"],
                    "联赛": info["league"],
                    "主队": info["home_team"],
                    "客队": info["away_team"],
                    "开赛时间": info["match_time"],
                    "推荐选项": bet.get("pick", ""),
                    "预测概率": round(bet.get("predicted_prob", 0) * 100, 1),
                    "市场赔率": bet.get("market_odds", 0),
                    "公平赔率": bet.get("fair_odds", 0),
                    "期望值(EV)": round(bet.get("ev", 0) * 100, 1),
                    "边际优势": round(bet.get("edge", 0) * 100, 1),
                    "建议投注": bet.get("stake_amount", 0),
                    "凯利比例": round(bet.get("kelly_fraction", 0) * 100, 1),
                    "置信度": f"{p.get('confidence', 0)}%",
                    "信心分": p.get("confidence", 0),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("期望值(EV)", ascending=False).head(max_rec)
        return df.reset_index(drop=True)

    def generate_summary(self, date_str: Optional[str] = None) -> str:
        """生成简短文字版推荐总结"""
        df = self.generate_recommendations(date_str)
        if df.empty:
            return f"\n📅 {date_str or datetime.now().strftime('%Y-%m-%d')}: 暂无高价值投注推荐\n"

        lines = [
            f"\n{'='*60}",
            f"  竞彩足球投注推荐 - {date_str or datetime.now().strftime('%Y-%m-%d')}",
            f"{'='*60}",
            f"  共 {len(df)} 场推荐\n",
        ]

        for _, row in df.iterrows():
            lines.append(
                f"  [{row['信心分']}分] {row['主队']} vs {row['客队']}\n"
                f"    推荐: {row['推荐选项']} @ {row['市场赔率']}\n"
                f"    EV: {row['期望值(EV)']}% | 预测概率: {row['预测概率']}%\n"
                f"    建议投注: {row['建议投注']:.0f}元\n"
            )

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def generate_html_report(
        self,
        recommendations: pd.DataFrame,
        date_str: Optional[str] = None,
    ) -> str:
        """
        生成HTML格式投注推荐报告

        Args:
            recommendations: 推荐DataFrame
            date_str: 日期

        Returns:
            HTML文件路径
        """
        if recommendations.empty:
            return ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_label = date_str or datetime.now().strftime("%Y-%m-%d")

        # 汇总统计
        total_value = recommendations["建议投注"].sum()
        avg_ev = recommendations["期望值(EV)"].mean()
        avg_conf = recommendations["信心分"].mean()
        max_conf = recommendations["信心分"].max()

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>竞彩足球投注推荐 - {date_label}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #c0392b; margin-top: 30px; padding-left: 10px; border-left: 4px solid #e74c3c; }}
        .summary {{ background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; display: flex; justify-content: space-around; text-align: center; }}
        .summary-item h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .summary-item .value {{ font-size: 28px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 8px; text-align: center; }}
        th {{ background-color: #e74c3c; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #fdf2f2; }}
        tr:hover {{ background-color: #fadbd8; }}
        .high-conf {{ background-color: #d5f5e3 !important; }}
        .mid-conf {{ background-color: #fef9e7 !important; }}
        .pick {{ display: inline-block; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 12px; }}
        .pick-home {{ background: #e74c3c; color: white; }}
        .pick-draw {{ background: #f39c12; color: white; }}
        .pick-away {{ background: #3498db; color: white; }}
        .stake {{ font-weight: bold; color: #27ae60; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; font-size: 12px; }}
        .disclaimer {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 20px 0; border-radius: 4px; font-size: 13px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚽ 竞彩足球投注推荐报告</h1>
    <p>日期: {date_label} | 生成时间: {timestamp}</p>

    <div class="disclaimer">
        <strong>⚠️ 免责声明:</strong> 本报告仅基于历史数据和统计分析生成, 不构成投注建议。
        彩票有风险, 投注需谨慎。请理性购彩, 切勿沉迷。
    </div>

    <div class="summary">
        <div class="summary-item">
            <h3>推荐场次</h3>
            <div class="value">{len(recommendations)}</div>
        </div>
        <div class="summary-item">
            <h3>总投注额</h3>
            <div class="value">{total_value:.0f}元</div>
        </div>
        <div class="summary-item">
            <h3>平均EV</h3>
            <div class="value">{avg_ev:.1f}%</div>
        </div>
        <div class="summary-item">
            <h3>平均信心</h3>
            <div class="value">{avg_conf:.0f}%</div>
        </div>
        <div class="summary-item">
            <h3>最高信心</h3>
            <div class="value">{max_conf:.0f}%</div>
        </div>
    </div>

    <h2>投注推荐详情</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>联赛</th>
                <th>主队 vs 客队</th>
                <th>推荐</th>
                <th>预测概率</th>
                <th>赔率</th>
                <th>EV</th>
                <th>建议投注</th>
                <th>信心</th>
            </tr>
        </thead>
        <tbody>
"""
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            pick = row.get("推荐选项", "")
            pick_class = {"主胜": "pick-home", "平局": "pick-draw", "客胜": "pick-away"}.get(pick, "")
            conf = row.get("信心分", 0)
            row_class = "high-conf" if conf >= 70 else ("mid-conf" if conf >= 50 else "")

            html += f"""            <tr class="{row_class}">
                <td>{idx}</td>
                <td>{row.get('联赛', '')}</td>
                <td>{row.get('主队', '')} vs {row.get('客队', '')}</td>
                <td><span class="pick {pick_class}">{pick}</span></td>
                <td>{row.get('预测概率', '')}%</td>
                <td>{row.get('市场赔率', '')}</td>
                <td>{row.get('期望值(EV)', '')}%</td>
                <td class="stake">{row.get('建议投注', 0):.0f}元</td>
                <td>{conf}%</td>
            </tr>
"""

        html += """        </tbody>
    </table>
"""

        # 详细说明
        if not recommendations.empty:
            html += """    <h2>单项详情</h2>
"""
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                html += f"""    <div style="background: #fdf2f2; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 4px solid #e74c3c;">
        <strong>#{idx} {row.get('主队', '')} vs {row.get('客队', '')}</strong><br>
        <span style="color: #666; font-size: 13px;">
            推荐: {row.get('推荐选项', '')} @ {row.get('市场赔率', '')} |
            公平赔率: {row.get('公平赔率', '')} |
            预测概率: {row.get('预测概率', '')}% |
            EV: {row.get('期望值(EV)', '')}% |
            投注: {row.get('建议投注', 0):.0f}元 |
            凯利: {row.get('凯利比例', '')}%
        </span>
    </div>
"""

        html += f"""
    <div class="footer">
        <p>Generated by Football Prediction System | {timestamp}</p>
        <p>数据来源: 天天盈球 | 分析方法: 泊松分布 + 凯利公式</p>
    </div>
</div>
</body>
</html>
"""

        # 写文件
        report_path = PREDICTIONS_DIR / f"recommendations_{date_label}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"推荐报告已生成: {report_path}")
        return str(report_path)

    def update_historical_data(self, days: int = 30) -> int:
        """
        更新历史比赛数据, 用于模型训练

        Args:
            days: 往回获取的天数

        Returns:
            新增的比赛记录数
        """
        from datetime import timedelta

        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
        hist_file = HISTORICAL_DIR / "match_results.csv"

        # 加载已有的历史数据
        existing_gids = set()
        if hist_file.exists():
            existing_df = pd.read_csv(hist_file)
            existing_gids = set(existing_df["gid"].dropna().unique())
            logger.info(f"已有历史数据: {len(existing_df)}条, {len(existing_gids)}个比赛")

        new_records = []
        today = datetime.now()

        for day_offset in range(days):
            date_str = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            matches = self.api.get_jingcai_matches(date_str)
            if not matches:
                continue

            for m in matches:
                gid = m.get("gid")
                if not gid or gid in existing_gids:
                    continue
                if not m.get("is_played") and m.get("status") != 2:
                    continue
                if m.get("home_score") is None or m.get("away_score") is None:
                    continue

                new_records.append({
                    "gid": gid,
                    "date": date_str,
                    "league_name": m.get("league_name", ""),
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                    "home_win_odds": m.get("home_win_odds"),
                    "draw_odds": m.get("draw_odds"),
                    "away_win_odds": m.get("away_win_odds"),
                })
                existing_gids.add(gid)

        if new_records:
            new_df = pd.DataFrame(new_records)

            if hist_file.exists():
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["gid"], keep="last")
                combined.to_csv(hist_file, index=False, encoding="utf-8-sig")
            else:
                new_df.to_csv(hist_file, index=False, encoding="utf-8-sig")

            logger.info(f"新增 {len(new_records)} 条历史记录, 总计存储于 {hist_file}")
        else:
            logger.info("无新增历史记录")

        return len(new_records)

    def close(self):
        """清理资源"""
        self.api.close()
