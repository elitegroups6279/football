#!/usr/bin/env python
"""
天天盈球足球数据爬虫与分析系统 - 主程序入口

功能概述：
  1. 数据爬取：从天天盈球网站爬取足球比赛和赔率数据
  2. 数据清洗：处理缺失值、异常值，标准化数据格式
  3. 特征工程：从赔率数据中提取特征，构建预测目标
  4. EDA分析：统计分析、赔率分析、球队统计
  5. 报告生成：生成HTML格式的分析报告

使用方法：
  python main.py --scrape          # 爬取数据
  python main.py --odds            # 爬取赔率
  python main.py --eda             # 执行EDA分析
  python main.py --all             # 执行全部流程
  python main.py --report          # 生成分析报告

配置文件：config.py
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from config import (
    DATA_DIR, MATCHES_DIR, ODDS_DIR, REPORTS_DIR,
    PREDICTIONS_DIR,
    HOT_LEAGUE_IDS, DEFAULT_BANKROLL, MIN_EV_THRESHOLD,
    MAX_RECOMMENDATIONS,
)
from scraper.match_scraper import MatchScraper
from scraper.odds_scraper import OddsScraper
from preprocessor.cleaner import DataCleaner, DataValidator
from preprocessor.feature_engineer import FeatureBuilder, TargetBuilder, OddsFeatureExtractor
from eda.analysis import (
    MatchStatistics, OddsAnalysis, TeamStatistics,
    EDAVisualizer,
)
from eda.report_generator import ReportGenerator
from prediction.betting_recommender import BettingRecommender


def setup_encoding():
    """设置控制台编码为UTF-8"""
    import io
    import sys
    # 重设标准输出编码
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def setup_logging(level=logging.INFO):
    """配置日志"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_scrape(args):
    """爬取比赛数据"""
    print("=" * 60)
    print("  开始爬取足球比赛数据")
    print("=" * 60)

    scraper = MatchScraper(use_api=args.use_api)

    if args.today:
        print("\n>> 爬取今日比赛...")
        df = scraper.scrape_today_matches()
    elif args.league_id:
        print(f"\n>> 爬取联赛 {args.league_id} 的比赛...")
        df = scraper.scrape_league_matches(args.league_id, max_pages=args.pages)
    elif args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        print(f"\n>> 爬取 {start_date} 到 {end_date} 的比赛...")
        df = scraper.scrape_matches_by_date_range(start_date, end_date)
    else:
        # 默认：爬取热门联赛数据
        print("\n>> 爬取热门联赛数据...")
        leagues = scraper.scrape_hot_leagues()
        print(f"  发现 {len(leagues)} 个热门联赛")

        all_matches = []
        for league in leagues[:args.max_leagues]:
            lid = league.get("id")
            name = league.get("name", "")
            if not lid:
                continue
            print(f"  爬取: {name} (ID={lid})...")
            df = scraper.scrape_league_matches(lid, name, max_pages=args.pages)
            if not df.empty:
                all_matches.append(df)

        if all_matches:
            import pandas as pd
            df = pd.concat(all_matches, ignore_index=True)
            df = df.drop_duplicates(subset=["gid"], keep="first") if "gid" in df.columns else df
            # 保存汇总
            scraper._save_matches(df, "all_leagues")
        else:
            df = None

    scraper.close()

    if df is not None and not df.empty:
        print(f"\n✅ 成功获取 {len(df)} 条比赛数据")
        print(f"   数据保存在: {MATCHES_DIR}")
    else:
        print("\n⚠️ 未获取到比赛数据")
        print("   提示: 网站可能有反爬机制，建议:")
        print("   1. 检查网络连接")
        print("   2. 尝试不同的联赛ID")
        print("   3. 使用 --use-api 参数切换API模式")


def cmd_odds(args):
    """爬取赔率数据"""
    print("=" * 60)
    print("  开始爬取赔率数据")
    print("=" * 60)

    odds_scraper = OddsScraper()

    # 从已有的比赛数据中读取gid
    csv_files = sorted(MATCHES_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csv_files:
        print(f"\n>> 从 {csv_files[0].name} 中读取比赛ID")
        results = odds_scraper.scrape_odds_from_match_csv(
            str(csv_files[0]),
            max_matches=args.max_matches,
        )
    elif args.gid:
        print(f"\n>> 爬取指定比赛 {args.gid} 的赔率")
        odds = odds_scraper.scrape_match_odds(args.gid)
        if odds:
            odds_scraper._save_odds_to_file(odds)
            results = {args.gid: odds}
        else:
            results = {}
    else:
        print("⚠️ 未找到比赛数据文件，请先运行爬取命令")
        print("   或者使用 --gid 指定比赛ID")
        results = {}

    odds_scraper.close()

    if results:
        print(f"\n✅ 成功获取 {len(results)} 场比赛的赔率数据")
        print(f"   数据保存在: {ODDS_DIR}")
    else:
        print("\n⚠️ 未获取到赔率数据")


def cmd_eda(args):
    """执行EDA分析"""
    print("=" * 60)
    print("  执行探索性数据分析(EDA)")
    print("=" * 60)

    # 查找最新的比赛数据
    csv_files = sorted(MATCHES_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        print("⚠️ 未找到比赛数据文件")
        print("   请先运行: python main.py --scrape")
        return

    import pandas as pd
    df = pd.read_csv(csv_files[0])
    print(f"\n>> 加载数据: {csv_files[0].name} ({len(df)}条)")

    # 清洗数据
    cleaner = DataCleaner()
    df = cleaner.clean_match_data(df)
    print(f">> 清洗后: {len(df)}条")

    # 验证数据质量
    validator = DataValidator()
    report = validator.validate_match_data(df)
    print(f">> 数据质量报告: {report['total_rows']}行, {report['total_columns']}列")
    if report["missing_values"]:
        print(f">> 缺失值: {report['missing_values']}")

    # ===== 统计分析 =====
    print("\n--- 1. 统计分析 ---")
    stats = MatchStatistics()

    # 1.1 星期分布
    weekday_stats = stats.weekday_distribution(df)
    if not weekday_stats.empty:
        print("\n[比赛星期分布]")
        print(weekday_stats.to_string(index=False))
        max_day = weekday_stats.loc[weekday_stats["match_count"].idxmax()]
        print(f"  最多比赛日: {max_day['weekday']} ({int(max_day['match_count'])}场)")

    # 1.2 进球分布
    goal_stats = stats.goal_distribution(df)
    if not goal_stats.empty:
        print("\n[进球数分布]")
        print(goal_stats.head(11).to_string(index=False))

    # 1.3 联赛平均进球
    league_goals = stats.league_avg_goals(df)
    if not league_goals.empty:
        print("\n[联赛平均进球 Top 10]")
        print(league_goals.head(10).to_string(index=False))

    # 1.4 联赛胜平负分布
    league_results = stats.league_result_distribution(df)
    if not league_results.empty:
        print("\n[联赛胜平负分布 Top 10]")
        print(league_results.head(10).to_string(index=False))

    # ===== 赔率分析 =====
    print("\n--- 2. 赔率分析 ---")
    odds_analysis = OddsAnalysis()

    # 2.1 公司赔率对比
    odds_files = list(ODDS_DIR.glob("*.json"))
    if odds_files:
        company_stats = odds_analysis.company_odds_comparison(str(ODDS_DIR))
        if not company_stats.empty:
            print("\n[博彩公司赔率对比]")
            print(company_stats.to_string(index=False))

        # 2.2 爆冷分析
        upset_matches = odds_analysis.upset_odds_analysis(df, str(ODDS_DIR))
        if not upset_matches.empty:
            print(f"\n[爆冷比赛] 发现 {len(upset_matches)} 场")

        # 2.3 让球盘口胜率
        handicap_stats = odds_analysis.handicap_win_rate(df, str(ODDS_DIR))
        if not handicap_stats.empty:
            print("\n[让球盘口胜率]")
            print(handicap_stats.to_string(index=False))

    # ===== 球队统计 =====
    print("\n--- 3. 球队统计 ---")
    team_stats = TeamStatistics()

    team_freq = team_stats.team_match_frequency(df)
    if not team_freq.empty:
        print("\n[最活跃球队 Top 10]")
        print(team_freq.head(10).to_string(index=False))

    home_advantage = team_stats.team_home_advantage(df)
    if not home_advantage.empty:
        print("\n[主场优势 Top 10]")
        print(home_advantage.head(10).to_string(index=False))

    # ===== 生成图表 =====
    print("\n--- 4. 生成可视化图表 ---")
    visualizer = EDAVisualizer()
    charts = {}

    if not weekday_stats.empty:
        path = visualizer.plot_weekday_distribution(weekday_stats)
        charts["weekday_distribution.png"] = path
        print(f"  [OK] 星期分布图: {path}")

    if not goal_stats.empty:
        path = visualizer.plot_goal_distribution(goal_stats)
        charts["goal_distribution.png"] = path
        print(f"  [OK] 进球分布图: {path}")

    if league_goals is not None and not league_goals.empty:
        path = visualizer.plot_league_avg_goals(league_goals)
        charts["league_avg_goals.png"] = path
        print(f"  [OK] 联赛平均进球图: {path}")

    if league_results is not None and not league_results.empty:
        path = visualizer.plot_league_result_distribution(league_results)
        charts["league_result_distribution.png"] = path
        print(f"  [OK] 联赛胜平负分布图: {path}")

    if odds_files:
        if not company_stats.empty:
            path = visualizer.plot_company_comparison(company_stats)
            charts["company_odds_comparison.png"] = path
            print(f"  [OK] 公司赔率对比图: {path}")

        if not handicap_stats.empty:
            path = visualizer.plot_handicap_win_rate(handicap_stats)
            charts["handicap_win_rate.png"] = path
            print(f"  [OK] 让球胜率图: {path}")

    if not team_freq.empty:
        path = visualizer.plot_team_activity(team_freq)
        charts["team_activity.png"] = path
        print(f"  [OK] 球队活跃度图: {path}")

    if home_advantage is not None and not home_advantage.empty:
        path = visualizer.plot_home_advantage(home_advantage)
        charts["home_advantage.png"] = path
        print(f"  [OK] 主场优势图: {path}")

    # ===== 生成报告 =====
    if args.report:
        print("\n--- 5. 生成报告 ---")
        reporter = ReportGenerator()
        report_path = reporter.generate_html_report(
            weekday_stats=weekday_stats,
            goal_stats=goal_stats,
            league_goals=league_goals,
            league_results=league_results,
            company_stats=company_stats if odds_files else None,
            team_freq=team_freq,
            home_advantage=home_advantage,
            upset_matches=upset_matches if odds_files else None,
            handicap_stats=handicap_stats if odds_files else None,
            charts=charts,
        )
        print(f"\n✅ EDA报告已生成: {report_path}")

    print("\n✅ EDA分析完成！")


def cmd_all(args):
    """执行完整流程：爬取 + 赔率 + EDA"""
    print("=" * 60)
    print("  完整数据分析流程")
    print("=" * 60)

    # Step 1: 爬取数据
    args.today = True
    cmd_scrape(args)

    # Step 2: 爬取赔率
    cmd_odds(args)

    # Step 3: EDA分析
    args.report = True
    cmd_eda(args)

    print("\n" + "=" * 60)
    print("  全流程完成！")
    print("=" * 60)


def cmd_feature(args):
    """执行特征工程"""
    print("=" * 60)
    print("  特征工程")
    print("=" * 60)

    import pandas as pd

    # 加载比赛数据
    csv_files = sorted(MATCHES_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        print("⚠️ 未找到比赛数据")
        return

    df = pd.read_csv(csv_files[0])
    cleaner = DataCleaner()
    df = cleaner.clean_match_data(df)
    print(f">> 加载比赛数据: {len(df)}条")

    # 构建特征
    builder = FeatureBuilder()
    features_df = builder.build_match_features(df)

    if not features_df.empty:
        print(f">> 构建特征: {len(features_df)}条, {len(features_df.columns)}列")
        print(f">> 特征列: {list(features_df.columns)}")

        # 保存特征
        output_path = DATA_DIR / "features.csv"
        features_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f">> 特征已保存: {output_path}")

    # 构建目标变量
    if all(col in df.columns for col in ["home_score", "away_score"]):
        df_clean = df.dropna(subset=["home_score", "away_score"])
        target = TargetBuilder.build_result_target(df_clean)
        print(f">> 目标变量(胜平负)构建完成: {len(target)}条")

        handicap = TargetBuilder.build_handicap_target(df_clean)
        print(f">> 目标变量(让球盘)构建完成: {len(handicap)}条")

    print("\n✅ 特征工程完成")


def cmd_predict(args):
    """执行竞彩预测"""
    print("=" * 60)
    print("  竞彩足球投注预测")
    print("=" * 60)

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"\n>> 预测日期: {date_str}")
    print(f">> 本金: {args.bankroll}元")
    print(f">> 最小EV: {args.min_ev * 100}%")
    print(f">> 最大推荐: {args.max_rec}场")

    recommender = BettingRecommender()

    try:
        if args.summary:
            # 文字版摘要
            summary = recommender.generate_summary(date_str)
            print(summary)
        else:
            # 完整版推荐
            df = recommender.generate_recommendations(
                date_str=date_str,
                bankroll=args.bankroll,
                min_ev=args.min_ev,
                max_rec=args.max_rec,
            )

            if df.empty:
                print(f"\n⚠️ {date_str} 无符合条件的高价值投注推荐")
                print("   可能原因: 当日无竞彩赛事 / EV未达阈值 / API数据异常")
                print(f"   可以尝试: --min-ev 0.02 降低阈值, 或 --date 指定其他日期")
            else:
                print(f"\n✅ 发现 {len(df)} 场价值投注:\n")
                for idx, (_, row) in enumerate(df.iterrows(), 1):
                    print(f"  [{idx}] {row['主队']} vs {row['客队']}")
                    print(f"     推荐: {row['推荐选项']} @ {row['市场赔率']}")
                    print(f"     EV: {row['期望值(EV)']}% | 概率: {row['预测概率']}% | 信心: {row['信心分']}%")
                    print(f"     建议投注: {row['建议投注']:.0f}元")
                    print()

                # 生成HTML报告
                report_path = recommender.generate_html_report(df, date_str)
                if report_path:
                    print(f">> HTML报告: {report_path}")
    finally:
        recommender.close()


def cmd_update_history(args):
    """更新历史比赛数据"""
    print("=" * 60)
    print("  更新历史比赛数据")
    print("=" * 60)

    recommender = BettingRecommender()

    try:
        # 默认获取最近30天
        days = args.days if args.days > 0 else 30
        count = recommender.update_historical_data(days=days)
        print(f"\n✅ 历史数据更新完成, 新增 {count} 条记录")
        print(f"   数据保存在: {Path('data') / 'historical' / 'match_results.csv'}")
    finally:
        recommender.close()


def main():
    parser = argparse.ArgumentParser(
        description="天天盈球足球数据爬虫与分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --scrape                      # 爬取热门联赛数据
  python main.py --scrape --today              # 爬取今日比赛
  python main.py --scrape --league-id 2079     # 爬取指定联赛(英超)
  python main.py --scrape --days 30            # 爬取近30天数据
  python main.py --odds                        # 爬取赔率
  python main.py --odds --gid 12345            # 爬取指定比赛赔率
  python main.py --eda                         # EDA分析
  python main.py --eda --report                # EDA分析+生成报告
  python main.py --feature                     # 特征工程
  python main.py --predict                     # 竞彩预测
  python main.py --predict --summary           # 竞彩预测(文字版)
  python main.py --predict --bankroll 2000     # 自定义本金
  python main.py --update-history              # 更新历史数据
  python main.py --all                         # 全流程
        """,
    )

    # 爬取参数
    parser.add_argument("--scrape", action="store_true", help="爬取比赛数据")
    parser.add_argument("--today", action="store_true", help="仅爬取今日比赛")
    parser.add_argument("--league-id", type=int, default=0, help="指定联赛ID")
    parser.add_argument("--days", type=int, default=0, help="爬取最近N天数据")
    parser.add_argument("--pages", type=int, default=3, help="爬取页数（默认3）")
    parser.add_argument("--max-leagues", type=int, default=10, help="最大联赛数（默认10）")
    parser.add_argument("--use-api", action="store_true", default=True, help="使用API模式")

    # 赔率参数
    parser.add_argument("--odds", action="store_true", help="爬取赔率数据")
    parser.add_argument("--gid", type=int, default=0, help="指定比赛ID爬取赔率")
    parser.add_argument("--max-matches", type=int, default=50, help="最大爬取比赛数（默认50）")

    # EDA参数
    parser.add_argument("--eda", action="store_true", help="执行EDA分析")
    parser.add_argument("--report", action="store_true", help="生成HTML报告")

    # 特征工程
    parser.add_argument("--feature", action="store_true", help="执行特征工程")

    # 预测参数
    parser.add_argument("--predict", action="store_true", help="执行竞彩预测")
    parser.add_argument("--date", type=str, default="", help="预测日期 (YYYY-MM-DD)")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="投注本金（默认1000元）")
    parser.add_argument("--min-ev", type=float, default=MIN_EV_THRESHOLD, help="最小EV阈值（默认0.05）")
    parser.add_argument("--max-rec", type=int, default=MAX_RECOMMENDATIONS, help="最大推荐场次（默认10）")
    parser.add_argument("--summary", action="store_true", help="仅输出文字摘要")

    # 历史数据
    parser.add_argument("--update-history", action="store_true", help="更新历史比赛数据")

    # 全流程
    parser.add_argument("--all", action="store_true", help="执行完整流程")

    # 日志
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()

    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    setup_encoding()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # 执行命令
    if args.all:
        cmd_all(args)
    elif args.scrape:
        cmd_scrape(args)
    elif args.odds:
        cmd_odds(args)
    elif args.eda:
        cmd_eda(args)
    elif args.feature:
        cmd_feature(args)
    elif args.predict:
        cmd_predict(args)
    elif args.update_history:
        cmd_update_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
