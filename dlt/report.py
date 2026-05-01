"""
大乐透报告输出 - 终端格式化输出 + 可执行入口
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional

from .crawler import DltCrawler
from .analysis import DltAnalysis
from .predictor import DltPredictor

logger = logging.getLogger(__name__)


class DltReport:
    """大乐透报告生成"""

    def __init__(self):
        self.crawler = DltCrawler()

    def print_predictions(self, groups: int = 5, show_analysis: bool = True):
        """
        打印大乐透预测结果

        Args:
            groups: 生成组数
            show_analysis: 是否显示分析摘要
        """
        # 1. 加载历史数据
        records = self.crawler.get_history()
        if not records:
            print("⚠️ 无历史数据, 请先运行爬取")
            return

        # 2. 分析
        analysis = DltAnalysis(records)
        stats = analysis.generate_summary_stats()

        # 3. 生成号码
        predictor = DltPredictor(analysis)
        results = predictor.generate_groups(groups)

        # 4. 输出
        print()
        print("=" * 56)
        print("       大乐透科学预测推荐")
        print(f"       生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"       基于 {stats['total_periods']} 期历史数据分析")
        print("=" * 56)

        # 推荐号码
        print(f"\n  {'推荐':>4} {'前区号码':<24} {'后区':<10} {'信心'}")
        print("  " + "-" * 42)
        for i, g in enumerate(results, 1):
            front_str = "  ".join(f"{n:02d}" for n in g["front"])
            back_str = "  ".join(f"{n:02d}" for n in g["back"])
            conf_stars = "★" * (g["confidence"] // 20) + "☆" * (5 - g["confidence"] // 20)
            print(f"  #{i:<2}  {front_str}  │ {back_str}  {conf_stars} {g['confidence']}%")

        if show_analysis:
            self._print_analysis(stats, analysis)

    def _print_analysis(self, stats: Dict, analysis: DltAnalysis):
        """打印分析摘要"""
        print(f"\n{'='*56}")
        print("  多维度分析报告")
        print(f"{'='*56}")

        # 冷热号
        hc = stats["hot_cold"]
        print(f"\n  冷热号分布:")
        print(f"    🔥 热号({len(hc['hot'])}个): {', '.join(f'{n:02d}' for n in hc['hot'][:8])}")
        print(f"    ❄️  冷号({len(hc['cold'])}个): {', '.join(f'{n:02d}' for n in hc['cold'][:8])}")

        # 后区常见配对
        bz = stats["back_zone"]
        print(f"\n  后区常见配对 (Top 5):")
        for p in bz["common_pairs"][:5]:
            print(f"    [{p['pair'][0]:02d} {p['pair'][1]:02d}] 出现 {p['count']} 次 ({p['freq']}%)")

        # 奇偶比
        oe = stats["odd_even"]
        print(f"\n  前区奇偶比分布:")
        for ratio, info in sorted(oe["distribution"].items()):
            bar = "█" * int(info["freq"] / 2)
            print(f"    {ratio}  {bar} {info['freq']}%")

        # 和值
        sa = stats["sum_analysis"]
        print(f"\n  前区和值分析:")
        print(f"    范围: {sa['min']} ~ {sa['max']} | 平均: {sa['avg']}")
        print(f"    常见区间: ", end="")
        for r in sa["common_ranges"][:3]:
            print(f"{r['range']}({r['freq']}%)  ", end="")
        print()

        # 连号
        cs = stats["consecutive"]
        print(f"\n  连号分析:")
        print(f"    含连号比例: {cs['consecutive_rate']}%")
        print(f"    无连号比例: {cs['no_consecutive_rate']}%")

        print(f"\n{'='*56}")
        print("  💡 提示: 以上号码基于历史统计分析生成,")
        print("     仅供参考娱乐, 投注需理性!")
        print(f"{'='*56}")
        print()


def setup_encoding():
    """设置控制台编码为UTF-8"""
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main():
    """命令行入口: python -m dlt.report"""
    import argparse
    setup_encoding()

    parser = argparse.ArgumentParser(description="大乐透科学预测系统")
    parser.add_argument("--groups", type=int, default=5, help="生成组数 (默认5)")
    parser.add_argument("--simple", action="store_true", help="简洁模式(不显示分析)")
    parser.add_argument("--update", action="store_true", help="尝试从网络更新数据")

    args = parser.parse_args()

    crawler = DltCrawler()
    if args.update:
        print("正在更新历史数据...")
        new = crawler.update_from_web()
        print(f"更新完成, 新增 {new} 期")

    report = DltReport()
    report.print_predictions(groups=args.groups, show_analysis=not args.simple)


if __name__ == "__main__":
    main()
