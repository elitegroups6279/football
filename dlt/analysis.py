"""
大乐透多维度统计分析引擎

分析维度:
  1. 频率分析 - 每个号码的历史出现次数
  2. 冷热号分析 - 近期活跃度
  3. 遗漏值分析 - 连续未出期数（越久越可能回补）
  4. 奇偶比分布 - 前区奇数/偶数比例
  5. 大小号分布 - 前区大号(18-35)/小号(1-17)比例
  6. 和值分析 - 前区5码之和
  7. 连号分析 - 相邻号码出现概率
  8. 后区分析 - 后区号码配对规律
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DltAnalysis:
    """大乐透多维度统计分析"""

    FRONT_RANGE = range(1, 36)   # 前区 1-35
    BACK_RANGE = range(1, 13)    # 后区 1-12
    SMALL_MAX = 17               # 小号上限
    BIG_MIN = 18                 # 大号下限
    RECENT_WEIGHT = 20           # 近期窗口期数

    def __init__(self, records: List[Dict]):
        """
        Args:
            records: 历史开奖数据 [{period, date, front:[5], back:[2]}]
        """
        self.records = records
        self.total = len(records)
        self.front_data = [r["front"] for r in records]
        self.back_data = [r["back"] for r in records]
        self.recent_data = self.front_data[:self.RECENT_WEIGHT] if self.total >= self.RECENT_WEIGHT else self.front_data.copy()

    # ==================== 1. 频率分析 ====================

    def frequency_analysis(self) -> Dict[str, Dict]:
        """
        计算每个号码的历史出现频率

        Returns:
            {front: {num: count, freq, rank}, back: {...}}
        """
        front_counter = Counter()
        for nums in self.front_data:
            front_counter.update(nums)

        back_counter = Counter()
        for nums in self.back_data:
            back_counter.update(nums)

        result = {
            "front": {},
            "back": {},
        }

        for num in self.FRONT_RANGE:
            count = front_counter.get(num, 0)
            result["front"][num] = {
                "count": count,
                "freq": round(count / self.total, 4) if self.total > 0 else 0,
            }

        for num in self.BACK_RANGE:
            count = back_counter.get(num, 0)
            result["back"][num] = {
                "count": count,
                "freq": round(count / self.total, 4) if self.total > 0 else 0,
            }

        return result

    def _get_hottest_numbers(self, n: int = 10) -> List[int]:
        """获取前区出现频率最高的N个号码"""
        counter = Counter()
        for nums in self.front_data:
            counter.update(nums)
        return [num for num, _ in counter.most_common(n)]

    # ==================== 2. 冷热号分析 ====================

    def hot_cold_analysis(self) -> Dict[str, List[int]]:
        """
        冷热号分类: 基于最近20期出现频率

        Returns:
            {hot: [...], warm: [...], cold: [...]}
        """
        counter = Counter()
        for nums in self.front_data:
            counter.update(nums)

        recent_counter = Counter()
        for nums in self.recent_data:
            recent_counter.update(nums)

        # 大乐透35选5, 每期每个号码出现概率 5/35 ≈ 14.3%
        # 20期中期望出现 ~2.86次
        # 热号: 明显高于平均 (>= 4次)
        # 温号: 正常范围 (1-3次)
        # 冷号: 未出现 (0次)
        hot, warm, cold = [], [], []
        for num in self.FRONT_RANGE:
            rc = recent_counter.get(num, 0)
            if rc >= 4:
                hot.append(num)
            elif rc >= 1:
                warm.append(num)
            else:
                cold.append(num)

        return {"hot": sorted(hot), "warm": sorted(warm), "cold": sorted(cold)}

    # ==================== 3. 遗漏值分析 ====================

    def omission_analysis(self) -> Dict[int, int]:
        """
        遗漏值分析: 每个号码连续未出现的期数

        遗漏值越大, 理论上回补概率越高

        Returns:
            {num: omission_periods}
        """
        omission = {}
        for num in self.FRONT_RANGE:
            count = 0
            for nums in self.front_data:
                if num in nums:
                    break
                count += 1
            omission[num] = count

        return omission

    # ==================== 4. 奇偶比分布 ====================

    def odd_even_analysis(self) -> Dict:
        """
        前区奇偶比分布分析

        Returns:
            {distribution: {ratio: count, freq}, current_trend: ...}
        """
        ratio_counter = Counter()
        for nums in self.front_data:
            odd_count = sum(1 for n in nums if n % 2 == 1)
            even_count = 5 - odd_count
            ratio = f"{odd_count}:{even_count}"
            ratio_counter[ratio] += 1

        distribution = {}
        for ratio, count in ratio_counter.most_common():
            distribution[ratio] = {
                "count": count,
                "freq": round(count / self.total * 100, 1),
            }

        # 最近10期的奇偶趋势
        recent_ratios = []
        for nums in self.recent_data[:10]:
            odd_count = sum(1 for n in nums if n % 2 == 1)
            recent_ratios.append(f"{odd_count}:{5-odd_count}")

        return {
            "distribution": distribution,
            "most_common": ratio_counter.most_common(1)[0][0] if ratio_counter else "3:2",
            "recent_trend": recent_ratios,
        }

    # ==================== 5. 大小号分布 ====================

    def big_small_analysis(self) -> Dict:
        """
        前区大小号(大:18-35, 小:1-17)分布

        Returns:
            {distribution: {...}, most_common, recent_trend}
        """
        ratio_counter = Counter()
        for nums in self.front_data:
            big_count = sum(1 for n in nums if n >= self.BIG_MIN)
            small_count = 5 - big_count
            ratio = f"{big_count}:{small_count}"
            ratio_counter[ratio] += 1

        distribution = {}
        for ratio, count in ratio_counter.most_common():
            distribution[ratio] = {
                "count": count,
                "freq": round(count / self.total * 100, 1),
            }

        recent_ratios = []
        for nums in self.recent_data[:10]:
            big_count = sum(1 for n in nums if n >= self.BIG_MIN)
            recent_ratios.append(f"{big_count}:{5-big_count}")

        return {
            "distribution": distribution,
            "most_common": ratio_counter.most_common(1)[0][0] if ratio_counter else "2:3",
            "recent_trend": recent_ratios,
        }

    # ==================== 6. 和值分析 ====================

    def sum_analysis(self) -> Dict:
        """
        前区和值分布 (5码之和)

        Returns:
            {min, max, avg, common_ranges: [{range, count, freq}]}
        """
        sums = [sum(nums) for nums in self.front_data]

        # 和值范围统计 (以20为区间)
        range_counter = Counter()
        for s in sums:
            r = (s // 20) * 20
            range_counter[f"{r}-{r+19}"] += 1

        common_ranges = []
        for r, count in range_counter.most_common(5):
            common_ranges.append({
                "range": r,
                "count": count,
                "freq": round(count / self.total * 100, 1),
            })

        return {
            "min": min(sums) if sums else 0,
            "max": max(sums) if sums else 0,
            "avg": round(sum(sums) / len(sums), 1) if sums else 0,
            "common_ranges": common_ranges,
        }

    # ==================== 7. 连号分析 ====================

    def consecutive_analysis(self) -> Dict:
        """
        连号分析: 相邻号码(差值为1)的出现概率

        Returns:
            {has_consecutive_pct, max_consecutive_dist, most_common_positions}
        """
        has_cons = 0
        cons_groups = []

        for nums in self.front_data:
            sorted_nums = sorted(nums)
            cons_count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i + 1] - sorted_nums[i] == 1:
                    cons_count += 1
            if cons_count > 0:
                has_cons += 1
                cons_groups.append(cons_count)

        return {
            "consecutive_rate": round(has_cons / self.total * 100, 1) if self.total > 0 else 0,
            "avg_consecutive_pairs": round(sum(cons_groups) / self.total, 2) if self.total > 0 else 0,
            "no_consecutive_rate": round((self.total - has_cons) / self.total * 100, 1) if self.total > 0 else 0,
        }

    # ==================== 8. 后区分析 ====================

    def back_zone_analysis(self) -> Dict:
        """
        后区号码综合分析

        Returns:
            {freq: {...}, odd_even: {...}, common_pairs: [...]}
        """
        # 频率
        counter = Counter()
        for nums in self.back_data:
            counter.update(nums)

        freq = {}
        for num in self.BACK_RANGE:
            freq[num] = {
                "count": counter.get(num, 0),
                "freq": round(counter.get(num, 0) / self.total, 4) if self.total > 0 else 0,
            }

        # 后区奇偶搭配
        oe_counter = Counter()
        for nums in self.back_data:
            odd = sum(1 for n in nums if n % 2 == 1)
            oe_counter[f"{odd}:{2-odd}"] += 1

        # 常见配对
        pair_counter = Counter()
        for nums in self.back_data:
            pair = tuple(sorted(nums))
            pair_counter[pair] += 1

        common_pairs = []
        for pair, count in pair_counter.most_common(10):
            common_pairs.append({
                "pair": list(pair),
                "count": count,
                "freq": round(count / self.total * 100, 1),
            })

        return {
            "frequency": freq,
            "odd_even": {k: v for k, v in oe_counter.most_common()},
            "common_pairs": common_pairs,
        }

    # ==================== 综合权重计算 ====================

    def compute_front_weights(self) -> Dict[int, float]:
        """
        计算前区每个号码的综合权重

        权重 = α * freq_score + β * cold_score + γ * recent_score + δ * neighbor_score

        Returns:
            {num: weight (0-100)}
        """
        freq = self.frequency_analysis()["front"]
        omission = self.omission_analysis()
        hot_cold = self.hot_cold_analysis()
        hot_set = set(hot_cold["hot"])
        cold_set = set(hot_cold["cold"])

        weights = {}

        # α: 频率权重
        max_freq = max(v["freq"] for v in freq.values()) if freq else 1

        # γ: 近期频率（最近20期）
        recent_counter = Counter()
        for nums in self.recent_data:
            recent_counter.update(nums)
        max_recent = max(recent_counter.values()) if recent_counter else 1

        for num in self.FRONT_RANGE:
            # α: 历史频率得分 (0-100)
            freq_score = (freq[num]["freq"] / max_freq * 100) if max_freq > 0 else 0

            # β: 遗漏值得分 (遗漏越久越高)
            omit = omission[num]
            cold_score = min(100, omit * 3)  # 每遗漏一期+3分, 上限100

            # γ: 近期热度 (最近20期)
            recent_count = recent_counter.get(num, 0)
            recent_score = (recent_count / max_recent * 100) if max_recent > 0 else 0

            # δ: 邻号关联 (出现在近期的附近号码)
            neighbor_score = 0
            for recent_nums in self.recent_data:
                for n in recent_nums:
                    if abs(n - num) <= 2:
                        neighbor_score += 5
            neighbor_score = min(100, neighbor_score)

            # 综合权重
            weight = (0.30 * freq_score + 0.25 * cold_score +
                      0.25 * recent_score + 0.20 * neighbor_score)

            # 如果是热号, 额外加权
            if num in hot_set:
                weight *= 1.15
            # 如果是冷号, 给一定的补偿分(不放弃但降低)
            elif num in cold_set and cold_score > 30:
                weight *= 1.05

            weights[num] = round(min(100, weight), 2)

        return weights

    def compute_back_weights(self) -> Dict[int, float]:
        """
        计算后区每个号码的综合权重

        Returns:
            {num: weight (0-100)}
        """
        counter = Counter()
        for nums in self.back_data:
            counter.update(nums)
        max_count = max(counter.values()) if counter else 1

        # 遗漏值
        omission = {}
        for num in self.BACK_RANGE:
            count = 0
            for nums in self.back_data:
                if num in nums:
                    break
                count += 1
            omission[num] = count

        weights = {}
        for num in self.BACK_RANGE:
            freq_score = (counter.get(num, 0) / max_count * 100) if max_count > 0 else 0
            cold_score = min(100, omission.get(num, 0) * 5)  # 每遗漏一期+5分
            weight = 0.5 * freq_score + 0.5 * cold_score
            weights[num] = round(min(100, weight), 2)

        return weights

    def generate_summary_stats(self) -> Dict:
        """
        生成完整统计分析摘要

        Returns:
            包含所有分析维度的字典
        """
        return {
            "total_periods": self.total,
            "front_frequency": self.frequency_analysis(),
            "hot_cold": self.hot_cold_analysis(),
            "omission": self.omission_analysis(),
            "odd_even": self.odd_even_analysis(),
            "big_small": self.big_small_analysis(),
            "sum_analysis": self.sum_analysis(),
            "consecutive": self.consecutive_analysis(),
            "back_zone": self.back_zone_analysis(),
        }
