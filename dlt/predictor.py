"""
大乐透号码生成器 - 综合多因子加权随机生成

生成流程:
  1. 从Analysis获取每个号码的综合权重
  2. 加权随机抽样选出5个前区号码
  3. 约束过滤 (奇偶比、大小比、和值范围)
  4. 不满足约束则重新抽样
  5. 同样流程生成后区2个号码
  6. 重复生成N组
"""

import logging
import random
from typing import Dict, List, Tuple, Optional

from .analysis import DltAnalysis

logger = logging.getLogger(__name__)


class DltPredictor:
    """大乐透号码生成器"""

    # 合理的奇偶比范围 (前区)
    VALID_ODD_EVEN_RATIOS = {"3:2", "2:3", "4:1", "1:4"}
    # 合理的大小比范围
    VALID_BIG_SMALL_RATIOS = {"3:2", "2:3", "4:1", "1:4", "5:0", "0:5"}
    # 合理的和值范围 (前区5码之和)
    SUM_MIN = 50
    SUM_MAX = 160

    def __init__(self, analysis: DltAnalysis):
        """
        Args:
            analysis: DltAnalysis 实例, 用于获取权重和约束
        """
        self.analysis = analysis
        self.front_weights = analysis.compute_front_weights()
        self.back_weights = analysis.compute_back_weights()
        self.stats = analysis.generate_summary_stats()

    def _weighted_sample(self, numbers: List[int], weights: Dict[int, float],
                         k: int, exclude: Optional[set] = None) -> List[int]:
        """
        加权随机抽样 (不放回)

        Args:
            numbers: 候选号码列表
            weights: {num: weight} 权重字典
            k: 抽取个数
            exclude: 需要排除的号码集合

        Returns:
            抽取的号码列表
        """
        candidates = [n for n in numbers if not exclude or n not in exclude]
        if len(candidates) < k:
            candidates = numbers[:]

        w = [weights.get(n, 1) for n in candidates]
        total = sum(w)
        if total <= 0:
            w = [1] * len(candidates)

        return random.choices(candidates, weights=w, k=k)

    def _check_front_constraints(self, nums: List[int]) -> bool:
        """
        检查前区号码是否满足分布约束

        Args:
            nums: 5个前区号码

        Returns:
            是否有效
        """
        if len(set(nums)) != 5:
            return False

        # 奇偶比
        odd = sum(1 for n in nums if n % 2 == 1)
        ratio = f"{odd}:{5-odd}"
        if ratio not in self.VALID_ODD_EVEN_RATIOS:
            return False

        # 大小比
        big = sum(1 for n in nums if n >= self.analysis.BIG_MIN)
        bs_ratio = f"{big}:{5-big}"
        if bs_ratio not in self.VALID_BIG_SMALL_RATIOS:
            return False

        # 和值范围
        total = sum(nums)
        if total < self.SUM_MIN or total > self.SUM_MAX:
            return False

        return True

    def _check_back_constraints(self, nums: List[int]) -> bool:
        """检查后区号码有效性"""
        return len(set(nums)) == 2 and all(1 <= n <= 12 for n in nums)

    def generate_one_group(self) -> Dict:
        """
        生成一组大乐透号码

        Returns:
            {front: [5], back: [2], confidence: int, stats: {...}}
        """
        front_numbers = list(self.analysis.FRONT_RANGE)
        back_numbers = list(self.analysis.BACK_RANGE)

        # 前区生成: 带约束的加权抽样
        front = None
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            candidates = self._weighted_sample(
                front_numbers, self.front_weights, k=5)
            if self._check_front_constraints(candidates):
                front = sorted(candidates)
                break
            attempts += 1

        # 如果约束太严没找到, 放宽约束再试
        if front is None:
            candidates = self._weighted_sample(
                front_numbers, self.front_weights, k=5)
            front = sorted(candidates)

        # 后区生成
        back = None
        attempts = 0
        while attempts < max_attempts:
            candidates = self._weighted_sample(
                back_numbers, self.back_weights, k=2)
            if self._check_back_constraints(candidates):
                back = sorted(candidates)
                break
            attempts += 1

        if back is None:
            back = sorted(self._weighted_sample(
                back_numbers, self.back_weights, k=2))

        # 信心评分: 基于各因子加权和
        confidence = self._score_confidence(front, back)

        # 号码特征
        odd = sum(1 for n in front if n % 2 == 1)
        big = sum(1 for n in front if n >= self.analysis.BIG_MIN)

        return {
            "front": front,
            "back": back,
            "confidence": confidence,
            "features": {
                "odd_even": f"{odd}:{5-odd}",
                "big_small": f"{big}:{5-big}",
                "sum": sum(front),
                "has_consecutive": any(front[i+1] - front[i] == 1 for i in range(4)),
            },
        }

    def generate_groups(self, count: int = 5) -> List[Dict]:
        """
        生成多组号码

        Args:
            count: 生成组数

        Returns:
            每组包含 front, back, confidence, features
        """
        groups = []
        for _ in range(count):
            group = self.generate_one_group()
            groups.append(group)

        # 按信心分排序
        groups.sort(key=lambda g: g["confidence"], reverse=True)
        return groups

    def _score_confidence(self, front: List[int], back: List[int]) -> int:
        """
        对生成的号码组合进行信心评分 (0-100)

        评分因子:
          - 权重匹配度: 选中的号码在权重中的排名
          - 特征匹配度: 奇偶比/大小比与历史常见模式匹配
          - 重复过滤: 避免与近期号码完全相同
        """
        score = 60  # 基础分

        # 1. 权重匹配 (+15)
        sorted_front = sorted(self.front_weights.items(), key=lambda x: -x[1])
        top10 = {n for n, _ in sorted_front[:10]}
        top20 = {n for n, _ in sorted_front[:20]}
        matched = sum(1 for n in front if n in top10)
        score += matched * 3  # 每匹配一个热号+3
        matched_top20 = sum(1 for n in front if n in top20)
        score += (matched_top20 - matched) * 1  # 额外top20每个+1

        # 2. 特征匹配 (+15)
        odd = sum(1 for n in front if n % 2 == 1)
        ratio = f"{odd}:{5-odd}"
        common_oe = self.stats["odd_even"]["most_common"]
        if ratio == common_oe:
            score += 8

        total = sum(front)
        avg_sum = self.stats["sum_analysis"]["avg"]
        if abs(total - avg_sum) <= 15:
            score += 7

        # 3. 多样性奖励: 与近期号码不同的程度 (+10)
        recent_sets = [set(nums) for nums in self.analysis.recent_data[:5]]
        front_set = set(front)
        if not any(front_set == rs for rs in recent_sets):
            score += 5
        if not any(len(front_set & rs) >= 4 for rs in recent_sets):
            score += 3

        return min(100, max(0, score))
