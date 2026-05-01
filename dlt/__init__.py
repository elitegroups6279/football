"""大乐透科学预测系统

大乐透规则:
  - 前区: 从1-35中选择5个号码
  - 后区: 从1-12中选择2个号码
  - 开奖: 每周一、三、六

模块:
  - crawler: 历史开奖数据爬取
  - analysis: 多维度统计分析
  - predictor: 号码生成引擎
  - report: 结果输出
"""

from .crawler import DltCrawler
from .analysis import DltAnalysis
from .predictor import DltPredictor
from .report import DltReport

__all__ = ["DltCrawler", "DltAnalysis", "DltPredictor", "DltReport"]
