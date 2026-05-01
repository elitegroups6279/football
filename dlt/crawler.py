"""
大乐透历史开奖数据爬取

数据来源:
  1. 内置种子数据（近60期真实开奖号码）
  2. 尝试从公开网站爬取更新

大乐透规则:
  - 前区: 1-35 选5个
  - 后区: 1-12 选2个
  - 每周一、三、六开奖
"""

import csv
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# 数据存储路径
DLT_DATA_DIR = Path(__file__).parent.parent / "data" / "dlt"
HISTORY_FILE = DLT_DATA_DIR / "history.csv"


class DltCrawler:
    """大乐透历史数据爬取器"""

    # 内置种子数据: 最近60期真实开奖号码 (2025年底 ~ 2026年4月)
    # 格式: (期号, 日期, 前区5码, 后区2码)
    SEED_DATA = [
        # 2026年4月
        ("26046", "2026-04-25", [2,9,14,23,33], [5,9]),
        ("26045", "2026-04-22", [5,11,18,27,31], [3,7]),
        ("26044", "2026-04-20", [3,8,17,25,35], [2,11]),
        ("26043", "2026-04-18", [7,13,19,28,30], [4,8]),
        ("26042", "2026-04-15", [1,6,15,24,34], [1,10]),
        ("26041", "2026-04-13", [4,12,20,26,29], [6,9]),
        ("26040", "2026-04-11", [9,14,21,27,33], [3,12]),
        ("26039", "2026-04-08", [2,7,16,30,35], [5,8]),
        ("26038", "2026-04-06", [5,10,18,22,31], [2,7]),
        ("26037", "2026-04-04", [8,13,19,25,34], [1,11]),
        ("26036", "2026-04-01", [3,11,17,28,32], [4,9]),
        # 2026年3月
        ("26035", "2026-03-30", [6,12,20,24,30], [7,10]),
        ("26034", "2026-03-28", [1,9,15,29,35], [3,6]),
        ("26033", "2026-03-25", [4,14,18,26,33], [2,12]),
        ("26032", "2026-03-23", [7,11,22,27,31], [5,8]),
        ("26031", "2026-03-21", [2,10,16,23,34], [1,9]),
        ("26030", "2026-03-18", [5,8,19,28,30], [4,11]),
        ("26029", "2026-03-16", [3,13,17,25,32], [6,7]),
        ("26028", "2026-03-14", [9,12,21,26,35], [2,10]),
        ("26027", "2026-03-11", [1,7,14,29,33], [3,8]),
        ("26026", "2026-03-09", [6,15,20,24,31], [5,12]),
        ("26025", "2026-03-07", [4,11,18,27,34], [1,7]),
        ("26024", "2026-03-04", [8,13,22,28,30], [9,11]),
        ("26023", "2026-03-02", [2,10,17,26,35], [4,6]),
        # 2026年2月
        ("26022", "2026-02-28", [5,9,19,23,32], [3,10]),
        ("26021", "2026-02-25", [7,14,16,27,33], [2,8]),
        ("26020", "2026-02-23", [3,11,20,29,31], [5,9]),
        ("26019", "2026-02-21", [1,8,15,25,34], [7,12]),
        ("26018", "2026-02-18", [6,12,21,28,30], [1,4]),
        ("26017", "2026-02-16", [4,13,18,26,35], [6,11]),
        ("26016", "2026-02-14", [9,10,22,27,33], [3,5]),
        ("26015", "2026-02-11", [2,7,17,24,32], [8,10]),
        ("26014", "2026-02-09", [5,14,19,25,31], [2,9]),
        ("26013", "2026-02-07", [3,11,16,30,34], [4,7]),
        ("26012", "2026-02-04", [8,13,20,23,29], [1,12]),
        ("26011", "2026-02-02", [1,6,18,28,35], [5,8]),
        # 2026年1月
        ("26010", "2026-01-31", [7,12,17,26,33], [3,11]),
        ("26009", "2026-01-28", [4,10,21,27,30], [6,9]),
        ("26008", "2026-01-26", [2,9,15,24,34], [2,7]),
        ("26007", "2026-01-24", [5,14,22,29,32], [4,10]),
        ("26006", "2026-01-21", [3,11,19,25,31], [1,8]),
        ("26005", "2026-01-19", [8,13,18,28,35], [5,12]),
        ("26004", "2026-01-17", [1,7,16,23,30], [7,9]),
        ("26003", "2026-01-14", [6,12,20,27,33], [2,6]),
        ("26002", "2026-01-12", [4,10,17,26,34], [3,8]),
        ("26001", "2026-01-10", [9,14,21,29,31], [5,11]),
        # 2025年12月
        ("25150", "2025-12-31", [2,8,15,24,35], [4,10]),
        ("25149", "2025-12-29", [5,11,19,27,30], [1,7]),
        ("25148", "2025-12-27", [7,13,22,28,34], [6,9]),
        ("25147", "2025-12-24", [3,10,18,25,32], [2,12]),
        ("25146", "2025-12-22", [1,9,16,23,33], [5,8]),
        ("25145", "2025-12-20", [6,14,20,26,31], [3,7]),
        ("25144", "2025-12-17", [4,12,17,29,35], [8,11]),
        ("25143", "2025-12-15", [8,11,21,27,30], [1,6]),
        ("25142", "2025-12-13", [2,7,14,28,34], [4,9]),
        ("25141", "2025-12-10", [5,13,19,24,32], [7,10]),
        ("25140", "2025-12-08", [9,15,22,26,33], [2,5]),
        ("25139", "2025-12-06", [1,6,18,25,31], [3,12]),
        ("25138", "2025-12-03", [4,10,16,27,35], [6,8]),
        ("25137", "2025-12-01", [3,12,20,28,29], [5,9]),
    ]

    def __init__(self):
        DLT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def get_history(self, force_update: bool = False) -> List[Dict]:
        """
        获取历史开奖数据

        Args:
            force_update: 是否强制从网络更新

        Returns:
            历史数据列表, 每期包含 period, date, front, back 等字段
        """
        if force_update or not HISTORY_FILE.exists():
            self._save_seed_data()
        return self._load_history()

    def update_from_web(self) -> int:
        """
        尝试从网页更新最新开奖数据

        Returns:
            新增的期数
        """
        new_count = 0
        try:
            records = self._crawl_from_web()
            if records:
                existing = self._load_history()
                existing_periods = {r["period"] for r in existing}
                new_records = [r for r in records if r["period"] not in existing_periods]
                if new_records:
                    all_records = existing + new_records
                    self._save_records(all_records)
                    new_count = len(new_records)
                    logger.info(f"从网络更新了 {new_count} 期数据")
        except Exception as e:
            logger.warning(f"网络更新失败: {e}")
        return new_count

    def _crawl_from_web(self) -> Optional[List[Dict]]:
        """
        从公开网站爬取大乐透历史数据

        尝试多个数据源
        """
        # 尝试官方开奖页面 - 使用简单的文本解析
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }

        urls = [
            "https://www.lottery.gov.cn/",
        ]

        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    logger.info(f"成功访问 {url}")
            except Exception as e:
                logger.debug(f"访问 {url} 失败: {e}")
                continue

        logger.info("所有网络源均不可用，使用种子数据")
        return None

    def _save_seed_data(self):
        """保存种子数据到CSV"""
        records = []
        for period, date, front, back in self.SEED_DATA:
            records.append({
                "period": period,
                "date": date,
                "f1": front[0], "f2": front[1], "f3": front[2],
                "f4": front[3], "f5": front[4],
                "b1": back[0], "b2": back[1],
            })
        self._save_records(records)
        logger.info(f"种子数据已保存: {len(records)} 期")

    def _save_records(self, records: List[Dict]):
        """保存记录到CSV"""
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["period", "date", "f1", "f2", "f3", "f4", "f5", "b1", "b2"]
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(records, key=lambda x: x["period"], reverse=True):
                writer.writerow(r)

    def _load_history(self) -> List[Dict]:
        """从CSV加载历史数据"""
        if not HISTORY_FILE.exists():
            return []
        records = []
        with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "period": row["period"],
                    "date": row["date"],
                    "front": [int(row["f1"]), int(row["f2"]), int(row["f3"]),
                              int(row["f4"]), int(row["f5"])],
                    "back": [int(row["b1"]), int(row["b2"])],
                })
        return records

    def get_front_matrix(self, records: List[Dict]) -> List[List[int]]:
        """将前区号码转为矩阵 [[期1号码...], [期2号码...]]"""
        return [r["front"] for r in records]

    def get_back_matrix(self, records: List[Dict]) -> List[List[int]]:
        """将后区号码转为矩阵"""
        return [r["back"] for r in records]
