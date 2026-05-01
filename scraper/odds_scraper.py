"""
赔率数据爬取 - 收集每场比赛的完整赔率数据

数据组织方式：
  - 每场比赛（按gid）一个独立的JSON文件
  - 包含多家博彩公司的欧赔、亚盘、大小球数据
  - 包含初盘和即时赔率

赔率数据结构：
  {
    "gid": 12345,
    "match_info": {...},
    "europe_odds": [
      {
        "company_id": 1,
        "company_name": "澳门",
        "home_win": 1.80,
        "draw": 3.40,
        "away_win": 3.80,
        "initial_home": 1.85,
        "initial_draw": 3.30,
        "initial_away": 3.60,
        "update_time": "2024-01-01 12:00:00"
      }
    ],
    "asia_odds": [...],
    "bigsmall_odds": [...]
  }
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd

from config import ODDS_DIR, MATCHES_DIR, BOOKMAKER_NAMES
from scraper.api_client import APIClient

logger = logging.getLogger(__name__)


class OddsScraper:
    """赔率数据爬取器"""

    def __init__(self):
        self.api_client = APIClient()

    def scrape_match_odds(self, gid: int) -> Optional[Dict[str, Any]]:
        """
        爬取单场比赛的完整赔率数据

        Args:
            gid: 比赛ID

        Returns:
            完整的赔率数据字典
        """
        odds_data = {
            "gid": gid,
            "scrape_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "europe_odds": [],
            "asia_odds": [],
            "bigsmall_odds": [],
            "match_info": None,
        }

        # 1. 获取欧赔
        europe = self.api_client.get_europe_odds(gid)
        if europe:
            odds_data["europe_odds"] = self._normalize_europe_odds(europe)
            logger.info(f"比赛 {gid}: 获取到 {len(europe)} 家公司的欧赔数据")
        else:
            logger.warning(f"比赛 {gid}: 未获取到欧赔数据")

        time.sleep(0.3)

        # 2. 获取亚盘
        asia = self.api_client.get_asia_odds(gid)
        if asia:
            odds_data["asia_odds"] = self._normalize_asia_odds(asia)
            logger.info(f"比赛 {gid}: 获取到 {len(asia)} 家公司的亚盘数据")
        else:
            logger.warning(f"比赛 {gid}: 未获取到亚盘数据")

        time.sleep(0.3)

        # 3. 获取大小球
        bigsmall = self.api_client.get_bigsmall_odds(gid)
        if bigsmall:
            odds_data["bigsmall_odds"] = self._normalize_bigsmall_odds(bigsmall)
            logger.info(f"比赛 {gid}: 获取到 {len(bigsmall)} 家公司的大小球数据")
        else:
            logger.warning(f"比赛 {gid}: 未获取到大小球数据")

        return odds_data

    def _normalize_europe_odds(self, odds_list: List[Dict]) -> List[Dict]:
        """
        标准化欧赔数据

        Args:
            odds_list: 原始欧赔数据列表

        Returns:
            标准化后的欧赔数据
        """
        normalized = []
        for odd in odds_list:
            company_id = odd.get("companyId", odd.get("company_id", odd.get("id")))
            entry = {
                "company_id": company_id,
                "company_name": BOOKMAKER_NAMES.get(company_id, f"公司{company_id}"),
                # 即时赔率
                "home_win": self._safe_float(odd.get("homeWin", odd.get("home_win"))),
                "draw": self._safe_float(odd.get("draw", odd.get("draw_odd"))),
                "away_win": self._safe_float(odd.get("awayWin", odd.get("away_win"))),
                # 初始赔率
                "initial_home_win": self._safe_float(odd.get("initialHomeWin", odd.get("initHomeWin"))),
                "initial_draw": self._safe_float(odd.get("initialDraw", odd.get("initDraw"))),
                "initial_away_win": self._safe_float(odd.get("initialAwayWin", odd.get("initAwayWin"))),
                # 变化时间
                "update_time": odd.get("updateTime", odd.get("update_time", "")),
            }
            normalized.append(entry)

        return normalized

    def _normalize_asia_odds(self, odds_list: List[Dict]) -> List[Dict]:
        """
        标准化亚盘数据

        Args:
            odds_list: 原始亚盘数据列表

        Returns:
            标准化后的亚盘数据
        """
        normalized = []
        for odd in odds_list:
            company_id = odd.get("companyId", odd.get("company_id", odd.get("id")))
            entry = {
                "company_id": company_id,
                "company_name": BOOKMAKER_NAMES.get(company_id, f"公司{company_id}"),
                # 即时盘口
                "handicap": self._safe_float(odd.get("handicap", odd.get("letBall"))),
                "home_odds": self._safe_float(odd.get("homeOdds", odd.get("home_odds", odd.get("homeRate")))),
                "away_odds": self._safe_float(odd.get("awayOdds", odd.get("away_odds", odd.get("awayRate")))),
                # 初始盘口
                "initial_handicap": self._safe_float(odd.get("initialHandicap", odd.get("initHandicap"))),
                "initial_home_odds": self._safe_float(odd.get("initialHomeOdds", odd.get("initHomeRate"))),
                "initial_away_odds": self._safe_float(odd.get("initialAwayOdds", odd.get("initAwayRate"))),
                "update_time": odd.get("updateTime", odd.get("update_time", "")),
            }
            normalized.append(entry)

        return normalized

    def _normalize_bigsmall_odds(self, odds_list: List[Dict]) -> List[Dict]:
        """
        标准化大小球数据

        Args:
            odds_list: 原始大小球数据列表

        Returns:
            标准化后的大小球数据
        """
        normalized = []
        for odd in odds_list:
            company_id = odd.get("companyId", odd.get("company_id", odd.get("id")))
            entry = {
                "company_id": company_id,
                "company_name": BOOKMAKER_NAMES.get(company_id, f"公司{company_id}"),
                # 即时盘口
                "total_goals": self._safe_float(odd.get("totalGoals", odd.get("total"))),
                "over_odds": self._safe_float(odd.get("overOdds", odd.get("over_odds", odd.get("bigRate")))),
                "under_odds": self._safe_float(odd.get("underOdds", odd.get("under_odds", odd.get("smallRate")))),
                # 初始盘口
                "initial_total_goals": self._safe_float(odd.get("initialTotal", odd.get("initTotal"))),
                "initial_over_odds": self._safe_float(odd.get("initialOverOdds", odd.get("initBigRate"))),
                "initial_under_odds": self._safe_float(odd.get("initialUnderOdds", odd.get("initSmallRate"))),
                "update_time": odd.get("updateTime", odd.get("update_time", "")),
            }
            normalized.append(entry)

        return normalized

    def scrape_multiple_odds(
        self,
        gids: List[int],
        max_workers: int = 3,
    ) -> Dict[int, Dict[str, Any]]:
        """
        批量爬取多场比赛的赔率

        Args:
            gids: 比赛ID列表
            max_workers: 并发数

        Returns:
            {gid: odds_data} 字典
        """
        results = {}
        total = len(gids)

        for i, gid in enumerate(gids, 1):
            logger.info(f"爬取赔率 [{i}/{total}]: 比赛 {gid}")
            odds = self.scrape_match_odds(gid)
            if odds:
                results[gid] = odds
                self._save_odds_to_file(odds)
            time.sleep(0.5)

        logger.info(f"赔率爬取完成: 成功 {len(results)}/{total}")
        return results

    def scrape_odds_from_match_csv(
        self,
        csv_path: Optional[str] = None,
        max_matches: int = 50,
    ) -> Dict[int, Dict[str, Any]]:
        """
        从保存的比赛数据中读取gid并爬取赔率

        Args:
            csv_path: CSV文件路径，默认使用最新保存的比赛数据
            max_matches: 最大爬取比赛数

        Returns:
            {gid: odds_data} 字典
        """
        # 查找最新的比赛数据文件
        if csv_path:
            match_files = [Path(csv_path)]
        else:
            match_files = sorted(MATCHES_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not match_files:
            logger.error("未找到比赛数据文件")
            return {}

        df = pd.read_csv(match_files[0])
        gids = df["gid"].dropna().unique()[:max_matches].tolist()

        logger.info(f"从 {match_files[0].name} 读取到 {len(gids)} 个比赛ID")
        return self.scrape_multiple_odds(gids)

    def _save_odds_to_file(self, odds_data: Dict[str, Any]):
        """保存赔率数据到JSON文件"""
        if not odds_data or "gid" not in odds_data:
            return

        gid = odds_data["gid"]
        filename = ODDS_DIR / f"{gid}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(odds_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"赔率数据已保存: {filename}")

    def load_odds_from_file(self, gid: int) -> Optional[Dict[str, Any]]:
        """从文件加载赔率数据"""
        filename = ODDS_DIR / f"{gid}.json"
        if filename.exists():
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def build_odds_dataframe(self, gid: int) -> Optional[pd.DataFrame]:
        """
        将单场比赛的赔率数据转为DataFrame（用于分析）

        Args:
            gid: 比赛ID

        Returns:
            包含各公司赔率的DataFrame
        """
        odds_data = self.load_odds_from_file(gid)
        if not odds_data:
            return None

        rows = []

        # 欧赔
        for odd in odds_data.get("europe_odds", []):
            row = {
                "gid": gid,
                "company_id": odd["company_id"],
                "company_name": odd["company_name"],
                "odds_type": "europe",
                "home_win": odd["home_win"],
                "draw": odd["draw"],
                "away_win": odd["away_win"],
                "initial_home_win": odd["initial_home_win"],
                "initial_draw": odd["initial_draw"],
                "initial_away_win": odd["initial_away_win"],
            }
            rows.append(row)

        # 亚盘
        for odd in odds_data.get("asia_odds", []):
            row = {
                "gid": gid,
                "company_id": odd["company_id"],
                "company_name": odd["company_name"],
                "odds_type": "asia",
                "handicap": odd["handicap"],
                "home_odds": odd["home_odds"],
                "away_odds": odd["away_odds"],
            }
            rows.append(row)

        # 大小球
        for odd in odds_data.get("bigsmall_odds", []):
            row = {
                "gid": gid,
                "company_id": odd["company_id"],
                "company_name": odd["company_name"],
                "odds_type": "bigsmall",
                "total_goals": odd["total_goals"],
                "over_odds": odd["over_odds"],
                "under_odds": odd["under_odds"],
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def consolidate_odds_summary(self, gid: int) -> Optional[Dict[str, Any]]:
        """
        汇总单场比赛的赔率统计摘要

        Args:
            gid: 比赛ID

        Returns:
            赔率统计摘要
        """
        odds_data = self.load_odds_from_file(gid)
        if not odds_data:
            return None

        summary = {"gid": gid}

        # 欧赔汇总
        euro = odds_data.get("europe_odds", [])
        if euro:
            home_wins = [o["home_win"] for o in euro if o["home_win"]]
            draws = [o["draw"] for o in euro if o["draw"]]
            away_wins = [o["away_win"] for o in euro if o["away_win"]]

            summary["europe"] = {
                "company_count": len(euro),
                "avg_home_win": sum(home_wins) / len(home_wins) if home_wins else None,
                "avg_draw": sum(draws) / len(draws) if draws else None,
                "avg_away_win": sum(away_wins) / len(away_wins) if away_wins else None,
                "min_home_win": min(home_wins) if home_wins else None,
                "max_home_win": max(home_wins) if home_wins else None,
                "std_home_win": self._std(home_wins),
            }

        # 亚盘汇总
        asia = odds_data.get("asia_odds", [])
        if asia:
            handicaps = [o["handicap"] for o in asia if o["handicap"]]
            summary["asia"] = {
                "company_count": len(asia),
                "avg_handicap": sum(handicaps) / len(handicaps) if handicaps else None,
            }

        return summary

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """安全转换为float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _std(values: List[float]) -> Optional[float]:
        """计算标准差"""
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def close(self):
        """清理资源"""
        self.api_client.close()
