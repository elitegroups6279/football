"""
比赛数据爬取 - 爬取足球比赛基本信息和赛果数据

数据字段说明：
  - gid: 比赛唯一ID
  - match_date: 比赛日期
  - league_id: 联赛ID
  - league_name: 联赛名称
  - home_team: 主队名称
  - away_team: 客队名称
  - home_score: 主队进球数
  - away_score: 客队进球数
  - full_score: 完整比分
  - half_score: 半场比分
  - status: 比赛状态
  - match_time: 比赛开赛时间
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd

from config import MATCHES_DIR
from scraper.api_client import APIClient
from scraper.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class MatchScraper:
    """比赛数据爬取器"""

    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self.api_client = APIClient() if use_api else None
        self.web_scraper = WebScraper()

    def scrape_hot_leagues(self) -> List[Dict[str, Any]]:
        """
        爬取热门联赛列表

        Returns:
            热门联赛列表
            [{"id": 2079, "name": "英超", "logo": "...", "isFive": true}]
        """
        leagues = []

        # 优先使用API
        if self.use_api:
            api_leagues = self.api_client.get_hot_league_list()
            if api_leagues:
                logger.info(f"从API获取到 {len(api_leagues)} 个热门联赛")
                return api_leagues

        # 回退：从网页解析
        logger.info("尝试从网页解析热门联赛...")
        html = self.web_scraper.fetch_league_index()
        if html:
            leagues = self.web_scraper.parse_hot_leagues_from_html(html)
            logger.info(f"从网页解析到 {len(leagues)} 个热门联赛")

        return leagues

    def scrape_league_matches(
        self,
        league_id: int,
        league_name: str = "",
        max_pages: int = 5,
    ) -> pd.DataFrame:
        """
        爬取指定联赛的所有比赛

        Args:
            league_id: 联赛ID
            league_name: 联赛名称（可选）
            max_pages: 最大爬取页数

        Returns:
            包含比赛数据的DataFrame
        """
        all_matches = []

        for page in range(1, max_pages + 1):
            logger.info(f"爬取联赛 {league_id} ({league_name}) 第{page}页...")

            # 尝试API
            if self.use_api:
                matches = self.api_client.get_league_match_list(league_id)
                if matches:
                    for match in matches:
                        match["league_name"] = league_name or match.get("leagueName", "")
                        all_matches.append(match)
                    break  # API通常不分页
                else:
                    # API失败，尝试网页
                    logger.info(f"API获取失败，尝试网页爬取...")

            # 网页爬取
            html = self.web_scraper.fetch_league_matches(league_id, page)
            if not html:
                break

            # 从HTML中提取比赛数据
            page_matches = self._parse_matches_from_html(html, league_id, league_name)
            if not page_matches:
                break

            all_matches.extend(page_matches)
            time.sleep(0.5)

        if not all_matches:
            logger.warning(f"联赛 {league_id} 未获取到比赛数据")
            return pd.DataFrame()

        df = self._normalize_matches(all_matches)
        self._save_matches(df, league_id)
        return df

    def scrape_today_matches(self) -> pd.DataFrame:
        """
        爬取今日所有比赛

        Returns:
            今日比赛DataFrame
        """
        logger.info("爬取今日比赛...")

        # 通过API获取热门联赛的比赛
        all_matches = []
        hot_leagues = self.scrape_hot_leagues()

        for league in hot_leagues[:10]:  # 取前10个热门联赛
            league_id = league.get("id")
            league_name = league.get("name", "")
            if not league_id:
                continue

            df = self.scrape_league_matches(league_id, league_name, max_pages=1)
            if not df.empty:
                all_matches.append(df)
            time.sleep(0.3)

        if all_matches:
            result = pd.concat(all_matches, ignore_index=True)
            # 保存全部今日比赛
            self._save_matches(result, "today_matches")
            return result

        return pd.DataFrame()

    def scrape_matches_by_date_range(
        self,
        start_date: str,
        end_date: str,
        league_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        按日期范围爬取比赛

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            league_ids: 联赛ID列表，默认使用热门联赛

        Returns:
            比赛数据DataFrame
        """
        # 获取联赛列表
        if not league_ids:
            hot_leagues = self.scrape_hot_leagues()
            league_ids = [l.get("id") for l in hot_leagues if l.get("id")]

        all_matches = []
        for league_id in league_ids:
            logger.info(f"爬取联赛 {league_id} 的比赛...")
            df = self.scrape_league_matches(league_id, max_pages=3)
            if not df.empty:
                all_matches.append(df)
            time.sleep(0.5)

        if all_matches:
            result = pd.concat(all_matches, ignore_index=True)
            result = result.drop_duplicates(subset=["gid"], keep="first")

            # 按日期过滤
            if "match_date" in result.columns:
                result["match_date"] = pd.to_datetime(result["match_date"])
                mask = (result["match_date"] >= start_date) & (result["match_date"] <= end_date)
                result = result[mask]

            self._save_matches(result, "all_matches")
            logger.info(f"共获取到 {len(result)} 场比赛")
            return result

        return pd.DataFrame()

    def _parse_matches_from_html(
        self,
        html: str,
        league_id: int,
        league_name: str,
    ) -> List[Dict[str, Any]]:
        """
        从HTML中解析比赛数据

        Args:
            html: 页面HTML
            league_id: 联赛ID
            league_name: 联赛名称

        Returns:
            比赛数据列表
        """
        matches = []

        # 尝试提取JSON数据（页面可能内嵌了JSON）
        import re
        json_patterns = [
            r'var\s+matchData\s*=\s*(\[.*?\]);',
            r'var\s+matchList\s*=\s*(\[.*?\]);',
            r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
            r'matchList:\s*(\[.*?\]),',
        ]

        for pattern in json_patterns:
            found = re.search(pattern, html, re.DOTALL)
            if found:
                try:
                    data = json.loads(found.group(1))
                    if isinstance(data, list):
                        for item in data:
                            item["league_id"] = league_id
                            item["league_name"] = league_name
                            matches.append(item)
                        if matches:
                            return matches
                    elif isinstance(data, dict):
                        # 尝试从字典中提取列表
                        for key in ["matchList", "matches", "list", "data"]:
                            if key in data and isinstance(data[key], list):
                                for item in data[key]:
                                    item["league_id"] = league_id
                                    item["league_name"] = league_name
                                    matches.append(item)
                                return matches
                except (json.JSONDecodeError, KeyError):
                    continue

        # 如果没找到JSON，尝试从HTML表格中提取
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # 查找比赛行
        for row in soup.select("tr, .matchRow, .match-item, [class*=match]"):
            cells = row.find_all("td")
            if len(cells) >= 5:
                match = {
                    "league_id": league_id,
                    "league_name": league_name,
                }
                # 尝试提取文本
                texts = [c.get_text(strip=True) for c in cells]
                if any(score in texts for score in ["vs", "VS", "-"]):
                    match["raw_text"] = " | ".join(texts)
                    matches.append(match)

        return matches

    def _normalize_matches(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        标准化比赛数据

        Args:
            matches: 原始比赛数据列表

        Returns:
            标准化的DataFrame
        """
        df = pd.DataFrame(matches)

        # 重命名常用字段为统一名称
        field_mapping = {
            "gid": "gid",
            "id": "gid",
            "matchId": "gid",
            "match_id": "gid",
            "homeTeamName": "home_team",
            "home_name": "home_team",
            "homeTeam": "home_team",
            "awayTeamName": "away_team",
            "away_name": "away_team",
            "awayTeam": "away_team",
            "leagueName": "league_name",
            "league_name": "league_name",
            "leagueId": "league_id",
            "league_id": "league_id",
            "homeTeamScore": "home_score",
            "home_score": "home_score",
            "awayTeamScore": "away_score",
            "away_score": "away_score",
            "matchTime": "match_time",
            "match_time": "match_time",
            "matchDate": "match_date",
            "match_date": "match_date",
            "status": "status",
            "matchStatus": "status",
        }

        df.rename(columns={k: v for k, v in field_mapping.items() if k in df.columns}, inplace=True)

        # 确保gid列存在
        if "gid" not in df.columns:
            logger.warning("数据中没有gid字段，使用行索引替代")
            df["gid"] = range(len(df))

        # 解析比赛时间
        if "match_time" in df.columns:
            try:
                df["match_date"] = pd.to_datetime(df["match_time"]).dt.date
            except (ValueError, TypeError):
                pass

        # 转换比分
        for col in ["home_score", "away_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _save_matches(self, df: pd.DataFrame, identifier):
        """保存比赛数据到CSV"""
        if df.empty:
            return

        if isinstance(identifier, int):
            filename = MATCHES_DIR / f"league_{identifier}_matches.csv"
        else:
            filename = MATCHES_DIR / f"{identifier}.csv"

        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"比赛数据已保存到: {filename} (共{len(df)}条)")
        return

    def close(self):
        """清理资源"""
        if self.api_client:
            self.api_client.close()
        self.web_scraper.close()
