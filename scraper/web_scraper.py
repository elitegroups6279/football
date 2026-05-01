"""
网页爬虫 - 直接从天天盈球网页中提取比赛数据

当API接口不可用时，通过解析HTML页面来获取数据。
支持爬取联赛列表、比赛详情等信息。
"""

import re
import json
import time
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import WEB_BASE_URL, HEADERS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class WebScraper:
    """天天盈球网页爬虫"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_league_index(self) -> Optional[str]:
        """
        获取联赛索引页面HTML

        Returns:
            HTML文本，失败返回None
        """
        url = f"{WEB_BASE_URL}/live/leagueIndex"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            # 检查页面编码
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.error(f"获取联赛索引页失败: {e}")
            return None

    def parse_hot_leagues_from_html(self, html: str) -> List[Dict[str, Any]]:
        """
        从HTML中解析热门联赛列表

        Args:
            html: 页面HTML

        Returns:
            热门联赛列表
        """
        leagues = []
        soup = BeautifulSoup(html, "html.parser")

        # 查找热门赛事区域 - 使用Vue绑定判断
        hot_event_items = soup.select(".hotEventList a, .hotEventList li")
        for item in hot_event_items:
            text = item.get_text(strip=True)
            if text and len(text) < 20:
                leagues.append({"name": text})

        # 去重
        seen = set()
        unique_leagues = []
        for league in leagues:
            if league["name"] not in seen:
                seen.add(league["name"])
                unique_leagues.append(league)

        return unique_leagues

    def parse_today_matches_from_html(self, html: str) -> List[Dict[str, Any]]:
        """
        从HTML中解析今日赛事

        Args:
            html: 页面HTML

        Returns:
            今日赛事列表
        """
        matches = []
        soup = BeautifulSoup(html, "html.parser")

        # 查找今日赛事区域
        today_section = soup.select(".todayEventList li, .todayEvent a")
        for item in today_section:
            text = item.get_text(strip=True)
            if text and len(text) < 30:
                matches.append({"text": text})

        return matches

    def fetch_league_matches(self, league_id: int, page: int = 1) -> Optional[str]:
        """
        获取联赛比赛列表页面

        Args:
            league_id: 联赛ID
            page: 页码

        Returns:
            HTML文本
        """
        # 联赛详情页URL模式
        url = f"{WEB_BASE_URL}/live/zq/matchDetail/data?leagueId={league_id}"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.error(f"获取联赛比赛列表失败(league_id={league_id}): {e}")
            return None

    def fetch_match_detail(self, match_id: int) -> Optional[str]:
        """
        获取比赛详情页面

        Args:
            match_id: 比赛ID

        Returns:
            HTML文本
        """
        url = f"{WEB_BASE_URL}/live/zq/matchDetail/info?gid={match_id}"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.error(f"获取比赛详情失败(match_id={match_id}): {e}")
            return None

    def fetch_odds_page(self, match_id: int, odds_type: str = "oz") -> Optional[str]:
        """
        获取赔率页面

        Args:
            match_id: 比赛ID
            odds_type: 赔率类型（oz=欧赔, yp=亚盘, dxq=大小球）

        Returns:
            HTML文本
        """
        url = f"{WEB_BASE_URL}/live/zq/matchDetail/{odds_type}?gid={match_id}"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.error(f"获取赔率页面失败(gid={match_id}, type={odds_type}): {e}")
            return None

    def search(self, keyword: str) -> Optional[List[Dict[str, Any]]]:
        """
        搜索联赛、球队、球员

        Args:
            keyword: 搜索关键词

        Returns:
            搜索结果列表
        """
        url = f"{WEB_BASE_URL}/search"
        try:
            resp = self.session.get(
                url,
                params={"keyword": keyword},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"

            # 尝试从HTML中提取JSON数据
            html = resp.text
            results = []
            soup = BeautifulSoup(html, "html.parser")

            # 提取搜索结果
            items = soup.select(".searchResultItem, .matchItem, .teamItem")
            for item in items:
                results.append({
                    "text": item.get_text(strip=True),
                    "html": str(item)[:200],
                })

            return results if results else [{"raw_length": len(html)}]

        except requests.RequestException as e:
            logger.error(f"搜索失败(keyword={keyword}): {e}")
            return None

    def close(self):
        """关闭会话"""
        self.session.close()
