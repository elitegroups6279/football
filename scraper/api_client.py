"""
API客户端 - 封装对天天盈球数据接口的调用

API结构说明：
  - 基础URL: https://sport.ttyingqiu.com/sportdata/f/{apiName}
  - 部分接口需要参数（如leagueId等）
  - 返回格式: {"code": "1", "msg": "成功", ...data}
  - code="1" 表示成功, code="0" 表示服务异常, code="3" 表示接口不存在

已知可用接口:
  - getHotLeagueList: 获取热门联赛列表（无需参数）
  - 其他接口需要根据具体参数调用
"""

import time
import json
import logging
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import API_BASE_URL, HEADERS, REQUEST_TIMEOUT, MAX_RETRIES, REQUEST_INTERVAL

logger = logging.getLogger(__name__)


class APIClient:
    """天天盈球API客户端"""

    def __init__(self):
        self.session = self._create_session()
        self.last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        """创建带重试机制的请求会话"""
        session = requests.Session()
        session.headers.update(HEADERS)

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _rate_limit(self):
        """请求频率限制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def call_api(
        self,
        api_name: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Optional[Dict[str, Any]]:
        """
        调用API接口

        Args:
            api_name: API名称（如 'getHotLeagueList'）
            params: 查询参数
            method: 请求方法（GET/POST）

        Returns:
            JSON响应字典，失败返回None
        """
        self._rate_limit()
        url = f"{API_BASE_URL}/{api_name}"

        try:
            if method.upper() == "GET":
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
            else:
                resp = self.session.post(
                    url,
                    data=params,
                    timeout=REQUEST_TIMEOUT,
                )

            resp.raise_for_status()
            data = resp.json()

            code = data.get("code")
            if code == "1":
                return data
            elif code == "3":
                logger.warning(f"API '{api_name}' 不存在 (code=3)")
                return None
            elif code == "0":
                logger.warning(f"API '{api_name}' 服务异常 (code=0): {data.get('msg')}")
                return None
            else:
                logger.warning(f"API '{api_name}' 返回未知状态码: {code}")
                return data

        except requests.exceptions.Timeout:
            logger.error(f"API '{api_name}' 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API '{api_name}' 请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"API '{api_name}' 响应不是有效JSON: {e}")
            return None

    def get_hot_league_list(self) -> Optional[list]:
        """获取热门联赛列表（无需参数）"""
        data = self.call_api("getHotLeagueList", method="GET")
        if data:
            return data.get("leagueList", [])
        return None

    def get_league_list(self, area_code: int = 1, page_size: int = 1000) -> Optional[list]:
        """
        获取联赛列表

        Args:
            area_code: 区域代码（1=欧洲, 2=美洲, 3=亚洲, 4=大洋洲, 5=非洲, 0=国际）
            page_size: 每页数量
        """
        data = self.call_api(
            "getLeagueList",
            params={"areaCode": area_code, "pageSize": page_size},
            method="GET",
        )
        if data:
            return data.get("leagueList", [])
        return None

    def get_country_list(self, qiu_flag: int = 1) -> Optional[list]:
        """
        获取国家列表

        Args:
            qiu_flag: 球类标志（1=足球, 2=篮球）
        """
        data = self.call_api(
            "getCountryList",
            params={"qiuFalg": qiu_flag},
            method="GET",
        )
        if data:
            return data.get("countryList", [])
        return None

    def get_league_match_list(
        self,
        league_id: int,
        season_id: Optional[int] = None,
        page_size: int = 500,
    ) -> Optional[list]:
        """
        获取联赛比赛列表

        Args:
            league_id: 联赛ID
            season_id: 赛季ID（可选）
            page_size: 每页数量
        """
        params = {"leagueId": league_id, "pageSize": page_size}
        if season_id:
            params["seasonId"] = season_id

        data = self.call_api("getLeagueMatchList", params=params, method="GET")
        if data:
            return data.get("matchList", data.get("list", []))
        return None

    def get_match_info(self, match_id: int) -> Optional[Dict]:
        """获取单场比赛信息"""
        data = self.call_api(
            "getMatchListById",
            params={"matchId": match_id},
            method="GET",
        )
        if data:
            return data.get("matchInfo", data)
        return None

    def get_europe_odds(self, match_id: int) -> Optional[list]:
        """
        获取欧赔数据

        Args:
            match_id: 比赛ID
        """
        data = self.call_api(
            "getListMatchEuropeOdds",
            params={"matchId": match_id},
            method="GET",
        )
        if data:
            return data.get("oddsList", data.get("list", []))
        return None

    def get_asia_odds(self, match_id: int) -> Optional[list]:
        """
        获取亚洲盘口数据

        Args:
            match_id: 比赛ID
        """
        data = self.call_api(
            "getListMatchAsiaPrimaryOdds",
            params={"matchId": match_id},
            method="GET",
        )
        if data:
            return data.get("oddsList", data.get("list", []))
        return None

    def get_bigsmall_odds(self, match_id: int) -> Optional[list]:
        """
        获取大小球数据

        Args:
            match_id: 比赛ID
        """
        data = self.call_api(
            "getListMatchBigSmallOdds",
            params={"matchId": match_id},
            method="GET",
        )
        if data:
            return data.get("oddsList", data.get("list", []))
        return None

    def close(self):
        """关闭会话"""
        self.session.close()
