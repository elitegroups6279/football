"""
竞彩API封装 - 使用POST+JSON body模式调用天天盈球接口

API发现总结:
  - 基础URL: https://sport.ttyingqiu.com/sportdata/f
  - 正确模式: POST请求, JSON body包含 apiName 字段
  - getMatchListByDate: game=407(竞彩), game=408(北京单场)
  - getMatchAllOdds: 单场比赛的完整赔率(欧赔+亚盘+大小球), 返回扁平字段
  - getMatchListById: 比赛详情(含最终比分)

getMatchListByDate 响应格式:
  - matchList[].matchId -> gid (比赛ID)
  - matchList[].homeName / awayName -> 主客队名
  - matchList[].homeId / awayId -> 主客队ID
  - matchList[].oddsEurope -> "2.34;3.45;2.99" 分号分隔的欧赔
  - matchList[].score -> ['0:0','2:1','','']  [半场,全场,加时,点球]
  - matchList[].status -> 0=未开始, 2=已结束
  - matchList[].isPlayed -> 0/1

getMatchAllOdds 响应格式(扁平):
  - winOdds/drawOdds/loseOdds -> 即时欧赔
  - firstWinOdds/firstDrawOdds/firstLoseOdds -> 初盘欧赔
  - letHandicap/letWinOdds/letLoseOdds -> 亚盘
  - bsHandicap/bigOdds/smallOdds -> 大小球
  - homeUpdown/awayUpdown/drawUpdown -> 赔率变化方向
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    API_BASE_URL, HEADERS, REQUEST_TIMEOUT, MAX_RETRIES, REQUEST_INTERVAL,
    GAME_JINGCAI, BOOKMAKER_NAMES,
)

logger = logging.getLogger(__name__)


class JingcaiAPI:
    """竞彩数据API客户端 - 使用POST+JSON body模式"""

    def __init__(self):
        self.session = self._create_session()
        self.last_request_time = 0.0
        self.BASE_URL = API_BASE_URL

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(HEADERS)
        session.headers.update({"Content-Type": "application/json;charset=UTF-8"})

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
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _post(self, api_name: str, body: Optional[Dict] = None) -> Optional[Dict]:
        """
        发送POST请求, apiName 放在JSON body中

        Args:
            api_name: API名称 (如 getMatchListByDate)
            body: 额外请求体参数

        Returns:
            解析后的JSON响应, 失败返回None
        """
        self._rate_limit()
        payload = {"apiName": api_name}
        if body:
            payload.update(body)

        try:
            resp = self.session.post(
                self.BASE_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            code = data.get("code")
            if code == "1":
                return data
            elif code == "0":
                logger.warning(f"API '{api_name}' 服务异常: {data.get('msg', '')}")
                return None
            elif code == "3":
                logger.warning(f"API '{api_name}' 不存在")
                return None
            else:
                logger.warning(f"API '{api_name}' 未知状态码: {code}")
                return data

        except requests.exceptions.Timeout:
            logger.error(f"API '{api_name}' 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API '{api_name}' 请求失败: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f"API '{api_name}' 响应不是有效JSON")
            return None

    def get_jingcai_matches(self, date_str: Optional[str] = None) -> List[Dict]:
        """
        获取竞彩足球比赛列表

        Args:
            date_str: 日期字符串 YYYY-MM-DD, 默认今天

        Returns:
            比赛列表, 每个比赛已标准化为snake_case字段
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"获取竞彩比赛数据: date={date_str}")
        data = self._post("getMatchListByDate", {
            "date": date_str,
            "game": GAME_JINGCAI,
        })

        if not data:
            logger.warning("未获取到竞彩比赛数据")
            return []

        raw_list = data.get("matchList", data.get("list", []))
        matches = [self._normalize_match(m) for m in raw_list]
        logger.info(f"获取到 {len(matches)} 场竞彩比赛")
        return matches

    def get_match_odds(self, gid: int) -> Optional[Dict]:
        """
        获取单场比赛的完整赔率数据

        getMatchAllOdds 返回扁平结构(单公司), 这里转换为与系统兼容的格式

        Args:
            gid: 比赛ID

        Returns:
            {europe_odds, asia_odds, bigsmall_odds} 标准化后的数据
        """
        logger.info(f"获取比赛赔率: gid={gid}")
        data = self._post("getMatchAllOdds", {"matchId": gid})

        if not data:
            return None

        result = {"gid": gid}

        # 欧赔 (单公司, 放入数组以兼容系统格式)
        euro_entry = {
            "company_id": 0,
            "company_name": "默认",
            "home_win": self._safe_float(data.get("winOdds")),
            "draw": self._safe_float(data.get("drawOdds")),
            "away_win": self._safe_float(data.get("loseOdds")),
            "initial_home_win": self._safe_float(data.get("firstWinOdds")),
            "initial_draw": self._safe_float(data.get("firstDrawOdds")),
            "initial_away_win": self._safe_float(data.get("firstLoseOdds")),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if any(v is not None for v in [euro_entry["home_win"], euro_entry["draw"],
                                        euro_entry["away_win"]]):
            result["europe_odds"] = [euro_entry]

        # 亚盘
        handicap_raw = data.get("letHandicap")
        handicap_val = self._parse_handicap(handicap_raw) if handicap_raw else None
        init_handicap_raw = data.get("firstLetHandicap")
        init_handicap_val = self._parse_handicap(init_handicap_raw) if init_handicap_raw else None

        asia_entry = {
            "company_id": 0,
            "company_name": "默认",
            "handicap": handicap_val,
            "home_odds": self._safe_float(data.get("letWinOdds")),
            "away_odds": self._safe_float(data.get("letLoseOdds")),
            "initial_handicap": init_handicap_val,
            "initial_home_odds": self._safe_float(data.get("firstLetWinOdds")),
            "initial_away_odds": self._safe_float(data.get("firstLetLoseOdds")),
        }
        if any(v is not None for v in [asia_entry["handicap"], asia_entry["home_odds"]]):
            result["asia_odds"] = [asia_entry]

        # 大小球
        bs_entry = {
            "company_id": 0,
            "company_name": "默认",
            "total_goals": self._safe_float(data.get("bsHandicap")),
            "over_odds": self._safe_float(data.get("bigOdds")),
            "under_odds": self._safe_float(data.get("smallOdds")),
            "initial_total_goals": self._safe_float(data.get("firstBsHandicap")),
            "initial_over_odds": self._safe_float(data.get("firstBigOdds")),
            "initial_under_odds": self._safe_float(data.get("firstSmallOdds")),
        }
        if any(v is not None for v in [bs_entry["total_goals"], bs_entry["over_odds"]]):
            result["bigsmall_odds"] = [bs_entry]

        logger.info(f"比赛 {gid}: 欧赔={1 if 'europe_odds' in result else 0}家, "
                    f"亚盘={1 if 'asia_odds' in result else 0}家, "
                    f"大小球={1 if 'bigsmall_odds' in result else 0}家")
        return result

    def get_match_result(self, match_id: int) -> Optional[Dict]:
        """
        获取比赛结果(含最终比分)

        Args:
            match_id: 比赛ID

        Returns:
            比赛详情(含比分)或None
        """
        data = self._post("getMatchListById", {"matchId": match_id})
        if not data:
            return None

        info = data.get("matchInfo", data.get("matchList", [{}]))
        if isinstance(info, list):
            info = info[0] if info else {}

        return self._normalize_match(info)

    def _normalize_match(self, m: Dict) -> Dict:
        """
        标准化比赛字段: 从 camelCase API字段 转 snake_case

        实际API字段名 (getMatchListByDate):
          matchId, homeName, awayName, homeId, awayId,
          oddsEurope("2.34;3.45;2.99"), oddsAsia, bigsmall,
          score(['0:0','2:1','','']), status, isPlayed
        """
        gid = m.get("matchId") or m.get("id") or m.get("gid")

        # 解析分号分隔的赔率 "2.34;3.45;2.99"
        odds_europe = self._parse_semicolon_odds(m.get("oddsEurope", ""))
        home_win_odds, draw_odds_parsed, away_win_odds = odds_europe if odds_europe else (None, None, None)

        # 解析比分 "0:0" -> (0, 0)
        home_score, away_score = self._parse_score_from_array(
            m.get("score", []))

        return {
            "gid": self._safe_int(gid),
            "match_no_cn": m.get("matchNoCn", ""),  # 竞彩编号 "周日001"
            "league_id": self._safe_int(m.get("leagueId")),
            "league_name": m.get("leagueName", ""),
            "home_team": m.get("homeName", ""),
            "away_team": m.get("awayName", ""),
            "home_team_id": self._safe_int(m.get("homeId")),
            "away_team_id": self._safe_int(m.get("awayId")),
            "match_date": m.get("matchDate", ""),
            "match_time": m.get("matchTime", ""),
            "home_score": home_score,
            "away_score": away_score,
            "status": m.get("status", 0),       # 0=未开始, 2=已结束
            "is_played": m.get("isPlayed", 0),   # 0=未赛, 1=已赛
            "home_rank": m.get("homeRank"),
            "away_rank": m.get("awayRank"),
            "home_win_odds": self._safe_float(home_win_odds),
            "draw_odds": self._safe_float(draw_odds_parsed),
            "away_win_odds": self._safe_float(away_win_odds),
            "round": m.get("round", ""),
            "stage_name": m.get("stageName", ""),
        }

    @staticmethod
    def _parse_semicolon_odds(odds_str: str) -> Optional[tuple]:
        """
        解析分号分隔的赔率字符串 "2.34;3.45;2.99"

        Returns:
            (home, draw, away) 或 None
        """
        if not odds_str or not isinstance(odds_str, str):
            return None
        parts = odds_str.split(";")
        if len(parts) < 3:
            return None
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_score_from_array(score_arr: list) -> tuple:
        """
        解析比分数组 ['0:0','2:1','',''] -> (2, 1)
        取第二个元素(全场比分)为主
        """
        if not score_arr or not isinstance(score_arr, list):
            return (None, None)
        # 第二个元素(索引1)是全场比赛比分
        fulltime = score_arr[1] if len(score_arr) > 1 else score_arr[0]
        if not fulltime or not isinstance(fulltime, str):
            return (None, None)
        try:
            parts = fulltime.split(":")
            if len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
        except (ValueError, TypeError):
            pass
        return (None, None)

    @staticmethod
    def _parse_handicap(handicap_str) -> Optional[float]:
        """
        解析亚盘盘口字符串:
          "0" -> 0.0 (平手)
          "0.25" -> 0.25 (平/半)
          "0/0.5" -> 0.25 (平/半, 中文格式)
          "-0.5" -> -0.5 (客让)
          "1" -> 1.0 (一球)
        """
        if handicap_str is None:
            return None
        if isinstance(handicap_str, (int, float)):
            return float(handicap_str)
        s = str(handicap_str).strip()
        if not s:
            return None
        # 处理 "0/0.5" 格式 -> 0.25
        if "/" in s:
            try:
                parts = s.split("/")
                return (float(parts[0]) + float(parts[1])) / 2.0
            except (ValueError, TypeError):
                return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _normalize_europe_odds(self, odds_list: List[Dict]) -> List[Dict]:
        """标准化欧赔: [{company_id, company_name, home_win, draw, away_win, ...}]"""
        result = []
        for o in odds_list:
            cid = self._safe_int(o.get("companyId") or o.get("company_id"))
            result.append({
                "company_id": cid,
                "company_name": BOOKMAKER_NAMES.get(cid, f"公司{cid}"),
                "home_win": self._safe_float(o.get("homeWin") or o.get("home_win")),
                "draw": self._safe_float(o.get("draw") or o.get("draw_odd")),
                "away_win": self._safe_float(o.get("awayWin") or o.get("away_win")),
                "initial_home_win": self._safe_float(
                    o.get("initialHomeWin") or o.get("initial_home_win")),
                "initial_draw": self._safe_float(
                    o.get("initialDraw") or o.get("initial_draw")),
                "initial_away_win": self._safe_float(
                    o.get("initialAwayWin") or o.get("initial_away_win")),
                "update_time": o.get("updateTime") or o.get("update_time", ""),
            })
        return result

    def _normalize_asia_odds(self, odds_list: List[Dict]) -> List[Dict]:
        result = []
        for o in odds_list:
            cid = self._safe_int(o.get("companyId") or o.get("company_id"))
            result.append({
                "company_id": cid,
                "company_name": BOOKMAKER_NAMES.get(cid, f"公司{cid}"),
                "handicap": self._safe_float(o.get("handicap") or o.get("letBall")),
                "home_odds": self._safe_float(
                    o.get("homeOdds") or o.get("home_odds") or o.get("homeRate")),
                "away_odds": self._safe_float(
                    o.get("awayOdds") or o.get("away_odds") or o.get("awayRate")),
                "initial_handicap": self._safe_float(
                    o.get("initialHandicap") or o.get("initial_handicap")),
                "initial_home_odds": self._safe_float(
                    o.get("initialHomeOdds") or o.get("initial_home_odds")),
                "initial_away_odds": self._safe_float(
                    o.get("initialAwayOdds") or o.get("initial_away_odds")),
            })
        return result

    def _normalize_bigsmall_odds(self, odds_list: List[Dict]) -> List[Dict]:
        result = []
        for o in odds_list:
            cid = self._safe_int(o.get("companyId") or o.get("company_id"))
            result.append({
                "company_id": cid,
                "company_name": BOOKMAKER_NAMES.get(cid, f"公司{cid}"),
                "total_goals": self._safe_float(o.get("totalGoals") or o.get("total")),
                "over_odds": self._safe_float(
                    o.get("overOdds") or o.get("over_odds") or o.get("bigRate")),
                "under_odds": self._safe_float(
                    o.get("underOdds") or o.get("under_odds") or o.get("smallRate")),
            })
        return result

    @staticmethod
    def _safe_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(v) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    def close(self):
        self.session.close()
