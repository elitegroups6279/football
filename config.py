"""
天天盈球足球数据爬虫与分析系统 - 配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据存储目录
DATA_DIR = PROJECT_ROOT / "data"
MATCHES_DIR = DATA_DIR / "matches"
ODDS_DIR = DATA_DIR / "odds"
REPORTS_DIR = DATA_DIR / "reports"

# 确保目录存在
for d in [DATA_DIR, MATCHES_DIR, ODDS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== API 配置 ====================

# 天天盈球API基础地址
API_BASE_URL = "https://sport.ttyingqiu.com/sportdata/f"

# API请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.ttyingqiu.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.ttyingqiu.com",
}

# 网页基础地址
WEB_BASE_URL = "https://www.ttyingqiu.com"

# ==================== API接口名称 ====================

API_ENDPOINTS = {
    "hot_league_list": "getHotLeagueList",          # 热门联赛列表
    "league_list": "getLeagueList",                  # 联赛列表
    "country_list": "getCountryList",                # 国家列表
    "today_league_match": "getTodayLeagueMatchList", # 今日赛事
    "league_match_list": "getLeagueMatchList",       # 联赛比赛列表
    "europe_odds": "getListMatchEuropeOdds",         # 欧洲赔率
    "asia_odds": "getListMatchAsiaPrimaryOdds",     # 亚洲盘口
    "bigsmall_odds": "getListMatchBigSmallOdds",     # 大小球
    "match_info": "getMatchListById",                # 比赛信息
}

# 已知可用（无需参数）的API
KNOWN_WORKING_APIS = ["getHotLeagueList"]

# ==================== 爬取配置 ====================

# 请求超时（秒）
REQUEST_TIMEOUT = 30

# 请求重试次数
MAX_RETRIES = 3

# 请求间隔（秒）- 避免被封
REQUEST_INTERVAL = 1.0

# 最大并发请求数
MAX_CONCURRENT = 5

# ==================== 联赛配置 ====================

# 关注的联赛ID（热门联赛）
HOT_LEAGUE_IDS = [
    2000,  # 世欧预
    2044,  # 欧冠
    2079,  # 英超
    2105,  # 意甲
    2117,  # 西甲
    2125,  # 德甲
    2483,  # 亚冠
    2558,  # 日职联
    2559,  # 日职乙
    2572,  # 韩K联
    2579,  # 澳超
]

# ==================== 数据分析配置 ====================

# 赔率公司名称映射
BOOKMAKER_NAMES = {
    1: "澳门",
    2: "威廉希尔",
    3: "立博",
    4: "Bet365",
    8: "易胜博",
    9: "伟德",
    12: "平博",
    14: "Interwetten",
    17: "皇冠",
    19: "10BET",
    22: "利记",
    23: "金宝博",
    24: "12bet",
    35: "盈禾",
    42: "18Bet",
    45: "Crown",
    48: "香港马会",
    928: "必发",
}

# 目标联赛用于模型训练
TARGET_LEAGUES_FOR_MODEL = [2079, 2105, 2117, 2125]  # 英超 意甲 西甲 德甲

# ==================== 预测系统配置 ====================

# 预测数据存储目录
PREDICTIONS_DIR = DATA_DIR / "predictions"
HISTORICAL_DIR = DATA_DIR / "historical"
MODELS_DIR = DATA_DIR / "models"

# 确保目录存在
for d in [PREDICTIONS_DIR, HISTORICAL_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 竞彩足球游戏ID (天天盈球定义)
GAME_JINGCAI = 407   # 竞彩足球
GAME_BEIDAN = 408    # 北京单场

# 默认投注参数
DEFAULT_BANKROLL = 1000.0        # 默认本金 (元)
MIN_EV_THRESHOLD = 0.05          # 最小期望值阈值 (5%)
DEFAULT_KELLY_FRACTION = 0.25    # 默认凯利系数 (1/4凯利)
MAX_RECOMMENDATIONS = 10         # 最大推荐场次数

# 置信度评分权重
CONFIDENCE_WEIGHTS = {
    "ev_score": 0.30,           # EV得分权重
    "odds_consensus": 0.20,     # 赔率共识度权重
    "poisson_fit": 0.25,        # 泊松模型拟合度权重
    "league_trend": 0.15,       # 联赛趋势权重
    "margin_stability": 0.10,   # 边际消除稳定性权重
}
