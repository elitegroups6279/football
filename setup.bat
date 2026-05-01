@echo off
chcp 65001 >nul
echo ============================================
echo  足球数据爬虫与分析系统 - 环境安装脚本
echo ============================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] 正在安装 Python 3.13...
    echo 请从 https://www.python.org/downloads/ 下载并安装 Python 3.13+
    echo 安装时请确保勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] Python 已安装
python --version

echo.
echo [2/3] 正在安装依赖包...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

if %errorlevel% neq 0 (
    echo 安装失败，尝试使用镜像源...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -q
)

echo.
echo [3/3] 安装完成！
echo.
echo ============================================
echo  使用说明:
echo ============================================
echo.
echo  python main.py --help       查看帮助
echo  python main.py --scrape     爬取比赛数据
echo  python main.py --odds       爬取赔率数据
echo  python main.py --eda        执行EDA分析
echo  python main.py --all        执行完整流程
echo  python main.py --all --report  全流程+生成报告
echo.
pause
