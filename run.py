#!/usr/bin/env python
"""
启动脚本 - 使用正确的Python解释器路径
"""
import sys
import os

# 使用当前目录下的Python或系统默认Python
python_paths = [
    r"C:\Users\admin\python-sdk\python3.13.2\python.exe",
]

for path in python_paths:
    if os.path.exists(path):
        python_exe = path
        break
else:
    python_exe = sys.executable

# 运行主程序
import subprocess
cmd = [python_exe, "main.py"] + sys.argv[1:]
sys.exit(subprocess.run(cmd).returncode)
