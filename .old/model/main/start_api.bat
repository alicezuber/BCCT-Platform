@echo off
echo 正在啟動異常檢測 API 服務...
cd %~dp0
set PYTHONPATH=%~dp0..;%PYTHONPATH%

chcp 65001 > nul
python api_server.py

pause