@echo off
:: filepath: d:\project\.HomeWorkProject\BCCT-platform\start.bat
chcp 65001 >nul
title BCCT-Platform 一鍵啟動工具
echo ===================================
echo     BCCT-Platform 啟動工具
echo ===================================
echo.

:: 檢查環境變數檔案
if not exist .env (
    echo [警告] 找不到 .env 檔案，將使用預設配置
    if exist .env.example (
        echo [資訊] 嘗試從 .env.example 複製...
        copy .env.example .env
        echo [成功] 已建立 .env 檔案，請檢查配置是否正確
    ) else (
        echo [錯誤] .env.example 也不存在，請先建立環境配置檔案
        pause
        exit /b 1
    )
)

echo [資訊] 啟動 API 服務...
start cmd /k "cd model && python -m pip install flask flask-cors python-dotenv -q && python server.py"

echo [資訊] 啟動前端應用...
start cmd /k "cd my-new-dapp && npm install && npm run dev"

echo.
echo [成功] BCCT-Platform 已啟動
echo 前端應用訪問地址: http://localhost:3000
echo API 服務運行於 .env 檔案中配置的 ShiLaCow_ip 地址
echo.
echo 請勿關閉此視窗，關閉此視窗將終止所有服務
pause