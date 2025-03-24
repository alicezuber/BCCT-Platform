# Version History

## v0.1.0
- 上傳模型訓練程式
- 避免輸出被異化掉

## v0.0.8
- 新增網頁端API代理功能
- 實現pages/api/proxy.js作為前端和Python服務器的中間層
- 創建API演示頁面api-demo.js展示如何從前端調用Python API
- 加入錯誤處理和載入狀態顯示

## v0.0.7
- 修復 start.bat 批次檔案的編碼問題
- 添加 chcp 65001 命令以支援 UTF-8 編碼顯示中文
- 修正註解格式，確保使用標準 Windows 批處理語法

## v0.0.6
- 新增 start.bat 一鍵啟動批次檔案
- 實現同時啟動前端和 API 服務
- 增加環境檔案檢查和自動複製功能
- 提供友善的啟動狀態提示

## v0.0.5
- 重構 README.md 文件，改進格式和結構
- 添加功能特點部分
- 添加 Git 輔助工具使用說明
- 改進文件連結顯示方式
- 調整章節順序，提升閱讀流暢度

## v0.0.4
- 更新 README.md 文件中的 API 示例

## v0.0.3
- 新增 pages/index.js 檔案
- 實現根路徑 (/) 到 /home 頁面的服務端重定向功能
- 使用 getServerSideProps 進行 301 永久重定向，提高效能

## v0.0.2
- 更新 model/README.md 中的 IP 地址為 25.7.171.251 (與 .env 文件中的 ShiLaCow_ip 保持一致)
- 確保 README.md 文件中的所有 API 示例使用正確的 IP 地址
- 補充說明 IP 地址來源於 .env 文件配置

## v0.0.1
- 解決 .env 檔案未被正確忽略的問題，使用git rm --cached .env
- 建立 .env.example 範本檔案供參考

