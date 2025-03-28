# Version History

## v0.1.9
- 修復 api_server.py 中模型檢查點加載的錯誤處理邏輯
- 優化 train.py 中的批次大小調整機制以應對顯存不足
- 添加訓練過程中的梯度檢查功能
- 增強指標計算的錯誤處理與日誌記錄
- 更新 focal loss 的實現以提升不平衡數據處理效果
- 增加測試時的數據增強功能 (TTA)
- 改進模型驗證階段的性能與穩定性
- 優化學習率調度器的參數設置
- 修正部分日誌訊息的格式與內容
- 添加最終模型狀態保存功能

## v0.1.8
- 修復來自使用者 @IWannaListenToUSayUwU 製造的拉取問題

## v0.1.7
- 新增異常檢測 REST API 服務 (api_server.py)
- 實現 POST 請求異常檢測功能端點 /detect
- 添加 API 服務啟動批次檔 (start_api.bat)
- 更新 README.md 添加 API 使用說明
- 添加 Flask 相關依賴到 requirements.txt

## v0.1.6
- 更新 PyTorch amp 相關呼叫以符合新版本要求
- 改善學習率調度器的警告訊息
- 優化指標計算的零除處理
- 增加中文化的指標說明日誌

## v0.1.5
- 修正 train.py 中的 autocast 使用方式
- 強制使用 CUDA 進行訓練
- 改進 GPU 相關的錯誤處理

## v0.1.4
- 更新 model/README.md 的安裝指示
- 新增 CUDA 和 PyTorch GPU 環境設定說明
- 加入環境驗證步驟

## v0.1.3
- 優化 main.py 與 train.py 的設備配置邏輯，支援無 CUDA 環境
- 新增設備檢查與適配機制
- 改進混合精度訓練的相容性

## v0.1.2
- 刪除和過濾不需要的檔案
- 建立gitignore

## v0.1.1
- 下拉最新程式

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

