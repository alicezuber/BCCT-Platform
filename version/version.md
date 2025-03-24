# 版本更新記錄

## 版本 0.1.0 - 2025-03-22
### 新增功能
- 新增四則運算 API 伺服器 (api.py)
- 支援加法、減法、乘法和除法運算
- 支援使用符號 (+, -, *, /) 和文字 (add, subtract, multiply, divide) 進行運算
- 新增測試腳本 (test_api.py, test_manual.py) 驗證 API 功能
- 完善錯誤處理，包括除以零和無效運算的情況

### 技術細節
- 使用 Flask 框架實現 RESTful API
- 使用 pytest 進行自動化測試
- 提供詳細的中文測試輸出
