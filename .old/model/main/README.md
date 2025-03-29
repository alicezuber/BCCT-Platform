# 異常檢測專案

## 概述
此專案旨在使用機器學習模型檢測資料集中的異常。它包括資料載入、模型訓練、評估和視覺化的模組。

## 專案結構
- `anomaly_detection.py`: 異常檢測的主要腳本。
- `config.py`: 專案的配置設定。
- `data_loader.py`: 處理資料載入和預處理。
- `evaluate.py`: 包含評估指標和邏輯。
- `main.py`: 執行專案的入口點。
- `model.py`: 定義機器學習模型的架構。
- `requirements.txt`: 列出所需的 Python 套件。
- `train.py`: 模型訓練的腳本。
- `utils.py`: 專案的實用函式。
- `visualize.py`: 結果的視覺化工具。
- `datasets/`: 儲存資料集的目錄。
- `img/`: 儲存圖片的目錄。
- `logs/`: 儲存日誌檔案的目錄。
- `models/`: 儲存訓練好的模型的目錄。

## 設定
1. 複製此儲存庫。
2. 安裝依賴項：`pip install -r requirements.txt`。
3. 將您的資料集放入 `datasets/` 目錄中。

## 使用方式
使用以下命令執行專案：
```bash
python main.py
```

## 授權
此專案採用 MIT 授權。詳情請參閱 LICENSE 檔案。
