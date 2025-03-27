# 異常檢測專案

本專案專注於使用機器學習模型進行異常檢測，包含完整的資料處理、模型訓練與評估流程，以及REST API服務。

## 安裝與設定

### 1. 環境需求
- Python 3.8+
- NVIDIA GPU (建議 8GB+ VRAM)
- CUDA Toolkit 12.6
- cuDNN 8.7.0

### 2. 安裝 PyTorch
```bash
# 使用 CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. 安裝其他依賴
```bash
pip install -r requirements.txt
```

### 4. 驗證安裝
```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"可用的 GPU 數量: {torch.cuda.device_count()}")
```

### 5. 啟動專案
```bash
# 訓練模型
python main.py

# 啟動API服務
python main/api_server.py
# 或使用批次檔
start_api.bat
```

## API服務使用說明

本專案提供REST API接口，用於進行異常檢測。

### API端點

#### 1. 健康檢查
- **URL**: `/health`
- **方法**: GET
- **回應範例**:
```json
{
  "status": "ok",
  "message": "API服務正常運行中"
}
```

#### 2. 異常檢測
- **URL**: `/detect`
- **方法**: POST
- **內容類型**: application/json
- **請求格式**:
```json
{
  "features": [0.1, 0.2, 0.3, 0.4, ...]
}
```
- **回應範例**:
```json
{
  "status": "success",
  "predictions": [
    {
      "sample_id": 0,
      "anomaly_score": 0.87,
      "is_anomaly": true
    }
  ],
  "metadata": {
    "model_version": "1.0.0",
    "threshold": 0.5
  }
}
```

### API使用範例

使用curl發送請求:
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

使用Python發送請求:
```python
import requests
import json

url = "http://localhost:5000/detect"
payload = {
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

## 專案結構
- `anomaly_detection.py`: 異常檢測的主要腳本
- `config.py`: 專案的配置設定
- `data_loader.py`: 處理資料載入和預處理
- `evaluate.py`: 包含評估指標和邏輯
- `main.py`: 執行專案的入口點
- `model.py`: 定義機器學習模型的架構
- `train.py`: 模型訓練的腳本
- `utils.py`: 專案的實用函式
- `visualize.py`: 結果的視覺化工具
- `api_server.py`: REST API服務器實現
- `start_api.bat`: API服務啟動批次檔

## 資料夾結構
- `datasets/`: 儲存資料集的目錄
- `img/`: 儲存圖片的目錄
- `logs/`: 儲存日誌檔案的目錄
- `models/`: 儲存訓練好的模型的目錄

## 授權
此專案採用 MIT 授權。詳情請參閱 LICENSE 檔案。
