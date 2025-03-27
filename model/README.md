# 異常檢測專案

本專案專注於使用機器學習模型進行異常檢測，包含完整的資料處理、模型訓練與評估流程。

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
python main.py
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

## 資料夾結構
- `datasets/`: 儲存資料集的目錄
- `img/`: 儲存圖片的目錄
- `logs/`: 儲存日誌檔案的目錄
- `models/`: 儲存訓練好的模型的目錄

## 授權
此專案採用 MIT 授權。詳情請參閱 LICENSE 檔案。
