from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import json
import logging
from utils import setup_logger, load_config
from model import create_model
import traceback

# 設置日誌
logger = logging.getLogger("anomaly_detection_api")
setup_logger(logger, "api")

app = Flask(__name__)

# 載入配置
config = load_config()

# 全局變量存儲模型
model = None
device = None

def load_model():
    """載入預訓練的異常檢測模型"""
    global model, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(config)
        
        # 找到最新的模型檢查點
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            logger.error("未找到模型檢查點文件")
            return False
            
        # 選擇最新的檢查點
        latest_checkpoint = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
        
        # 載入模型權重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"成功載入模型檢查點: {latest_checkpoint}")
        return True
    except Exception as e:
        logger.error(f"載入模型時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    if model is None:
        return jsonify({"status": "error", "message": "模型未初始化"}), 503
    return jsonify({"status": "ok", "message": "API服務正常運行中"}), 200

@app.route('/detect', methods=['POST'])
def detect_anomaly():
    """異常檢測端點"""
    if model is None:
        return jsonify({"status": "error", "message": "模型未初始化"}), 503
        
    try:
        # 從請求中獲取數據
        data = request.json
        if not data or 'features' not in data:
            return jsonify({"status": "error", "message": "缺少必要的特徵數據"}), 400
            
        # 處理輸入數據
        features = np.array(data['features'], dtype=np.float32)
        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # 如果數據是單個樣本，添加批次維度
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        # 使用模型進行預測
        with torch.no_grad():
            output = model(input_tensor)
            
        # 處理輸出
        predictions = output.cpu().numpy()
        anomaly_scores = predictions.flatten().tolist()
        threshold = 0.5  # 可根據需要調整閾值
        is_anomaly = [score > threshold for score in anomaly_scores]
        
        # 準備響應
        result = {
            "status": "success",
            "predictions": [
                {"sample_id": i, "anomaly_score": score, "is_anomaly": anomaly} 
                for i, (score, anomaly) in enumerate(zip(anomaly_scores, is_anomaly))
            ],
            "metadata": {
                "model_version": config.model_version,
                "threshold": threshold
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"預測過程中發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"處理請求時發生錯誤: {str(e)}"}), 500

if __name__ == '__main__':
    # 在啟動服務器前載入模型
    if load_model():
        logger.info("模型載入成功，API服務準備就緒")
        # 預設在本地運行，生產環境中可能需要調整host和port
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("模型載入失敗，API服務無法啟動")