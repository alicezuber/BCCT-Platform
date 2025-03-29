from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import json
import logging
from utils import setup_logger, load_config
from model import create_model
import traceback
from flask_cors import CORS

# 設置日誌
logger = logging.getLogger("anomaly_detection_api")
setup_logger(logger, "api")

app = Flask(__name__)
CORS(app)  # 啟用跨域支援

# 載入配置
config = load_config()

# 全局變量存儲模型
model = None
device = None
input_dim = None

def load_model():
    """載入預訓練的異常檢測模型"""
    global model, device, input_dim
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 檢查配置中是否有輸入維度
        input_dim = getattr(config, 'input_dim', 10)
        logger.info(f"使用配置中的輸入維度: {input_dim}")

        # 創建模型
        model = create_model(config, input_dim=input_dim)
        
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
    return jsonify({
        "status": "ok", 
        "message": "API服務正常運行中",
        "model_info": {
            "input_dim": input_dim,
            "device": str(device)
        }
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect_anomaly():
    """異常檢測端點 - 傳入特徵數據"""
    if model is None:
        return jsonify({"status": "error", "message": "模型未初始化"}), 503
        
    try:
        # 從請求中獲取數據
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "請求中缺少JSON數據"}), 400
            
        # 檢查請求格式
        if 'features' in data:
            # 列表格式的特徵
            features = np.array(data['features'], dtype=np.float32)
        else:
            return jsonify({
                "status": "error", 
                "message": "請求格式錯誤，需要 'features' 欄位",
                "example": {
                    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                }
            }), 400
            
        # 檢查特徵維度
        if len(features.shape) == 1:
            if features.shape[0] != input_dim and input_dim is not None:
                return jsonify({
                    "status": "error", 
                    "message": f"特徵數量不匹配，需要 {input_dim} 個特徵，但提供了 {features.shape[0]} 個",
                    "expected_features": input_dim,
                    "received_features": features.shape[0]
                }), 400
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            if features.shape[1] != input_dim and input_dim is not None:
                return jsonify({
                    "status": "error", 
                    "message": f"特徵數量不匹配，需要 {input_dim} 個特徵，但提供了 {features.shape[1]} 個"
                }), 400
            input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            
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
                "threshold": threshold,
                "feature_count": input_dim
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"預測過程中發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"處理請求時發生錯誤: {str(e)}"
        }), 500

@app.route('/api/docs', methods=['GET'])
def get_api_docs():
    """API文檔端點"""
    docs = {
        "api_version": "1.0.0",
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "健康檢查端點，用於確認API服務狀態",
                "response_example": {
                    "status": "ok", 
                    "message": "API服務正常運行中",
                    "model_info": {
                        "input_dim": 10,
                        "device": "cuda"
                    }
                }
            },
            {
                "path": "/api/detect",
                "method": "POST",
                "description": "異常檢測端點，用於根據輸入特徵進行異常檢測",
                "request_formats": [
                    {
                        "format": "特徵列表格式",
                        "example": {
                            "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        },
                        "note": "特徵值以數值列表形式傳入，必須按順序提供所有特徵"
                    }
                ],
                "response_example": {
                    "status": "success",
                    "predictions": [
                        {"sample_id": 0, "anomaly_score": 0.87, "is_anomaly": true}
                    ],
                    "metadata": {
                        "model_version": "1.0.0",
                        "threshold": 0.5,
                        "feature_count": 10
                    }
                }
            },
            {
                "path": "/api/docs",
                "method": "GET",
                "description": "API文檔端點，提供API使用說明"
            }
        ]
    }
    return jsonify(docs), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "找不到請求的資源",
        "available_endpoints": ["/health", "/api/detect", "/api/docs"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "message": "請求方法不允許",
        "hint": "請檢查您是否使用了正確的HTTP方法(GET/POST)"
    }), 405

if __name__ == '__main__':
    # 在啟動服務器前載入模型
    if load_model():
        logger.info("模型載入成功，API服務準備就緒")
        # 預設在本地運行，生產環境中可能需要調整host和port
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("模型載入失敗，API服務無法啟動")