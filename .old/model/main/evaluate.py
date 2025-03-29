import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_curve, auc
)
from typing import Dict, List, Tuple
import json
from pathlib import Path
import joblib

from utils import logger, safe_visualization
from visualize import VisualizationSystem

def evaluate_model(model: torch.nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  device: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """評估模型性能
    
    Args:
        model: 待評估的模型
        test_loader: 測試數據加載器
        device: 計算設備
        
    Returns:
        metrics: 評估指標字典
        predictions: 預測結果
        true_labels: 真實標籤
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
            true_labels.extend(targets.numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    metrics = calculate_metrics(true_labels, predictions)
    return metrics, predictions, true_labels

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """計算各項評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測值或機率
        
    Returns:
        包含各項指標的字典
    """
    # 轉換為二元預測
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary)
    }
    
    return metrics

def analyze_feature_importance(model: torch.nn.Module, 
                             X_test: np.ndarray, 
                             feature_names: List[str],
                             device: str) -> Dict[str, float]:
    """分析特徵重要性
    
    使用排列重要性方法評估每個特徵的重要程度
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        feature_names: 特徵名稱列表
        device: 計算設備
        
    Returns:
        特徵重要性得分字典
    """
    model.eval()
    importance_scores = {}
    
    # 獲取基準預測
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        base_pred = model(X_tensor)
    
    # 計算每個特徵的重要性
    for idx, feature in enumerate(feature_names):
        # 打亂特定特徵
        X_shuffled = X_test.copy()
        X_shuffled[:, idx] = np.random.permutation(X_shuffled[:, idx])
        
        # 獲取新預測
        X_shuffled_tensor = torch.FloatTensor(X_shuffled).to(device)
        with torch.no_grad():
            new_pred = model(X_shuffled_tensor)
        
        # 計算預測變化
        importance = torch.abs(base_pred - new_pred).mean().item()
        importance_scores[feature] = importance
    
    return importance_scores

def save_evaluation_results(metrics: Dict[str, float],
                          importance_scores: Dict[str, float],
                          save_dir: str = 'evaluation'):
    """保存評估結果
    
    Args:
        metrics: 模型評估指標
        importance_scores: 特徵重要性得分
        save_dir: 保存目錄
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存評估指標
    with open(save_path / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    # 保存特徵重要性
    with open(save_path / 'feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump(importance_scores, f, indent=4, ensure_ascii=False)
    
    logger.info(f"評估結果已保存至 {save_dir}")

def generate_evaluation_report(metrics: Dict[str, float],
                             importance_scores: Dict[str, float],
                             model_info: Dict,
                             save_path: str = 'evaluation_report.md'):
    """生成評估報告
    
    Args:
        metrics: 模型評估指標
        importance_scores: 特徵重要性得分
        model_info: 模型相關信息
        save_path: 保存路徑
    """
    report = [
        "# 異常檢測模型評估報告",
        "\n## 1. 模型概況",
        f"- 模型類型: {model_info.get('model_type', 'Unknown')}",
        f"- 參數量: {model_info.get('num_params', 'Unknown')}",
        f"- 訓練時間: {model_info.get('training_time', 'Unknown')} seconds",
        
        "\n## 2. 性能指標",
        f"- AUC: {metrics['auc']:.4f}",
        f"- 準確率: {metrics['accuracy']:.4f}",
        f"- F1分數: {metrics['f1']:.4f}",
        f"- 精確率: {metrics['precision']:.4f}",
        f"- 召回率: {metrics['recall']:.4f}",
        
        "\n## 3. 特徵重要性",
        "| 特徵 | 重要性得分 |",
        "|------|------------|"
    ]
    
    # 添加排序後的特徵重要性
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, score in sorted_features:
        report.append(f"| {feature} | {score:.4f} |")
    
    # 寫入報告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"評估報告已生成: {save_path}")

@safe_visualization
def visualize_evaluation_results(metrics: Dict[str, float],
                               predictions: np.ndarray,
                               true_labels: np.ndarray,
                               history: Dict[str, List[float]],
                               save_dir: str = 'img'):
    """視覺化評估結果
    
    Args:
        metrics: 評估指標
        predictions: 預測值
        true_labels: 真實標籤
        history: 訓練歷史
        save_dir: 圖表保存目錄
    """
    viz = VisualizationSystem()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 繪製 ROC 曲線
    viz.plot_roc_curve(
        true_labels,
        predictions,
        str(save_dir / 'roc_curve.png')
    )
    
    # 繪製訓練指標
    viz.plot_metrics(
        history,
        str(save_dir / 'training_metrics.png')
    )
    
    # 繪製混淆矩陣
    viz.plot_confusion_matrix(
        true_labels,
        (predictions > 0.5).astype(int),
        str(save_dir / 'confusion_matrix.png')
    )
    
    logger.info(f"評估圖表已保存至 {save_dir}")

def load_model_for_evaluation(model_path: str,
                            device: str) -> Tuple[torch.nn.Module, Dict]:
    """加載模型用於評估
    
    Args:
        model_path: 模型文件路徑
        device: 計算設備
        
    Returns:
        model: 加載的模型
        model_info: 模型相關信息
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        model_info = {
            'model_type': checkpoint.get('model_type', 'Unknown'),
            'num_params': checkpoint.get('num_params', 0),
            'training_time': checkpoint.get('training_time', 0),
            'best_metrics': checkpoint.get('best_metrics', {})
        }
        
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, model_info
        
    except Exception as e:
        logger.error(f"加載模型失敗: {str(e)}")
        raise