import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import logger, safe_visualization

class VisualizationSystem:
    """模塊化可視化系統"""
    
    def __init__(self):
        """初始化可視化系統"""
        self.setup_output_dir()
    
    @staticmethod
    def setup_output_dir():
        """設置輸出目錄"""
        img_dir = Path('img')
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    @safe_visualization
    def plot_training_loss(self, history: dict, save_path: str):
        """繪製訓練和驗證損失曲線"""
        plt.figure(figsize=(15, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        """繪製 ROC 曲線"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(15, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_metrics(self, history: dict, save_path: str):
        """繪製訓練指標"""
        fig, ax = plt.subplots(2, 2, figsize=(20, 12))
        
        # Loss curve
        ax[0,0].plot(history['train_loss'], label='Train')
        ax[0,0].plot(history['val_loss'], label='Validation')
        ax[0,0].set_title('Loss Curve')
        ax[0,0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[0,0].grid(True)
        
        # AUC curve
        ax[0,1].plot(history['auc'], color='green', label='AUC')
        ax[0,1].set_title('Validation AUC')
        ax[0,1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[0,1].grid(True)
        
        # Precision/Recall curves
        ax[1,0].plot(history['precision'], label='Precision')
        ax[1,0].plot(history['recall'], label='Recall')
        ax[1,0].set_title('Precision/Recall')
        ax[1,0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[1,0].grid(True)
        
        # F1 score curve
        ax[1,1].plot(history['f1'], color='red', label='F1')
        ax[1,1].set_title('F1 Score')
        ax[1,1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_feature_importance(self, importance_df: pd.DataFrame, save_path: str):
        """繪製特徵重要性圖表"""
        plt.figure(figsize=(15, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features')
        plt.xlabel('Feature Importance Score')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_attention_weights(self, attention_weights: List[np.ndarray], save_path: str):
        """繪製注意力權重熱力圖"""
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(num_layers, 1, figsize=(15, 4*num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for idx, weights in enumerate(attention_weights):
            if isinstance(weights, np.ndarray):
                im = axes[idx].imshow(weights, cmap='viridis', aspect='auto')
                axes[idx].set_title(f'Layer {idx+1} Attention Weights')
                plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_prediction_distribution(self, predictions: np.ndarray, save_path: str):
        """繪製預測值分布圖"""
        plt.figure(figsize=(12, 6))
        sns.histplot(predictions, bins=50)
        plt.title('Distribution of Prediction Scores')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def plot_learning_rate(self, lr_history: List[float], save_path: str):
        """繪製學習率變化曲線"""
        plt.figure(figsize=(12, 6))
        plt.plot(lr_history)
        plt.title('Learning Rate Over Time')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @safe_visualization
    def create_summary_plot(self, metrics: Dict[str, float], save_path: str):
        """創建性能指標總覽圖"""
        plt.figure(figsize=(12, 6))
        metrics_items = list(metrics.items())
        names = [item[0] for item in metrics_items]
        values = [item[1] for item in metrics_items]
        
        bars = plt.bar(names, values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        
        # 在柱狀圖上添加數值標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def generate_full_report(self, 
                           history: dict,
                           predictions: np.ndarray,
                           true_labels: np.ndarray,
                           feature_importance: pd.DataFrame,
                           attention_weights: List[np.ndarray],
                           metrics: Dict[str, float],
                           save_dir: str = 'img'):
        """生成完整的視覺化報告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 繪製所有圖表
        self.plot_training_loss(history, save_dir / 'training_loss.png')
        self.plot_roc_curve(true_labels, predictions, save_dir / 'roc_curve.png')
        self.plot_confusion_matrix(true_labels, (predictions > 0.5).astype(int),
                                 save_dir / 'confusion_matrix.png')
        self.plot_metrics(history, save_dir / 'all_metrics.png')
        self.plot_feature_importance(feature_importance, save_dir / 'feature_importance.png')
        self.plot_attention_weights(attention_weights, save_dir / 'attention_weights.png')
        self.plot_prediction_distribution(predictions, save_dir / 'pred_distribution.png')
        self.create_summary_plot(metrics, save_dir / 'summary.png')
        
        logger.info(f"所有視覺化圖表已保存至: {save_dir}")
        
        # 生成 HTML 報告
        self._generate_html_report(save_dir)

    def _generate_html_report(self, img_dir: Path):
        """生成 HTML 格式的視覺化報告"""
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>異常檢測模型評估報告</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "img { max-width: 100%; margin: 20px 0; }",
            ".section { margin-bottom: 40px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>異常檢測模型評估報告</h1>"
        ]
        
        # 添加所有圖表
        for img_path in img_dir.glob("*.png"):
            html_content.extend([
                f"<div class='section'>",
                f"<h2>{img_path.stem}</h2>",
                f"<img src='{img_path.name}' alt='{img_path.stem}'>",
                "</div>"
            ])
        
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        # 保存 HTML 報告
        with open(img_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"HTML 報告已生成: {img_dir / 'report.html'}")