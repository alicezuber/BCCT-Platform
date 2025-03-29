import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import Config, parse_arguments
from data_loader import DataGenerator, save_dataset, AnomalyDataset
from model import EnhancedAnomalyDetector
from train import TrainingEngine
from evaluate import evaluate_model, analyze_feature_importance, save_evaluation_results, generate_evaluation_report
from visualize import VisualizationSystem
from utils import logger, setup_matplotlib_style

def setup_environment():
    """初始化運行環境"""
    # 設置種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 檢查 CUDA 可用性並設置設備
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
        logger.info("使用 CUDA 進行訓練")
    else:
        logger.info("CUDA 不可用，使用 CPU 進行訓練")
    
    # 設置輸出目錄
    directories = ['datasets', 'models', 'logs', 'img']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 設置matplotlib樣式
    setup_matplotlib_style()

def main():
    """主程序入口"""
    try:
        # 初始化環境
        setup_environment()
        
        # 解析參數
        args = parse_arguments()
        config = Config.from_args(args)
        
        # 生成數據集
        logger.info("開始生成數據集...")
        data_generator = DataGenerator(vars(config))
        datasets = data_generator.generate_datasets()
        
        # 保存數據集
        save_paths = save_dataset(datasets, args.save_path)
        logger.info(f"數據集已保存至: {save_paths}")
        
        # 數據預處理
        logger.info("開始數據預處理...")
        dataset = AnomalyDataset(*datasets)
        X_scaled, y, feature_columns = dataset.preprocess()
        
        # 創建數據加載器
        train_loader, val_loader, test_loader = dataset.create_data_loaders(config)
        
        # 初始化模型
        logger.info("初始化模型...")
        model = EnhancedAnomalyDetector(config, input_dim=X_scaled.shape[1])
        logger.info(f"模型總參數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 訓練模型
        logger.info("開始訓練模型...")
        engine = TrainingEngine(config, model, train_loader, val_loader)
        engine.train()
        
        # 評估模型
        logger.info("開始模型評估...")
        metrics, predictions, true_labels = evaluate_model(model, test_loader, config.device)
        
        # 分析特徵重要性
        logger.info("分析特徵重要性...")
        importance_scores = analyze_feature_importance(
            model=model,
            X_test=X_scaled,
            feature_names=feature_columns,
            device=config.device
        )
        
        # 保存評估結果
        save_evaluation_results(metrics, importance_scores)
        
        # 生成評估報告
        model_info = {
            'model_type': 'EnhancedAnomalyDetector',
            'num_params': sum(p.numel() for p in model.parameters()),
            'training_time': engine.training_state['train_time']
        }
        generate_evaluation_report(metrics, importance_scores, model_info)
        
        # 視覺化結果
        logger.info("生成視覺化報告...")
        viz = VisualizationSystem()
        viz.generate_full_report(
            history=engine.history,
            predictions=predictions,
            true_labels=true_labels,
            feature_importance=pd.DataFrame({
                'feature': feature_columns,
                'importance': list(importance_scores.values())
            }),
            attention_weights=model.attention_weights,
            metrics=metrics
        )
        
        logger.info("程序執行完成!")
        
    except Exception as e:
        logger.error(f"程序執行失敗: {str(e)}")
        logger.error("堆疊追踪:", exc_info=True)
        raise

if __name__ == "__main__":
    main()