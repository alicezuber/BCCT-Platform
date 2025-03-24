import argparse
import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """統一配置管理系統"""
    def __init__(self):
        # 原有配置
        self.seed: int = 42
        self.num_orders: int = 50000
        self.num_days: int = 2000
        
        # 模型架構配置
        self.hidden_dim: int = 256  # 增加隱藏層維度
        self.lstm_layers: int = 3   # 增加 LSTM 層數
        self.transformer_layers: int = 4  # 增加 Transformer 層數
        self.num_heads: int = 8
        self.dropout: float = 0.3   # 增加 dropout
        
        # 訓練配置
        self.batch_size: int = 128  # 增大批次大小
        self.epochs: int = 200      # 增加訓練輪數
        self.lr: float = 2e-4       # 調整學習率
        self.weight_decay: float = 0.02  # 增加權重衰減
        self.gradient_accumulation: int = 2  # 增加梯度累積步數
        self.mixed_precision: bool = True
        self.early_stop: int = 15   # 增加早停輪數
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 新增配置
        self.scheduler_patience: int = 8  # 學習率調度器的耐心值
        self.scheduler_factor: float = 0.5  # 學習率調整因子
        self.warmup_epochs: int = 5  # 預熱訓練輪數
        self.label_smoothing: float = 0.1  # 標籤平滑係數
        self.tta_num: int = 8  # TTA 數量
        self.focal_loss_alpha: float = 0.25  # Focal Loss alpha 參數
        self.focal_loss_gamma: float = 2.0  # Focal Loss gamma 參數

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """從命令行參數加載配置"""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

def parse_arguments() -> argparse.Namespace:
    """統一參數解析"""
    parser = argparse.ArgumentParser(description='異常檢測模型訓練程式')
    
    # 數據集相關參數
    parser.add_argument('--num_events', type=int, default=1000, help='事件數量')
    parser.add_argument('--num_orders', type=int, default=50000, help='訂單數量')
    parser.add_argument('--num_days', type=int, default=2000, help='時間跨度（天）')
    
    # 模型相關參數
    parser.add_argument('--hidden_dim', type=int, default=128, help='隱藏層維度')
    parser.add_argument('--lstm_layers', type=int, default=2, help='LSTM層數')
    parser.add_argument('--transformer_layers', type=int, default=3, help='Transformer層數')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力頭數')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='訓練輪數')
    parser.add_argument('--lr', type=float, default=0.0001, help='學習率')
    parser.add_argument('--price_loss_weight', type=float, default=0.1, help='價格預測損失權重')
    
    # 其他配置
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='運算設備')
    parser.add_argument('--save_path', type=str, default='datasets', help='數據保存路徑')
    parser.add_argument('--model_path', type=str, default='anomaly_detector.pth', help='模型保存路徑')

    # 優化參數
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='梯度累積步數')
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度訓練')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='權重衰減係數')
    
    args = parser.parse_args()
    
    # 更新 DATA_CONFIG
    global DATA_CONFIG
    DATA_CONFIG = {
        'num_events': args.num_events,
        'num_days': args.num_days,
        'num_orders': args.num_orders
    }
    
    # 參數驗證
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("Dropout率必須在0到1之間")
        
    if args.gradient_accumulation < 1:
        raise ValueError("梯度累積步數必須大於0")
        
    if args.lr <= 0:
        raise ValueError("學習率必須大於0")
    
    return args