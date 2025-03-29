
# 首先定義日誌系統
import os
import sys
import time
import math
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass
from collections import deque

def setup_logging():
    """配置日誌系統"""
    try:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'anomaly_detection_{timestamp}.log'
        
        try:
            log_file.touch()
        except Exception as e:
            print(f"無法創建日誌文件，將使用當前目錄: {str(e)}")
            log_file = Path(f'anomaly_detection_{timestamp}.log')
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 配置文件處理器 - 設置為 WARNING 以上級別
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        
        # 配置控制台處理器 - 設置為 INFO 以上級別，但限制某些訊息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 配置根日誌記錄器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    except Exception as e:
        print(f"日誌系統初始化失敗: {str(e)}")
        print("將只使用控制台輸出")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levellevel)s - %(message)s'
        )
        return logging.getLogger()

# 設置日誌系統
logger = setup_logging()

# 其他導入
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import joblib
import traceback
import functools

# 更新性能優化模塊導入
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import psutil

try:
    import GPUtil
    HAS_GPU_UTIL = True
    logger.info("成功導入 GPUtil，將能夠監控 GPU 使用情況")
except ImportError:
    HAS_GPU_UTIL = False
    logger.warning("GPUtil 未安裝，將不能監控 GPU 使用情況")
    logger.info("可以使用 'pip install gputil' 安裝 GPUtil 以啟用 GPU 監控")

# 自定義異常類別
class DataProcessingError(Exception):
    """數據處理錯誤的自定義異常"""
    pass

class ModelError(Exception):
    """模型相關錯誤的自定義異常"""
    pass

class VisualizationError(Exception):
    """視覺化相關錯誤的自定義異常"""
    pass

def safe_data_processing(func):
    """數據處理裝飾器，用於安全處理數據相關操作"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} 執行失敗: {str(e)}")
            logger.error(traceback.format_exc())
            raise DataProcessingError(f"{func.__name__} 失敗: {str(e)}")
    return wrapper

def safe_visualization(func):
    """視覺化裝飾器，用於安全處理圖表繪製"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"視覺化 {func.__name__} 失敗: {str(e)}")
            logger.error(traceback.format_exc())
            raise VisualizationError(f"視覺化失敗: {str(e)}")
    return wrapper

class ErrorHandler:
    """統一錯誤處理類"""
    @staticmethod
    def handle_data_error(e: Exception, context: str = ""):
        """處理數據相關錯誤"""
        logger.error(f"數據處理錯誤 {context}: {str(e)}")
        logger.error(traceback.format_exc())
        raise DataProcessingError(f"數據處理錯誤 {context}: {str(e)}")
    
    @staticmethod
    def handle_model_error(e: Exception, context: str = ""):
        """處理模型相關錯誤"""
        logger.error(f"模型錯誤 {context}: {str(e)}")
        logger.error(traceback.format_exc())
        raise ModelError(f"模型錯誤 {context}: {str(e)}")
    
    @staticmethod
    def handle_visualization_error(e: Exception, context: str = ""):
        """處理視覺化相關錯誤"""
        logger.error(f"視覺化錯誤 {context}: {str(e)}")
        logger.error(traceback.format_exc())
        raise VisualizationError(f"視覺化錯誤 {context}: {str(e)}")

# 設置 matplotlib 樣式和參數
def setup_matplotlib_style():
    """配置 matplotlib 的顯示樣式和參數"""
    try:
        # 基礎配置
        plt.rcParams.update({
            # 圖表尺寸與品質
            'figure.figsize': [10, 6],
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # 網格設置
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.color': '#cccccc',
            
            # 字體大小
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            
            # 線條樣式
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'lines.markeredgewidth': 1,
            
            # 散點圖樣式
            'scatter.marker': '.',
            'scatter.edgecolors': 'none',
            
            # 配色方案
            'axes.prop_cycle': plt.cycler(color=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ])
        })
        
        # 嘗試設置中文字體
        font_families = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'Arial', 'DejaVu Sans']
        for font in font_families:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
                # 測試字體是否可用
                fig = plt.figure()
                plt.text(0.5, 0.5, '測試中文', fontsize=12)
                plt.close(fig)
                logger.info(f"成功設置字體: {font}")
                break
            except Exception as e:
                logger.warning(f"字體 {font} 設置失敗: {str(e)}")
                continue
        
        plt.rcParams['axes.unicode_minus'] = False
        logger.info("matplotlib 樣式設置完成")
        
    except Exception as e:
        logger.error(f"matplotlib 樣式設置失敗: {str(e)}")
        logger.error(traceback.format_exc())

# 初始化 matplotlib 配置
setup_matplotlib_style()

# ========================
# 1. 增強型配置系統
# ========================
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

# ========================
# 2. 內存優化數據加載
# ========================
class MemoryOptimizedDataset(Dataset):
    """支持內存映射和延遲加載的數據集"""
    def __init__(self, tensor: torch.Tensor, use_mmap: bool = True):
        self.use_mmap = use_mmap
        if use_mmap and not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor)
            
        if use_mmap:
            self.data = np.memmap(
                tensor.numpy(),
                dtype=tensor.numpy().dtype,
                mode='r',
                shape=tensor.shape
            )
        else:
            self.data = tensor
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data[idx]
        return torch.from_numpy(np.array(sample)) if self.use_mmap else self.data[idx]

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.gmm = GaussianMixture(n_components=3, random_state=42)
        
    def generate_base_timeseries(self, num_days):
        """生成基礎時間序列，包含趨勢、季節性和噪聲"""
        t = np.linspace(0, 4*np.pi, num_days)
        trend = t * 0.1
        seasonal = np.sin(t) + 0.5 * np.sin(2*t)
        noise = np.random.normal(0, 0.1, num_days)
        return trend, seasonal, noise
    
    def generate_correlated_features(self):
        """生成相關特徵"""
        num_days = self.config['num_days']
        
        # 基礎時間序列
        trend, seasonal, noise = self.generate_base_timeseries(num_days)
        
        # GDP 影響因子
        gdp_base = np.exp(trend + 0.3*seasonal + 0.1*noise) * 1e9
        gdp_growth = np.cumsum(np.random.normal(0.001, 0.0005, num_days))
        gdp = gdp_base * (1 + gdp_growth)
        
        # 碳排放與 GDP 相關
        carbon_emission = (gdp/1e9) * (0.8 + 0.2*np.random.random(num_days))
        carbon_emission += np.random.normal(0, 50, num_days)
        
        # 可再生能源與碳排放負相關
        renewable_energy = 2000 - carbon_emission * 0.5
        renewable_energy *= (1 + 0.1*seasonal + 0.05*noise)
        renewable_energy = np.maximum(renewable_energy, 0)
        
        # 碳價與供需相關
        carbon_price = 30 + 5*seasonal + 2*noise
        carbon_price += 0.1 * (carbon_emission - renewable_energy)/100
        carbon_price = np.clip(carbon_price, 10, 100)
        
        # 碳交易量與價格、GDP相關
        carbon_trade_vol = (carbon_price * 100 + gdp/1e8) * (1 + 0.2*noise)
        
        return {
            'GDP': gdp,
            'carbon_emission': carbon_emission,
            'renewable_energy': renewable_energy,
            'carbon_price': carbon_price,
            'carbon_trade_vol': carbon_trade_vol
        }
    
    def generate_anomalous_orders(self, num_orders, features):
        """使用 GMM 生成異常訂單，確保總數量符合要求"""
        # 計算正常訂單和異常訂單的數量
        anomaly_ratio = 0.1
        num_normal = int(num_orders * (1 - anomaly_ratio))
        num_anomalies = int(num_orders * anomaly_ratio)
        
        # 確保總數等於 num_orders
        if num_normal + num_anomalies != num_orders:
            num_normal += (num_orders - (num_normal + num_anomalies))
        
        # 生成正常訂單
        normal_data = np.column_stack([
            np.random.normal(25000, 5000, num_normal),  # order_amount
            np.random.normal(50, 10, num_normal),      # carbon_amount
        ])
        
        # 訓練 GMM
        self.gmm.fit(normal_data)
        
        # 生成異常點
        anomalous_data = np.random.uniform(
            low=[40000, 80],
            high=[60000, 120],
            size=(num_anomalies, 2)
        )
        
        # 組合數據
        all_data = np.vstack([normal_data, anomalous_data])
        is_anomalous = np.zeros(num_orders, dtype=bool)
        is_anomalous[num_normal:] = True
        
        # 打亂數據順序
        shuffle_idx = np.random.permutation(num_orders)
        all_data = all_data[shuffle_idx]
        is_anomalous = is_anomalous[shuffle_idx]
        
        # 驗證數據長度
        assert len(all_data) == num_orders, \
            f"Data length mismatch: {len(all_data)} != {num_orders}"
        assert len(is_anomalous) == num_orders, \
            f"Label length mismatch: {len(is_anomalous)} != {num_orders}"
            
        return all_data, is_anomalous
    
    def generate_datasets(self):
        """生成完整數據集"""
        print(f"開始生成數據集，目標數量：{self.config['num_orders']} 筆訂單")
        
        # 生成時間序列特徵
        timeseries_features = self.generate_correlated_features()
        
        # 生成訂單數據
        order_data, anomalies = self.generate_anomalous_orders(
            self.config['num_orders'],
            timeseries_features
        )
        
        # 創建日期範圍
        dates = pd.date_range(
            start='2020-01-01',
            periods=self.config['num_days'],
            freq='D'
        )
        
        # 外部數據
        external_df = pd.DataFrame({
            'record_date': dates,
            'carbon_emission': timeseries_features['carbon_emission'],
            'GDP': timeseries_features['GDP'],
            'renewable_energy': timeseries_features['renewable_energy'],
            'carbon_price': timeseries_features['carbon_price'],
            'carbon_trade_vol': timeseries_features['carbon_trade_vol']
        })
        
        # 生成隨機訂單時間，確保在日期範圍內
        order_dates = np.random.choice(dates, size=len(order_data), replace=True)
        
        # 訂單數據
        orders_df = pd.DataFrame({
            'order_id': range(1, len(order_data) + 1),
            'order_amount': order_data[:, 0],
            'carbon_amount': order_data[:, 1],
            'order_time': order_dates,
            'is_anomalous': anomalies
        })
        
        # 驗證數據長度
        assert len(orders_df) == self.config['num_orders'], \
            f"Orders length mismatch: {len(orders_df)} != {self.config['num_orders']}"
        assert len(external_df) == self.config['num_days'], \
            f"External data length mismatch: {len(external_df)} != {self.config['num_days']}"
            
        print(f"生成的數據集大小：")
        print(f"訂單數據：{len(orders_df)} 筆")
        print(f"外部數據：{len(external_df)} 筆")
        
        return orders_df, external_df

# 修改數據生成部分
def generate_datasets(config):
    data_generator = DataGenerator(config)
    return data_generator.generate_datasets()

# ========================
# 3. 模型架構優化
# ========================
class PositionalEncoding(nn.Module):
    """改進的位置編碼實現"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 計算位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 應用正弦和餘弦函數
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次維度並註冊為緩衝區
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
        """
        try:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
        except Exception as e:
            logger.error(f"位置編碼失敗: {str(e)}")
            logger.error(f"輸入張量形狀: {x.shape}")
            logger.error(f"位置編碼形狀: {self.pe.shape}")
            raise

class ProjectionLayer(nn.Module):
    """投影層，用於調整特徵維度"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.projection(x)
        except Exception as e:
            logger.error(f"特徵投影失敗: {str(e)}")
            logger.error(f"輸入張量形狀: {x.shape}")
            raise

class DeviceAwareModule(nn.Module):
    """提供設備管理功能的基礎模組"""
    def __init__(self, device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """安全地將張量轉移到指定設備"""
        try:
            return tensor.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.warning(f"顯存不足，嘗試清理快取後重試")
                return tensor.to(self.device)
            raise

class EnhancedAnomalyDetector(DeviceAwareModule):
    """增強型異常檢測模型。

    整合了 LSTM 和 Transformer 的深度學習模型，用於時序異常檢測。
    
    Args:
        config (Config): 模型配置對象，包含所有超參數
        input_dim (int): 輸入特徵維度
        
    Attributes:
        projection (ProjectionLayer): 特徵投影層
        pos_encoder (PositionalEncoding): 位置編碼層
        lstm (nn.LSTM): LSTM 層
        transformer_layers (nn.ModuleList): Transformer 編碼層列表
        attention_weights (list): 存儲注意力權重的列表
    """
    def __init__(self, config: Config, input_dim: int):
        super().__init__(device=config.device)
        
        # 特徵投影
        self.projection = ProjectionLayer(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(
            d_model=config.hidden_dim,
            dropout=config.dropout
        )
        
        # LSTM 模塊
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        
        # Transformer 模塊
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=2*config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=4*config.hidden_dim,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.transformer_layers)
        ])
        
        # 多任務輸出頭
        self.anomaly_head = nn.Sequential(
            nn.LayerNorm(2*config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(2*config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.attention_weights = []  # 用於存儲注意力權重
        
        # 初始化參數
        self._init_weights()
        
    def _init_weights(self):
        """改進的參數初始化邏輯"""
        for name, param in self.named_parameters():
            if len(param.shape) > 1:  # 只對2維以上的參數使用特定初始化方法
                if 'lstm' in name:
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)
                elif ('transformer' in name) or ('projection' in name):  # 修正運算符
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                    else:
                        nn.init.zeros_(param)
                elif 'linear' in name:
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.zeros_(param)
            else:  # 對於1維參數（如bias），使用零初始化
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向計算。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, seq_len, features]  or 
                            [batch_size, features]

        Returns:
            torch.Tensor: 異常分數，形狀為 [batch_size]
            
        Raises:
            RuntimeError: 當計算過程出現錯誤時
        """
        try:
            # 確保輸入張量在正確的設備上
            x = self.to_device(x)
            
            self.attention_weights = []  # 重置注意力權重
            
            # 特徵投影
            x = self.projection(x)
            
            # 修正輸入維度 [batch_size, features] -> [batch_size, 1, features]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # 應用位置編碼
            x = self.pos_encoder(x)
                
            # LSTM 處理
            lstm_out, _ = self.lstm(x)
            
            # Transformer 處理
            transformer_out = lstm_out
            for layer in self.transformer_layers:
                transformer_out = layer(transformer_out)
                # 獲取並存儲注意力權重
                if hasattr(layer, 'attention_weights'):
                    self.attention_weights.append(layer.attention_weights.detach())
            
            # 異常檢測輸出
            output = self.anomaly_head(transformer_out[:, -1, :])
            return output.squeeze(-1)
            
        except Exception as e:
            logger.error(f"前向傳播失敗: {str(e)}")
            logger.error(f"輸入張量形狀: {x.shape if x is not None else 'None'}")
            logger.error(f"異常堆疊: {traceback.format_exc()}")
            raise

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """自定義 Transformer 編碼層，支持注意力權重獲取"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 多頭注意力
        src2, self.attention_weights = self.self_attn(src, src, src, 
                                                     attn_mask=src_mask,
                                                     key_padding_mask=src_key_padding_mask,
                                                     need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前饋網絡
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def _get_attention_weights(self):
        return self.attention_weights

# ========================
# 4. 優化訓練引擎
# ========================
class TrainingError(Exception):
    """訓練過程中的錯誤"""
    pass

class MetricsError(Exception):
    """指標計算過程中的錯誤"""
    pass

class TrainingEngine:
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 訓練歷史記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 早停機制
        self.best_val_loss = float('inf')
        self.best_auc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
        # 優化器 & 學習率調度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 混合精度
        self.scaler = GradScaler('cuda', enabled=config.mixed_precision)
        self.autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 添加訓練狀態追蹤
        self.training_state = {
            'last_successful_epoch': -1,
            'recoverable_errors': 0,
            'max_recoverable_errors': 3,
            'train_time': 0.0
        }
    
    def _try_recover_training(self, error: Exception) -> bool:
        """嘗試從訓練錯誤中恢復
        
        Returns:
            bool: 是否成功恢復
        """
        if self.training_state['recoverable_errors'] >= self.training_state['max_recoverable_errors']:
            logger.error("超過最大可恢復錯誤次數，停止訓練")
            return False
            
        try:
            logger.warning(f"嘗試從錯誤中恢復: {str(error)}")
            self.training_state['recoverable_errors'] += 1
            
            # 如果有最佳模型狀態，嘗試回退
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state['model_state_dict'])
                self.optimizer.load_state_dict(self.best_model_state['optimizer_state_dict'])
                logger.info("成功回退到最佳模型狀態")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"恢復失敗: {str(e)}")
            return False
    
    def _safe_forward_pass(self, inputs, targets):
        """安全的前向傳播"""
        try:
            outputs = self.model(inputs).squeeze()
            loss = self._focal_loss(outputs, targets)
            return outputs, loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 嘗試減少批次大小
                if inputs.shape[0] > 1:
                    logger.warning("顯存不足，嘗試減少批次大小")
                    mid = inputs.shape[0] // 2
                    outputs1, loss1 = self._safe_forward_pass(inputs[:mid], targets[:mid])
                    outputs2, loss2 = self._safe_forward_pass(inputs[mid:], targets[mid:])
                    return torch.cat([outputs1, outputs2]), (loss1 + loss2) / 2
            raise TrainingError(f"前向傳播失敗: {str(e)}")
    
    def _validate_batch(self, inputs, targets):
        """驗證輸入批次的有效性"""
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            raise ValueError("輸入包含 NaN  or Inf")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("目標包含 NaN  or Inf")
    
    def train_epoch(self):
        """執行單個訓練週期，包含錯誤處理"""
        self.model.train()
        total_loss = 0.0
        accum_steps = 0
        
        try:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                try:
                    # 驗證輸入
                    self._validate_batch(inputs, targets)
                    
                    inputs = inputs.to(self.config.device, non_blocking=True)
                    targets = targets.to(self.config.device, non_blocking=True).float()
                    
                    # 混合精度前向傳播
                    with autocast(self.autocast_device, enabled=self.config.mixed_precision):
                        outputs, loss = self._safe_forward_pass(inputs, targets)
                        loss = loss / self.config.gradient_accumulation
                    
                    # 反向傳播
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        # 檢查梯度是否有效
                        if self._check_gradients():
                            self.scaler.unscale_(self.optimizer)
                            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_steps += 1
                        total_loss += loss.item() * self.config.gradient_accumulation
                        
                except Exception as e:
                    logger.error(f"批次 {batch_idx} 處理失敗: {str(e)}")
                    if not self._try_recover_training(e):
                        raise
        
        except Exception as e:
            raise TrainingError(f"訓練週期失敗: {str(e)}")
        
        return total_loss / (accum_steps + 1e-7)
    
    def _check_gradients(self) -> bool:
        """檢查梯度是否有效"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"參數 {name} 的梯度包含 NaN  or Inf")
                    return False
        return True
    
    def _focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0):
        """Focal Loss 用於處理不平衡數據"""
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def validate(self):
        """執行驗證階段，並計算評估指標"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.config.device, non_blocking=True)
                targets = targets.to(self.config.device, non_blocking=True).float()

                # TTA (Test-Time Augmentation)
                num_aug = 5
                preds = []
                for _ in range(num_aug):
                    outputs = self.model(self._augment_features(inputs)).squeeze()
                    preds.append(torch.sigmoid(outputs))

                # 平均化 TTA 結果
                outputs = torch.stack(preds).mean(0)
                loss = self._focal_loss(outputs.log(), targets)
                val_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        # 計算評估指標
        metrics = self._calculate_metrics(np.array(all_targets), np.array(all_preds))
        self.scheduler.step(metrics['auc'])
        
        return val_loss / len(self.val_loader), metrics

    def _safe_metric_calculation(self, y_true, y_pred, metric_fn, default_value=0.0):
        """安全的指標計算
        
        Args:
            y_true: 真實標籤
            y_pred: 預測值
            metric_fn: 指標計算函數
            default_value: 計算失敗時的默認值
            
        Returns:
            float: 計算出的指標值 or默認值
        """
        try:
            return metric_fn(y_true, y_pred)
        except Exception as e:
            logger.warning(f"指標計算失敗: {str(e)}，使用默認值 {default_value}")
            return default_value

    def _calculate_metrics(self, y_true, y_pred):
        """計算評估指標，添加錯誤處理"""
        try:
            # 確保輸入數據有效
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("空的預測結果 or標籤")
                
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                raise ValueError("預測值包含 NaN  or Inf")
                
            # 二元分類閾值
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            # 安全的指標計算
            metrics = {
                "auc": self._safe_metric_calculation(
                    y_true, y_pred, roc_auc_score, default_value=0.5
                ),
                "accuracy": self._safe_metric_calculation(
                    y_true, y_pred_bin, accuracy_score, default_value=0.0
                ),
                "f1": self._safe_metric_calculation(
                    y_true, y_pred_bin, 
                    lambda yt, yp: f1_score(yt, yp, zero_division=1),
                    default_value=0.0
                ),
                "precision": self._safe_metric_calculation(
                    y_true, y_pred_bin, 
                    lambda yt, yp: precision_score(yt, yp, zero_division=1),
                    default_value=0.0
                ),
                "recall": self._safe_metric_calculation(
                    y_true, y_pred_bin, 
                    lambda yt, yp: recall_score(yt, yp, zero_division=1),
                    default_value=0.0
                )
            }
            
            # 驗證指標有效性
            for name, value in metrics.items():
                if not (0.0 <= value <= 1.0):
                    logger.warning(f"指標 {name} 超出有效範圍: {value}")
                    metrics[name] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"指標計算完全失敗: {str(e)}")
            return {
                "auc": 0.5,
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }

    def _augment_features(self, x, noise_scale=0.1):
        """為測試時的增強生成輕微擾動"""
        return x + torch.randn_like(x) * noise_scale

    def _update_history(self, train_loss: float, val_loss: float, val_metrics: dict):
        """更新訓練歷史記錄"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['auc'].append(val_metrics['auc'])
        self.history['precision'].append(val_metrics['precision'])
        self.history['recall'].append(val_metrics['recall'])
        self.history['f1'].append(val_metrics['f1'])

    def train(self):
        """完整的訓練過程，包含錯誤處理和恢復機制"""
        total_start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                epoch_start_time = time.time()
                
                try:
                    # 訓練 & 驗證
                    train_loss = self.train_epoch()
                    val_loss, val_metrics = self.validate()
                    
                    # 更新訓練狀態
                    self.training_state['last_successful_epoch'] = epoch
                    
                    # 更新歷史記錄
                    self._update_history(train_loss, val_loss, val_metrics)
                    
                    # 早停 & 保存模型
                    improved = False
                    if val_metrics['auc'] > self.best_auc:
                        self.best_auc = val_metrics['auc']
                        improved = True
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        improved = True
                    
                    if improved:
                        self.patience_counter = 0
                        self.best_model_state = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_val_loss': self.best_val_loss,
                            'best_auc': self.best_auc,
                            'training_state': self.training_state
                        }
                        torch.save(self.best_model_state, 'best_model.pth')
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stop:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            self.model.load_state_dict(self.best_model_state['model_state_dict'])
                            break
                    
                    # 更新學習率
                    self.scheduler.step(val_metrics['auc'])
                    
                    # 輸出進度
                    elapsed = time.time() - epoch_start_time
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs} | "
                        f"Time: {elapsed:.2f}s | "
                        f"LR: {current_lr:.2e} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"AUC: {val_metrics['auc']:.4f} | "
                        f"F1: {val_metrics['f1']:.4f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Epoch {epoch+1} 執行失敗: {str(e)}")
                    if not self._try_recover_training(e):
                        raise
            
            # 訓練完成，記錄總時間
            self.training_state['train_time'] = time.time() - total_start_time
            logger.info(f"訓練完成，總用時: {self.training_state['train_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"訓練過程失敗: {str(e)}")
            raise TrainingError(f"訓練失敗: {str(e)}")
        
        finally:
            # 保存最終訓練狀態
            if self.best_model_state is not None:
                torch.save(self.best_model_state, 'final_model.pth')

# ========================
# 5. 改進的可視化系統
# ========================
class VisualizationSystem:
    """模塊化可視化系統"""
    @staticmethod
    def setup_output_dir():
        """設置輸出目錄"""
        img_dir = Path('img')
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    @staticmethod
    def plot_training_loss(history: dict, save_path: str):
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

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        """繪製 ROC 曲線"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(15, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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

    @staticmethod
    def plot_auc_history(history: dict, save_path: str):
        """繪製驗證集上的 AUC 歷史"""
        plt.figure(figsize=(15, 6))
        plt.plot(history['auc'], label='Validation AUC')
        plt.title('AUC Score Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('AUC Score')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_attention(model: EnhancedAnomalyDetector, save_path: str):
        """改進的注意力權重可視化"""
        try:
            if not model.attention_weights:
                logger.warning("沒有可用的注意力權重")
                return
                
            num_layers = len(model.attention_weights)
            fig, axes = plt.subplots(num_layers, 1, figsize=(15, 4*num_layers))
            if num_layers == 1:
                axes = [axes]
            
            for idx, attn_weights in enumerate(model.attention_weights):
                if isinstance(attn_weights, torch.Tensor):
                    attn_weights = attn_weights.cpu().detach().numpy()
                
                if len(attn_weights.shape) == 4:
                    attn_weights = attn_weights.mean(axis=(0, 1))
                
                im = axes[idx].imshow(attn_weights, cmap='viridis', aspect='auto')
                axes[idx].set_title(f'Layer {idx+1} Attention Weights')
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.ax.set_title('Weight')
                
                # 只在有實際圖例內容時才添加圖例
                handles = []
                labels = []
                if handles and labels:  # 只在有內容時添加圖例
                    axes[idx].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
                
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"注意力權重可視化失敗: {str(e)}")
            logger.error(f"異常堆疊: {traceback.format_exc()}")

    @staticmethod
    def plot_metrics(history: dict, save_path: str):
        """繪製訓練指標"""
        try:
            fig, ax = plt.subplots(2, 2, figsize=(20, 12))  # 加寬圖形以容納圖例
            
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
            
            plt.tight_layout()  # 調整布局以確保圖例不重疊
            plt.savefig(save_path, bbox_inches='tight')  # 確保圖例完整保存
            plt.close()
        except Exception as e:
            logger.error(f"Metrics visualization failed: {str(e)}")

def plot_feature_importance(importance_df: pd.DataFrame, save_path: str):
    """繪製特徵重要性圖表"""
    plt.figure(figsize=(15, 6))  # 加寬圖形以容納圖例
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance Score')
    plt.legend(['Feature Importance'], loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')  # 確保圖例完整保存
    plt.close()

def evaluate_and_visualize(model: nn.Module,
                          test_loader: DataLoader,
                          engine: TrainingEngine,
                          device: str) -> Dict[str, float]:
    """評估模型性能並生成視覺化結果。

    Args:
        model (nn.Module): 待評估的模型
        test_loader (DataLoader): 測試數據加載器
        engine (TrainingEngine): 訓練引擎實例
        device (str): 計算設備名稱

    Returns:
        Dict[str, float]: 評估指標字典，包含：
            - auc: ROC曲線下面積
            - accuracy: 準確率
            - f1: F1分數
            - precision: 精確率
            - recall: 召回率
            
    Side Effects:
        - 在 img/ 目錄下生成以下圖表：
            - training_loss.png: 訓練損失曲線
            - roc_curve.png: ROC曲線
            - auc_history.png: AUC變化曲線
            - all_metrics.png: 綜合指標圖表
    """
    # 設置輸出目錄
    img_dir = VisualizationSystem.setup_output_dir()
    
    # 收集預測結果
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # 轉換為numpy數組
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 繪製損失曲線
    VisualizationSystem.plot_training_loss(
        engine.history,
        img_dir / 'training_loss.png'
    )
    
    # 繪製 ROC 曲線
    VisualizationSystem.plot_roc_curve(
        all_targets,
        all_preds,
        img_dir / 'roc_curve.png'
    )
    
    # 繪製 AUC 歷史
    VisualizationSystem.plot_auc_history(
        engine.history,
        img_dir / 'auc_history.png'
    )
    
    # 繪製其他指標
    VisualizationSystem.plot_metrics(
        engine.history,
        img_dir / 'all_metrics.png'
    )
    
    # 計算並返回評估指標
    metrics = {
        'auc': roc_auc_score(all_targets, all_preds),
        'accuracy': accuracy_score(all_targets, all_preds > 0.5),
        'f1': f1_score(all_targets, all_preds > 0.5),
        'precision': precision_score(all_targets, all_preds > 0.5),
        'recall': recall_score(all_targets, all_preds > 0.5)
    }
    
    return metrics

# ========================
# 主程式
# ========================
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

def save_dataset(datasets, save_path: str = 'datasets'):
    """保存生成的數據集
    
    Args:
        datasets: 包含 orders_df 和 external_df 的元組
        save_path: 數據集保存路徑
    """
    try:
        # 建立保存目錄
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 生成時間戳記
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 解包數據集
        orders_df, external_df = datasets
        
        # 保存訂單數據
        orders_file = save_path / f'orders_{timestamp}.csv'
        orders_df.to_csv(orders_file, index=False)
        logger.info(f'訂單數據已保存至: {orders_file}')
        
        # 保存外部數據
        external_file = save_path / f'external_{timestamp}.csv'
        external_df.to_csv(external_file, index=False)
        logger.info(f'外部數據已保存至: {external_file}')
        
        # 保存配置
        config_file = save_path / f'config_{timestamp}.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(DATA_CONFIG, f, indent=4)
        logger.info(f'配置檔案已保存至: {config_file}')
        
        return {
            'orders_path': str(orders_file),
            'external_path': str(external_file),
            'config_path': str(config_file)
        }
        
    except Exception as e:
        logger.error(f'保存數據集時發生錯誤: {str(e)}')
        raise

def analyze_feature_importance(model: DeviceAwareModule, 
                             X_test: np.ndarray, 
                             feature_names: List[str],
                             device: str) -> pd.DataFrame:
    """分析特徵重要性。

    使用排列重要性方法評估每個特徵對模型預測的影響程度。

    Args:
        model (DeviceAwareModule): 訓練好的模型
        X_test (np.ndarray): 測試數據
        feature_names (List[str]): 特徵名稱列表
        device (str): 計算設備名稱 ('cuda'  or 'cpu')

    Returns:
        pd.DataFrame: 包含特徵重要性分數的數據框，列包括：
            - feature: 特徵名稱
            - importance: 重要性分數
            
    Raises:
        ValueError: 當輸入數據格式不正確時
        RuntimeError: 當計算過程出錯時
    """
    try:
        model.eval()  # 確保模型在評估模式
        importances = []
        
        # 使用模型的 to_device 方法
        X_tensor = model.to_device(torch.FloatTensor(X_test))
        
        # 獲取基準預測
        with torch.no_grad():
            base_pred = model(X_tensor)
            if isinstance(base_pred, tuple):
                base_pred = base_pred[0]
        
        for i in range(len(feature_names)):
            X_permuted = X_test.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # 使用模型的 to_device 方法
            X_permuted_tensor = model.to_device(torch.FloatTensor(X_permuted))
            
            with torch.no_grad():
                permuted_pred = model(X_permuted_tensor)
                if isinstance(permuted_pred, tuple):
                    permuted_pred = permuted_pred[0]
                
            # 計算特徵重要性（基於預測變化）
            importance = torch.abs(base_pred - permuted_pred).mean().item()
            importances.append(importance)
        
        # 創建特徵重要性 DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"特徵重要性分析失敗: {str(e)}")
        logger.error(f"輸入形狀: X_test - {X_test.shape}")
        logger.error(f"特徵數量: {len(feature_names)}")
        logger.error(f"異常堆疊: {traceback.format_exc()}")
        raise

def plot_feature_importance(importance_df: pd.DataFrame, save_path: str):
    """繪製特徵重要性圖表"""
    plt.figure(figsize=(15, 6))  # 加寬圖形以容納圖例
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance Score')
    plt.legend(['Feature Importance'], loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')  # 確保圖例完整保存
    plt.close()

def save_model(model: nn.Module, 
              engine: TrainingEngine, 
              config: Config, 
              test_metrics: Dict[str, float],
              feature_columns: List[str]):
    """保存模型、配置和評估結果"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'feature_columns': feature_columns,
        'test_metrics': test_metrics,
        'training_history': engine.history,
        'best_val_loss': engine.best_val_loss,
        'best_auc': engine.best_auc
    }
    
    # 保存完整模型檔案
    torch.save(save_dict, 'anomaly_detector.pth')
    
    # 保存模型分析報告
    report = {
        'model_summary': str(model),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'config': vars(config),
        'feature_columns': feature_columns,
        'test_metrics': test_metrics,
        'training_history': {
            'final_train_loss': engine.history['train_loss'][-1],
            'final_val_loss': engine.history['val_loss'][-1],
            'best_val_loss': engine.best_val_loss,
            'best_auc': engine.best_auc
        }
    }
    
    with open('model_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    logger.info(f"模型和分析報告已保存")

def safe_tensor_conversion(data: Union[np.ndarray, pd.Series], dtype=torch.float32) -> torch.Tensor:
    """安全地將各種數據格式轉換為 PyTorch 張量。

    支持 numpy 數組和 pandas Series 的轉換，並提供完善的錯誤處理。

    Args:
        data (Union[np.ndarray, pd.Series]): 要轉換的數據
        dtype (torch.dtype, optional): 目標數據類型。默認為 torch.float32

    Returns:
        torch.Tensor: 轉換後的張量

    Raises:
        TypeError: 當輸入數據類型不支持時
        ValueError: 當數據包含無效值時
    """
    try:
        if isinstance(data, pd.Series):
            data = data.values
        return torch.tensor(data, dtype=dtype)
    except Exception as e:
        logger.error(f"張量轉換失敗: {str(e)}")
        logger.error(f"數據類型: {type(data)}")
        logger.error(f"數據形狀: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        raise

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, config):
    """創建數據加載器，包含錯誤處理
    
    Args:
        X_train, X_val, X_test: 訓練、驗證和測試特徵
        y_train, y_val, y_test: 訓練、驗證和測試標籤
        config: 配置對象
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        # 創建數據集
        train_dataset = TensorDataset(
            safe_tensor_conversion(X_train),
            safe_tensor_conversion(y_train)
        )
        val_dataset = TensorDataset(
            safe_tensor_conversion(X_val),
            safe_tensor_conversion(y_val)
        )
        test_dataset = TensorDataset(
            safe_tensor_conversion(X_test),
            safe_tensor_conversion(y_test)
        )
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"創建數據加載器失敗: {str(e)}")
        logger.error(f"數據形狀:")
        logger.error(f"X_train: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")
        logger.error(f"y_train: {y_train.shape if hasattr(y_train, 'shape') else 'unknown'}")
        raise

def main():
    # Initialize DATA_CONFIG as a global variable
    global DATA_CONFIG
    DATA_CONFIG = {}
    
    args = parse_arguments()
    config = Config.from_args(args)
    
    # 設置隨機種子和設備
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
    
    # 生成數據集
    datasets = generate_datasets(DATA_CONFIG)
    
    # 保存數據集到指定路徑
    save_paths = save_dataset(datasets, args.save_path)
    logger.info(f"數據集已保存到: {save_paths}")

    # 改進的數據預處理
    orders_df, external_df = datasets
    
    # 1. 時間特徵工程
    orders_df["order_time"] = pd.to_datetime(orders_df["order_time"])
    orders_df["hour"] = orders_df["order_time"].dt.hour
    orders_df["day_of_week"] = orders_df["order_time"].dt.dayofweek
    orders_df["day_of_month"] = orders_df["order_time"].dt.day
    orders_df["month"] = orders_df["order_time"].dt.month
    
    # 2. 外部數據處理和合併
    external_df["record_date"] = pd.to_datetime(external_df["record_date"])
    orders_df = orders_df.merge(external_df, left_on="order_time", right_on="record_date", how="left")
    
    # 3. 特徵工程
    # 計算訂單金額與碳排放量的比率
    orders_df["amount_carbon_ratio"] = orders_df["order_amount"] / (orders_df["carbon_amount"] + 1e-6)
    
    # 計算移動平均和標準差
    window_sizes = [7, 30]
    for window in window_sizes:
        orders_df[f"amount_ma_{window}"] = orders_df.groupby("day_of_week")["order_amount"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        orders_df[f"amount_std_{window}"] = orders_df.groupby("day_of_week")["order_amount"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # 4. 特徵選擇
    feature_columns = [
        "order_amount", "carbon_amount", "amount_carbon_ratio",
        "hour", "day_of_week", "day_of_month", "month",
        "carbon_emission", "GDP", "renewable_energy", "carbon_price", "carbon_trade_vol"
    ] + [f"amount_ma_{w}" for w in window_sizes] + [f"amount_std_{w}" for w in window_sizes]
    
    # 5. 特徵預處理
    X = orders_df[feature_columns].copy()
    y = orders_df["is_anomalous"]
    
    # 處理缺失值
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    # 特徵縮放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 保存特徵縮放器以供將來使用
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # 6. 數據集分割
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=config.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=config.seed, stratify=y_val_test
    )
    
    # 7. 數據加載器創建
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        config
    )
    
    # 8. 模型初始化和訓練
    model = EnhancedAnomalyDetector(config, input_dim=X_scaled.shape[1])
    logger.info(f"模型總參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練引擎
    engine = TrainingEngine(config, model, train_loader, val_loader)
    engine.train()
    
    # 9. 評估和可視化
    # 在測試集上進行評估
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            test_predictions.extend(probs.cpu().numpy())
            test_targets.extend(targets.numpy())
    
    # 計算並打印測試集指標
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_metrics = {
        'auc': roc_auc_score(test_targets, test_predictions),
        'accuracy': accuracy_score(test_targets, test_predictions > 0.5),
        'f1': f1_score(test_targets, test_predictions > 0.5),
        'precision': precision_score(test_targets, test_predictions > 0.5),
        'recall': recall_score(test_targets, test_predictions > 0.5)
    }
    
    logger.info("測試集評估結果:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name.upper()}: {value:.4f}")
    
    # 繪製訓練指標
    VisualizationSystem.plot_metrics(engine.history, 'training_metrics.png')
    
    # 繪製注意力權重
    if hasattr(model, 'attention_weights') and model.attention_weights:
        VisualizationSystem.plot_attention(model, 'attention_weights.png')
    
    # 特徵重要性分析
    try:
        feature_importance = analyze_feature_importance(
            model=model,
            X_test=X_test,
            feature_names=feature_columns,
            device=config.device  # 明確傳入設備參數
        )
        plot_feature_importance(feature_importance, 'feature_importance.png')
        logger.info("特徵重要性分析完成")
    except Exception as e:
        logger.error(f"特徵重要性分析失敗: {str(e)}")
        logger.warning("跳過特徵重要性分析和可視化")
    
    # 保存模型和配置
    save_model(model, engine, config, test_metrics, feature_columns)

def test_visualization_setup():
    """測試視覺化設置"""
    try:
        # 測試基本繪圖功能
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('測試圖表')
        plt.close(fig)
        logger.info("基本繪圖測試成功")
        
        # 測試中文顯示
        fig, ax = plt.subplots()
        ax.set_title('中文測試')
        plt.close(fig)
        logger.info("中文顯示測試成功")
        
        # 測試樣式設置
        logger.info(f"當前可用的樣式: {plt.style.available}")
        
    except Exception as e:
        logger.error(f"視覺化測試失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_data_conversion():
    """測試數據轉換功能"""
    try:
        # 測試 numpy array 轉換
        x = np.random.randn(10, 5)
        tensor = safe_tensor_conversion(x)
        logger.info(f"NumPy array 轉換成功: {tensor.shape}")
        
        # 測試 pandas Series 轉換
        s = pd.Series(np.random.randn(10))
        tensor = safe_tensor_conversion(s)
        logger.info(f"Pandas Series 轉換成功: {tensor.shape}")
        
        # 測試 pandas DataFrame 轉換
        df = pd.DataFrame(np.random.randn(10, 5))
        tensor = safe_tensor_conversion(df.values)
        logger.info(f"Pandas DataFrame 轉換成功: {tensor.shape}")
        
    except Exception as e:
        logger.error(f"數據轉換測試失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def initialize_system():
    """系統初始化和測試"""
    logger.info("開始系統初始化...")
    
    # 測試視覺化設置
    test_visualization_setup()
    
    # 測試數據轉換
    test_data_conversion()
    
    logger.info("系統初始化完成")

class PerformanceMonitor:
    """性能監控類"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.current_metrics = PerformanceMetrics()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """獲取系統性能指標"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'swap_percent': psutil.swap_memory().percent
        }
        
        # GPU 指標（如果可用）
        if HAS_GPU_UTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一個 GPU
                    metrics.update({
                        'gpu_load': gpu.load * 100,
                        'gpu_memory_used': gpu.memoryUsed,
                        'gpu_memory_total': gpu.memoryTotal,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"獲取 GPU 指標失敗: {str(e)}")
                
        return metrics

if __name__ == "__main__":
    try:
        # 系統初始化
        initialize_system()
        
        # 執行主程式
        main()
        
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise

