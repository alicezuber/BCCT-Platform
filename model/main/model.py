import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging
from utils import logger

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

class EnhancedAnomalyDetector(DeviceAwareModule):
    """增強型異常檢測模型"""
    def __init__(self, config, input_dim: int):
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
                elif ('transformer' in name) or ('projection' in name):
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
        """前向計算"""
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
            raise