import logging
import traceback
import functools
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import psutil

try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False

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
        
        # 配置控制台處理器 - 設置為 INFO 以上級別
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
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()

# 設置日誌系統
logger = setup_logging()

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

class TrainingError(Exception):
    """訓練過程中的錯誤"""
    pass

class MetricsError(Exception):
    """指標計算過程中的錯誤"""
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

def setup_matplotlib_style():
    """配置 matplotlib 的顯示樣式和參數"""
    try:
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.color': '#cccccc',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'lines.markeredgewidth': 1,
            'scatter.marker': '.',
            'scatter.edgecolors': 'none',
            'axes.prop_cycle': plt.cycler(color=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ])
        })
        
        # 設置中文字體
        font_families = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'Arial', 'DejaVu Sans']
        for font in font_families:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
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

class PerformanceMonitor:
    """性能監控類"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = []
        
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

def safe_tensor_conversion(data: Any, dtype=torch.float32) -> torch.Tensor:
    """安全地將各種數據格式轉換為 PyTorch 張量"""
    try:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)
    except Exception as e:
        logger.error(f"張量轉換失敗: {str(e)}")
        logger.error(f"數據類型: {type(data)}")
        logger.error(f"數據形狀: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        raise