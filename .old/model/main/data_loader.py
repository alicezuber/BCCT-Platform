import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from datetime import datetime
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Union
from utils import logger, safe_data_processing

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
        """使用 GMM 生成異常訂單"""
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
        
        return orders_df, external_df

@safe_data_processing
def save_dataset(datasets, save_path: str = 'datasets'):
    """保存生成的數據集"""
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
        json.dump({}, f, indent=4)  # 替換為實際配置
    logger.info(f'配置檔案已保存至: {config_file}')
    
    return {
        'orders_path': str(orders_file),
        'external_path': str(external_file),
        'config_path': str(config_file)
    }

class AnomalyDataset:
    """異常檢測數據集處理類"""
    def __init__(self, orders_df: pd.DataFrame, external_df: pd.DataFrame):
        self.orders_df = orders_df
        self.external_df = external_df
        self.scaler = StandardScaler()
        
    def preprocess(self):
        """數據預處理"""
        # 時間特徵工程
        self.orders_df["order_time"] = pd.to_datetime(self.orders_df["order_time"])
        self.orders_df["hour"] = self.orders_df["order_time"].dt.hour
        self.orders_df["day_of_week"] = self.orders_df["order_time"].dt.dayofweek
        self.orders_df["day_of_month"] = self.orders_df["order_time"].dt.day
        self.orders_df["month"] = self.orders_df["order_time"].dt.month
        
        # 合併外部數據
        self.external_df["record_date"] = pd.to_datetime(self.external_df["record_date"])
        merged_df = self.orders_df.merge(
            self.external_df,
            left_on="order_time",
            right_on="record_date",
            how="left"
        )
        
        # 特徵工程
        merged_df["amount_carbon_ratio"] = merged_df["order_amount"] / (merged_df["carbon_amount"] + 1e-6)
        
        # 計算移動平均和標準差
        window_sizes = [7, 30]
        for window in window_sizes:
            merged_df[f"amount_ma_{window}"] = merged_df.groupby("day_of_week")["order_amount"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            merged_df[f"amount_std_{window}"] = merged_df.groupby("day_of_week")["order_amount"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # 選擇特徵
        self.feature_columns = [
            "order_amount", "carbon_amount", "amount_carbon_ratio",
            "hour", "day_of_week", "day_of_month", "month",
            "carbon_emission", "GDP", "renewable_energy", "carbon_price", "carbon_trade_vol"
        ] + [f"amount_ma_{w}" for w in window_sizes] + [f"amount_std_{w}" for w in window_sizes]
        
        # 處理缺失值
        X = merged_df[self.feature_columns].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # 特徵縮放
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = merged_df["is_anomalous"].values
        
        # 保存特徵縮放器
        joblib.dump(self.scaler, 'feature_scaler.joblib')
        
        return self.X_scaled, self.y, self.feature_columns
    
    def create_data_loaders(self, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """創建數據加載器"""
        from sklearn.model_selection import train_test_split
        
        # 分割數據集
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            self.X_scaled, self.y,
            test_size=0.3,
            random_state=config.seed,
            stratify=self.y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=0.5,
            random_state=config.seed,
            stratify=y_val_test
        )
        
        # 創建數據加載器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
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