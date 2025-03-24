import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import time
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
import traceback

from utils import logger, TrainingError

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
        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 添加訓練狀態追蹤
        self.training_state = {
            'last_successful_epoch': -1,
            'recoverable_errors': 0,
            'max_recoverable_errors': 3,
            'train_time': 0.0
        }
    
    def _try_recover_training(self, error: Exception) -> bool:
        """嘗試從訓練錯誤中恢復"""
        if self.training_state['recoverable_errors'] >= self.training_state['max_recoverable_errors']:
            logger.error("超過最大可恢復錯誤次數，停止訓練")
            return False
            
        try:
            logger.warning(f"嘗試從錯誤中恢復: {str(error)}")
            self.training_state['recoverable_errors'] += 1
            
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
            raise ValueError("輸入包含 NaN 或 Inf")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            raise ValueError("目標包含 NaN 或 Inf")
    
    def train_epoch(self):
        """執行單個訓練週期"""
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
                    with autocast(device_type=self.autocast_device, enabled=self.config.mixed_precision):
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
                    logger.warning(f"參數 {name} 的梯度包含 NaN 或 Inf")
                    return False
        return True
    
    def _focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0):
        """Focal Loss 用於處理不平衡數據"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def validate(self):
        """執行驗證階段"""
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
        metrics = self._calculate_metrics(
            torch.tensor(all_targets),
            torch.tensor(all_preds)
        )
        return val_loss / len(self.val_loader), metrics

    def _augment_features(self, x, noise_scale=0.1):
        """為測試時的增強生成輕微擾動"""
        return x + torch.randn_like(x) * noise_scale

    def _calculate_metrics(self, y_true, y_pred):
        """計算評估指標"""
        from sklearn.metrics import roc_auc_score, accuracy_score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        try:
            # 確保輸入數據有效
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("空的預測結果或標籤")
                
            y_pred_bin = (y_pred > 0.5).float()
            
            metrics = {
                "auc": roc_auc_score(y_true.cpu(), y_pred.cpu()),
                "accuracy": accuracy_score(y_true.cpu(), y_pred_bin.cpu()),
                "f1": f1_score(y_true.cpu(), y_pred_bin.cpu()),
                "precision": precision_score(y_true.cpu(), y_pred_bin.cpu()),
                "recall": recall_score(y_true.cpu(), y_pred_bin.cpu())
            }
            
            # 驗證指標有效性
            for name, value in metrics.items():
                if not (0.0 <= value <= 1.0):
                    logger.warning(f"指標 {name} 超出有效範圍: {value}")
                    metrics[name] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"指標計算失敗: {str(e)}")
            return {
                "auc": 0.5,
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }

    def train(self):
        """完整的訓練過程"""
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

    def _update_history(self, train_loss: float, val_loss: float, val_metrics: dict):
        """更新訓練歷史記錄"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['auc'].append(val_metrics['auc'])
        self.history['precision'].append(val_metrics['precision'])
        self.history['recall'].append(val_metrics['recall'])
        self.history['f1'].append(val_metrics['f1'])