import pandas as pd
import numpy as np
# import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
# from sklearn.linear_model import LinearRegression


def impute_missing_values_1(data, target_columns, estimation_features, n_neighbors=5, z_threshold=3, min_features=3):
    # 确保 target_columns 是列表
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # 检查哪些目标列实际存在于数据中
    available_target_columns = [col for col in target_columns if col in data.columns]
    if not available_target_columns:
        raise ValueError(f"None of the specified target columns {target_columns} are present in the data. Available columns are: {data.columns.tolist()}")
    
    print(f"Available target columns: {available_target_columns}")
    print(f"Missing target columns: {set(target_columns) - set(available_target_columns)}")

    # 检查哪些估算特征实际存在于数据中
    available_estimation_features = [col for col in estimation_features if col in data.columns]
    if not available_estimation_features:
        raise ValueError(f"None of the specified estimation features are present in the data. Available columns are: {data.columns.tolist()}")
    
    print(f"Available estimation features: {available_estimation_features}")
    print(f"Missing estimation features: {set(estimation_features) - set(available_estimation_features)}")

    imputation_stats = {col: {
        'initial_missing': 0,
        'final_missing': 0,
        'filled_values': 0,
        'outliers_detected': 0,
        'outliers_replaced': 0,
        'small_values_adjusted': 0
    } for col in available_target_columns}

    try:
        data = data.copy()

        # 对分类变量进行独热编码
        categorical_features = data[available_estimation_features].select_dtypes(include=['object', 'category']).columns
        if not categorical_features.empty:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            available_estimation_features = list(set(available_estimation_features) - set(categorical_features)) + list(encoded_feature_names)

        for target_column in available_target_columns:
            print(f"\nProcessing {target_column}")
            imputation_stats[target_column]['initial_missing'] = data[target_column].isnull().sum()
            print(f"Initial missing {target_column} values: {imputation_stats[target_column]['initial_missing']}")

            if not pd.api.types.is_numeric_dtype(data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Skipping.")
                continue

            # 确定目标列的合理最小值
            valid_values = data[target_column].dropna()
            min_valid_value = valid_values.min()
            reasonable_min = max(min_valid_value * 0.5, 1)  # 使用有效值的一半或1，取较大者
            print(f"Determined reasonable minimum value for {target_column}: {reasonable_min}")

            # 特征选择
            numeric_features = data[available_estimation_features].select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_features) > 0:
                # 去除包含 NaN 的行用于计算互信息
                valid_data = data.dropna(subset=[target_column] + list(numeric_features))
                if len(valid_data) == 0:
                    print(f"Warning: No valid data for mutual information calculation for {target_column}. Using all numeric features.")
                    selected_features = list(numeric_features)
                else:
                    mi_scores = mutual_info_regression(valid_data[numeric_features], valid_data[target_column])
                    selected_features = [feature for feature, score in zip(numeric_features, mi_scores) if score > 0]
                selected_features = selected_features[:min_features] if len(selected_features) > min_features else selected_features
            else:
                selected_features = []
            
            selected_features.extend([f for f in available_estimation_features if f not in selected_features])
            print(f"Selected features for estimation: {selected_features}")

            # KNN估算
            impute_data = data[selected_features + [target_column]].copy()
            non_missing_count = impute_data[target_column].notna().sum()
            if non_missing_count > n_neighbors:
                imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                imputed_values = imputer.fit_transform(impute_data)
                
                mask = data[target_column].isnull()
                estimated_values = imputed_values[mask, -1]
                small_estimates = estimated_values < reasonable_min

                if small_estimates.any():
                    print(f"Warning: {small_estimates.sum()} values were estimated below {reasonable_min}. Adjusting these values.")
                    estimated_values[small_estimates] = reasonable_min
                    imputation_stats[target_column]['small_values_adjusted'] = small_estimates.sum()

                data.loc[mask, target_column] = estimated_values
            else:
                print(f"Not enough non-missing values for KNN imputation. Using median imputation.")
                data[target_column].fillna(max(data[target_column].median(), reasonable_min), inplace=True)

            if imputation_stats[target_column]['outliers_detected'] > 0:
                lower_bound = data[target_column].quantile(0.25)
                upper_bound = data[target_column].quantile(0.75)
                iqr = upper_bound - lower_bound
                smart_lower = max(lower_bound - 1.5 * iqr, reasonable_min)
                smart_upper = upper_bound + 1.5 * iqr
                
                imputation_stats[target_column]['outliers_replaced'] = imputation_stats[target_column]['outliers_detected']
                print(f"Replaced outliers with values between {smart_lower} and {smart_upper}")

            # 最终检查，确保没有不合理的小值
            small_values = data[target_column] < reasonable_min
            if small_values.any():
                print(f"Warning: {small_values.sum()} values are still below {reasonable_min}. Adjusting these values.")
                data.loc[small_values, target_column] = reasonable_min
                imputation_stats[target_column]['small_values_adjusted'] += small_values.sum()

            imputation_stats[target_column]['final_missing'] = data[target_column].isnull().sum()
            imputation_stats[target_column]['filled_values'] = imputation_stats[target_column]['initial_missing'] - imputation_stats[target_column]['final_missing']
            print(f"After processing, missing {target_column} values: {imputation_stats[target_column]['final_missing']}")
            print(f"Filled missing values: {imputation_stats[target_column]['filled_values']}")
            print(f"Small values adjusted: {imputation_stats[target_column]['small_values_adjusted']}")

            print(f"\n{target_column} statistics:")
            print(data[target_column].describe())

        return data, imputation_stats
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        for col in available_target_columns:
            imputation_stats[col]['error'] = str(e)
        return data, imputation_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImputationModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TorchScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, tensor: torch.Tensor):
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        self.std[self.std == 0] = 1  # 防止除零
        
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std
    
    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean
    
    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        self.fit(tensor)
        return self.transform(tensor)
    
    def state_dict(self) -> Dict:
        return {
            'mean': self.mean.cpu().numpy() if self.mean is not None else None,
            'std': self.std.cpu().numpy() if self.std is not None else None
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.mean = torch.tensor(state_dict['mean']) if state_dict['mean'] is not None else None
        self.std = torch.tensor(state_dict['std']) if state_dict['std'] is not None else None

def get_model_path(target_column: str, models_dir: str = 'imputation_models') -> Tuple[Path, Path, Path]:
    """获取模型相关文件的路径"""
    base_dir = Path(models_dir)
    base_dir.mkdir(exist_ok=True)
    
    model_path = base_dir / f"{target_column}_model.pth"
    config_path = base_dir / f"{target_column}_config.json"
    scaler_path = base_dir / f"{target_column}_scaler.pkl"
    
    return model_path, config_path, scaler_path

def save_model(model: ImputationModel,
               feature_scaler: TorchScaler,
               config: Dict,
               target_column: str,
               models_dir: str = 'imputation_models'):
    """保存模型、配置和scaler"""
    model_path, config_path, scaler_path = get_model_path(target_column, models_dir)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    
    # 保存配置
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # 保存scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(feature_scaler.state_dict(), f)
    
    print(f"Model and configurations saved for column: {target_column}")

def load_model(target_column: str, 
               input_dim: int, 
               models_dir: str = 'imputation_models',
               device: torch.device = None) -> Tuple[Optional[ImputationModel], Optional[Dict], Optional[TorchScaler]]:
    """加载模型、配置和scaler"""
    model_path, config_path, scaler_path = get_model_path(target_column, models_dir)
    
    if not all(p.exists() for p in [model_path, config_path, scaler_path]):
        return None, None, None
    
    try:
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载scaler
        with open(scaler_path, 'rb') as f:
            scaler_state = pickle.load(f)
        feature_scaler = TorchScaler()
        feature_scaler.load_state_dict(scaler_state)
        
        # 创建并加载模型
        model = ImputationModel(input_dim, config['output_dim']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"Model and configurations loaded for column: {target_column}")
        return model, config, feature_scaler
    
    except Exception as e:
        print(f"Error loading model for {target_column}: {str(e)}")
        return None, None, None

def preprocess_features(data: torch.Tensor) -> Tuple[torch.Tensor, TorchScaler]:
    """预处理特征：处理异常值、NaN值和标准化"""
    # 首先处理 NaN 值
    if torch.isnan(data).any():
        # 对每一列计算均值，忽略 NaN
        col_means = torch.nanmean(data, dim=0)
        # 使用 where 替换 NaN 值
        data = torch.where(torch.isnan(data), col_means.unsqueeze(0).expand(data.shape[0], -1), data)
    
    # 处理无穷值
    data = torch.where(torch.isinf(data), torch.full_like(data, 0), data)
    
    # 处理异常值
    Q1 = torch.quantile(data, 0.25, dim=0)
    Q3 = torch.quantile(data, 0.75, dim=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data = torch.clamp(data, lower_bound, upper_bound)
    
    scaler = TorchScaler()
    # 标准化
    data = scaler.fit_transform(data)
    
    return data, scaler

def impute_missing_values_2(data: pd.DataFrame,
                         target_column: str,
                         feature_columns: List[str],
                         model_params: Optional[Dict] = None,
                         test_size: float = 0.2,
                         device: Optional[torch.device] = None,
                         models_dir: str = 'imputation_models') -> Tuple[pd.DataFrame, Dict]:
    """
    使用PyTorch深度学习模型填补缺失值的优化版本，支持模型的保存和加载
    """
    if model_params is None:
        model_params = {
            'epochs': 25,
            'batch_size': 32,
            'learning_rate': 0.001,  # Increased learning rate
            'early_stopping_patience': 10,
            'weight_decay': 0.01,    # Increased regularization
            'gradient_clip': 0.5,    # Reduced gradient clipping
            'scheduler_factor': 0.3,
            'scheduler_patience': 3,
            'scheduler_min_lr': 1e-5
        }
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_filled = data.copy()
    stats = {}
    
    # 检查列是否存在
    all_columns = [target_column] + feature_columns
    missing_columns = [col for col in all_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in data: {missing_columns}")
    
    # 准备数据
    X_data = data[feature_columns].copy()
    y_data = data[target_column].copy()
    
    # 检查数据类型并转换
    for col in feature_columns:
        if X_data[col].dtype == object:
            print(f"Warning: Converting column {col} to numeric")
            X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
    
    if y_data.dtype == object:
        print(f"Warning: Converting target column {target_column} to numeric")
        y_data = pd.to_numeric(y_data, errors='coerce')
    
    # 将数据转换为张量，确保类型正确
    X = torch.tensor(X_data.values, dtype=torch.float32)
    y = torch.tensor(y_data.values, dtype=torch.float32)
    
    # 获取缺失值掩码
    mask = pd.isna(y_data)
    stats['initial_missing'] = mask.sum()
    
    if stats['initial_missing'] == 0:
        print("No missing values to impute")
        return data_filled, stats
    
    # 尝试加载已有模型
    loaded_model, config, feature_scaler = load_model(
        target_column, 
        input_dim=len(feature_columns),
        models_dir=models_dir,
        device=device
    )
    
    if loaded_model is not None:
        print("Using pre-trained model")
        model = loaded_model
        task = config['task']
        unique_values = torch.tensor(config['unique_values']) if 'unique_values' in config else None
        
        # 使用保存的scaler处理新数据
        X_processed = feature_scaler.transform(X)
    else:
        print("Training new model")
        # 预处理特征
        X_processed, feature_scaler = preprocess_features(X)
        
        # 准备训练数据
        train_mask = ~mask
        X_train = X_processed[train_mask]
        y_train = y[train_mask]
        
        # 移除训练数据中的任何 NaN 值
        valid_indices = ~torch.isnan(y_train)
        X_train = X_train[valid_indices]
        y_train = y_train[valid_indices]
        
        if len(y_train) == 0:
            raise ValueError("No valid training data available after removing NaN values")
        
        # 确定任务类型
        unique_values = torch.unique(y_train)
        task = 'classification' if len(unique_values) < 10 else 'regression'
        
        if task == 'classification':
            output_dim = len(unique_values)
            y_train = torch.searchsorted(unique_values, y_train)
        else:
            output_dim = 1
        
        # 划分训练集和验证集
        dataset_size = len(X_train)
        indices = torch.randperm(dataset_size)
        split = int(dataset_size * (1 - test_size))
        train_idx, val_idx = indices[:split], indices[split:]
        
        # 创建数据加载器
        train_dataset = CustomDataset(X_train[train_idx], y_train[train_idx])
        val_dataset = CustomDataset(X_train[val_idx], y_train[val_idx])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(model_params['batch_size'], len(train_dataset)),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(model_params['batch_size'], len(val_dataset)),
            shuffle=False
        )
        
        # 创建模型
        model = ImputationModel(X.shape[1], output_dim).to(device)
        
        # 定义损失函数和优化器
        criterion = (nn.CrossEntropyLoss() if task == 'classification'
                    else nn.MSELoss())
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_params['learning_rate'],
            weight_decay=model_params['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=model_params['scheduler_factor'],
            patience=model_params['scheduler_patience'],
            min_lr=model_params['scheduler_min_lr']
        )
        
        # 训练模型
        model, best_val_loss = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, model_params, task
        )
        
        # 保存模型和配置
        config = {
            'task': task,
            'output_dim': output_dim,
            'feature_columns': feature_columns,
            'unique_values': unique_values.tolist() if task == 'classification' else None,
            'best_val_loss': float(best_val_loss),
            'model_params': model_params
        }
        
        save_model(model, feature_scaler, config, target_column, models_dir)
    
    # 预测缺失值
    model.eval()
    with torch.no_grad():
        X_missing = X_processed[mask].to(device)
        if len(X_missing) > 0:  # 确保有缺失值需要预测
            predictions = model(X_missing)
            
            if task == 'classification':
                predictions = torch.argmax(predictions, dim=1)
                predictions = unique_values[predictions]
            else:
                predictions = predictions.squeeze()
            
            # 将预测值填回DataFrame
            data_filled.loc[mask, target_column] = predictions.cpu().numpy()
    
    stats['filled_values'] = stats['initial_missing']
    stats['final_missing'] = data_filled[target_column].isna().sum()
    print(f"Filled {stats['filled_values']} missing values")
    
    return data_filled, stats

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device,
                model_params: Dict,
                task: str) -> Tuple[nn.Module, float]:
    """训练模型并返回最佳模型"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(model_params['epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 检查并处理任何剩余的 NaN 值
            if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                continue
                
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if task == 'classification':
                loss = criterion(outputs, batch_y.long())
            else:
                loss = criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['gradient_clip'])
            optimizer.step()
            
            train_losses.append(loss.item())
        
        if not train_losses:
            print("Warning: No valid training batches in this epoch")
            continue
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                
                # 检查并处理任何剩余的 NaN 值
                if torch.isnan(val_X).any() or torch.isnan(val_y).any():
                    continue
                    
                outputs = model(val_X)
                
                if task == 'classification':
                    loss = criterion(outputs, val_y.long())
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    loss = criterion(outputs.squeeze(), val_y)
                    predictions = outputs.squeeze()
                
                val_losses.append(loss.item())
                val_predictions.extend(predictions.cpu())
                val_targets.extend(val_y.cpu())
        
        if not val_losses:
            print("Warning: No valid validation batches in this epoch")
            continue
            
        val_loss = torch.tensor(val_losses).mean()
        
        # 计算评估指标
        if task == 'regression':
            val_metric = torch.sqrt(torch.mean((torch.tensor(val_predictions) - 
                                              torch.tensor(val_targets)) ** 2))
            print(f"Epoch [{epoch+1}/{model_params['epochs']}], "
                  f"Train Loss: {np.mean(train_losses):.4f}, "
                  f"Val RMSE: {val_metric:.4f}")
        else:
            val_metric = (torch.tensor(val_predictions) == 
                         torch.tensor(val_targets)).float().mean()
            print(f"Epoch [{epoch+1}/{model_params['epochs']}], "
                  f"Train Loss: {np.mean(train_losses):.4f}, "
                  f"Val ACC: {val_metric:.4f}")
            val_metric = 1 - val_metric  # 转换为最小化问题
        
        if not torch.isnan(val_metric):
            scheduler.step(val_metric)
            
            # 保存最佳模型状态
            if val_metric < best_val_loss:
                best_val_loss = val_metric
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= model_params['early_stopping_patience']:
                    print("Early stopping triggered")
                    break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_loss


