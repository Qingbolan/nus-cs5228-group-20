import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy import stats
import math

class ModelConfig:
    def __init__(self, feature_columns, y_stats=None):
        self.feature_columns = feature_columns
        self.y_stats = y_stats if y_stats is not None else {}

class PowerMixtureModel(nn.Module):
    def __init__(self, input_dim, n_components=5):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.n_components = n_components
        
        # Enhanced feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Component networks
        self.component_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 2)
            ) for _ in range(n_components)
        ])
        
        # Mixture weights with temperature scaling
        self.weight_net = nn.Sequential(
            nn.Linear(128, n_components),
            nn.LogSoftmax(dim=1)
        )
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        means = [100, 150, 200, 300, 400]
        for i, net in enumerate(self.component_nets):
            final_layer = net[-1]
            mean_idx = torch.zeros_like(final_layer.weight[:1])
            mean_idx[0, -1] = math.log(means[i])
            final_layer.weight.data[:1] = mean_idx
            final_layer.bias.data[:1] = 0.0
            
            std_idx = torch.zeros_like(final_layer.weight[1:])
            std_idx[0, -1] = math.log(means[i] * 0.1)
            final_layer.weight.data[1:] = std_idx
            final_layer.bias.data[1:] = 0.0
    
    def forward(self, x):
        x = self.input_norm(x)
        features = self.feature_net(x)
        
        components = []
        for net in self.component_nets:
            params = net(features)
            mean, log_std = params.chunk(2, dim=1)
            mean = torch.exp(mean)
            std = torch.exp(log_std) * self.temperature
            components.append((mean, std))
        
        log_weights = self.weight_net(features)
        weights = torch.exp(log_weights)
        
        return components, weights

def power_specific_loss(components, weights, targets, lambda_tail=0.1):
    log_probs = []
    
    for (mean, std) in components:
        dist = torch.distributions.LogNormal(mean.log(), std)
        log_probs.append(dist.log_prob(targets.unsqueeze(1)))
    
    log_probs = torch.cat(log_probs, dim=1)
    log_mix = torch.logsumexp(log_probs + weights.log(), dim=1)
    
    tail_penalty = torch.mean(torch.relu(300 - targets) * torch.exp(-log_mix))
    
    return -log_mix.mean() + lambda_tail * tail_penalty

def find_distribution_peaks(data: pd.Series, bandwidth=10):
    kde = stats.gaussian_kde(data, bw_method=bandwidth/data.std())
    x_range = np.linspace(data.min(), data.max(), 1000)
    y = kde(x_range)
    
    peaks = []
    for i in range(1, len(y)-1):
        if y[i-1] < y[i] > y[i+1]:
            peaks.append(x_range[i])
    
    return sorted(peaks)

def preprocess_power(y_data: pd.Series) -> Tuple[np.ndarray, Dict]:
    valid_data = y_data[~pd.isna(y_data)]
    
    quantiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    q_values = valid_data.quantile(quantiles)
    
    lower = max(0, q_values[0.001])
    upper = q_values[0.999] * 1.2
    
    y_log = np.log1p(valid_data)
    
    stats = {
        'lower': float(lower),
        'upper': float(upper),
        'mean_log': float(np.mean(y_log)),
        'std_log': float(np.std(y_log)),
        'quantiles': {str(q): float(val) for q, val in zip(quantiles, q_values)},
        'peaks': find_distribution_peaks(valid_data)
    }
    
    y_normalized = (y_log - stats['mean_log']) / (stats['std_log'] + 1e-8)
    
    return y_normalized, stats

def postprocess_power_predictions(predictions: np.ndarray, stats: Dict) -> np.ndarray:
    predictions = predictions * (stats['std_log'] + 1e-8) + stats['mean_log']
    predictions = np.expm1(predictions)
    
    def smooth_quantile_mapping(x):
        q_values = np.array([float(v) for v in stats['quantiles'].values()])
        q_positions = np.array([float(q) for q in stats['quantiles'].keys()])
        
        idx = np.searchsorted(q_values, x)
        if idx == 0:
            return q_values[0]
        elif idx == len(q_values):
            return q_values[-1]
        
        alpha = (x - q_values[idx-1]) / (q_values[idx] - q_values[idx-1])
        alpha = 1 / (1 + np.exp(-12 * (alpha - 0.5)))
        
        return q_values[idx-1] + alpha * (q_values[idx] - q_values[idx-1])
    
    predictions = np.array([smooth_quantile_mapping(x) for x in predictions.flatten()])
    predictions = np.maximum(predictions, 0)
    predictions = np.minimum(predictions, stats['upper'])
    
    return predictions

def train_power_model(model, train_loader, val_loader, device, epochs=500):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            components, weights = model(batch_X)
            loss = power_specific_loss(components, weights, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                components, weights = model(batch_X)
                val_loss = power_specific_loss(components, weights, batch_y)
                val_losses.append(val_loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def save_model_artifacts(model, scaler, config, model_dir: str, target_column: str):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / f"{target_column}_model.pt")
    
    with open(model_dir / f"{target_column}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(model_dir / f"{target_column}_config.pkl", 'wb') as f:
        pickle.dump(config, f)

def load_model_artifacts(model_dir: str, target_column: str, device: torch.device) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[ModelConfig]]:
    model_dir = Path(model_dir)
    model_path = model_dir / f"{target_column}_model.pt"
    scaler_path = model_dir / f"{target_column}_scaler.pkl"
    config_path = model_dir / f"{target_column}_config.pkl"
    
    if not all(p.exists() for p in [model_path, scaler_path, config_path]):
        return None, None, None
    
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        model = PowerMixtureModel(len(config.feature_columns)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, scaler, config
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def evaluate_model(original_data: pd.DataFrame, 
                  filled_data: pd.DataFrame, 
                  target_column: str,
                  mask: pd.Series):
    original_stats = original_data[~mask][target_column].describe()
    filled_stats = filled_data[mask][target_column].describe()
    
    print("\nModel Evaluation Results:")
    print("\n1. Statistical Comparison:")
    print("\nOriginal Data Statistics:")
    print(original_stats)
    print("\nFilled Data Statistics:")
    print(filled_stats)
    
    Q1 = original_stats['25%']
    Q3 = original_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = filled_data[mask][
        (filled_data[mask][target_column] < lower_bound) |
        (filled_data[mask][target_column] > upper_bound)
    ]
    
    print(f"\n2. Outlier Analysis:")
    print(f"Number of outlier predictions: {len(outliers)} ({len(outliers)/sum(mask)*100:.2f}%)")
    if len(outliers) > 0:
        print("Outlier Statistics:")
        print(outliers[target_column].describe())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(data=original_data[~mask][target_column], kde=True, label='Original', alpha=0.6)
    sns.histplot(data=filled_data[mask][target_column], kde=True, label='Filled', alpha=0.6)
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(132)
    box_data = pd.DataFrame({
        'Value': pd.concat([original_data[~mask][target_column], filled_data[mask][target_column]]),
        'Type': ['Original'] * sum(~mask) + ['Filled'] * sum(mask)
    })
    sns.boxplot(data=box_data, x='Type', y='Value')
    plt.title('Box Plot Comparison')
    
    plt.subplot(133)
    sns.kdeplot(data=original_data[~mask][target_column], label='Original')
    sns.kdeplot(data=filled_data[mask][target_column], label='Filled')
    plt.title('Density Plot Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'outliers_count': len(outliers),
        'outliers_percentage': len(outliers)/sum(mask)*100,
        'mean_difference': abs(original_stats['mean'] - filled_stats['mean']),
        'std_difference': abs(original_stats['std'] - filled_stats['std']),
        'median_difference': abs(original_stats['50%'] - filled_stats['50%']),
        'distribution_statistics': {
            'original_mean': original_stats['mean'],
            'filled_mean': filled_stats['mean'],
            'original_std': original_stats['std'],
            'filled_std': filled_stats['std'],
            'original_median': original_stats['50%'],
            'filled_median': filled_stats['50%']
        }
    }

def impute_power_values(data: pd.DataFrame,
                       target_column: str,
                       feature_columns: List[str],
                       model_dir: str = 'power_models',
                       device: Optional[torch.device] = None) -> Tuple[pd.DataFrame, Dict]:
    """针对功率特征的缺失值填充主函数"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_filled = data.copy()
    stats = {}
    
    # 检查并处理数据
    X_data = data[feature_columns].copy()
    y_data = data[target_column].copy()
    
    # 处理特征
    X_data = X_data.replace([np.inf, -np.inf], np.nan)
    for col in feature_columns:
        X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
        X_data[col] = X_data[col].fillna(X_data[col].median())
    
    # 获取缺失值掩码
    mask = pd.isna(y_data)
    stats['initial_missing'] = mask.sum()
    
    if stats['initial_missing'] == 0:
        print("No missing values to impute")
        return data_filled, stats
    
    # 尝试加载现有模型
    model, scaler, config = load_model_artifacts(model_dir, target_column, device)
    
    if model is None:
        print("Training new model...")
        # 预处理特征
        scaler = RobustScaler(quantile_range=(1, 99))
        X_scaled = scaler.fit_transform(X_data)
        X_scaled = np.clip(X_scaled, -10, 10)
        
        # 预处理目标值 - 只处理非缺失值
        valid_y = y_data[~mask]
        y_normalized, y_stats = preprocess_power(valid_y)
        config = ModelConfig(feature_columns=feature_columns, y_stats=y_stats)
        
        # 准备训练数据 - 只使用非缺失值的数据
        X = torch.tensor(X_scaled[~mask], dtype=torch.float32)
        y = torch.tensor(y_normalized, dtype=torch.float32)
        
        # 分割训练集和验证集
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), 
            train_size=0.8, 
            random_state=42
        )
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            X[train_idx],
            y[train_idx]
        )
        val_dataset = torch.utils.data.TensorDataset(
            X[val_idx],
            y[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # 创建并训练模型
        model = PowerMixtureModel(len(feature_columns)).to(device)
        train_power_model(model, train_loader, val_loader, device)
        
        # 保存模型和相关文件
        save_model_artifacts(model, scaler, config, model_dir, target_column)
    else:
        print("Using existing model...")
        y_stats = config.y_stats
        X_scaled = scaler.transform(X_data)
        X_scaled = np.clip(X_scaled, -10, 10)
    
    # 预测缺失值
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        X_missing = X_tensor[mask].to(device)
        if len(X_missing) > 0:
            components, weights = model(X_missing)
            # 使用加权平均作为预测值
            predictions = torch.zeros_like(weights[:, 0])
            for (mean, _), w in zip(components, weights.t()):
                predictions += mean.squeeze() * w
            predictions = predictions.cpu().numpy()
            predictions = postprocess_power_predictions(predictions, y_stats)
            data_filled.loc[mask, target_column] = predictions
    
    # 计算统计信息并评估
    stats['filled_values'] = stats['initial_missing']
    stats['final_missing'] = data_filled[target_column].isna().sum()
    
    if stats['filled_values'] > 0:
        filled_values = data_filled.loc[mask, target_column]
        stats['fill_stats'] = {
            'mean': float(filled_values.mean()),
            'std': float(filled_values.std()),
            'min': float(filled_values.min()),
            'max': float(filled_values.max()),
            'median': float(filled_values.median())
        }
        
        print(f"\nFilled {stats['filled_values']} missing values")
        print(f"Final missing values: {stats['final_missing']}")
        print("\nFilled values statistics:")
        for key, value in stats['fill_stats'].items():
            print(f"{key}: {value:.2f}")
        
        # 模型评估
        eval_results = evaluate_model(data, data_filled, target_column, mask)
        stats['evaluation'] = eval_results
        
        # 添加额外的评估指标
        original_valid = data[~mask][target_column]
        stats['evaluation'].update({
            'distribution_metrics': {
                'skewness_difference': abs(original_valid.skew() - filled_values.skew()),
                'kurtosis_difference': abs(original_valid.kurtosis() - filled_values.kurtosis()),
                'peaks': find_distribution_peaks(filled_values)
            }
        })
        
        print("\nDistribution Metrics:")
        print(f"Skewness Difference: {stats['evaluation']['distribution_metrics']['skewness_difference']:.4f}")
        print(f"Kurtosis Difference: {stats['evaluation']['distribution_metrics']['kurtosis_difference']:.4f}")
    
    return data_filled, stats

def run_power_imputation(data: pd.DataFrame, 
                        target_column: str = 'power',
                        feature_columns: Optional[List[str]] = None,
                        model_dir: str = 'power_models'):
    """运行功率特征缺失值填充的便捷函数"""
    
    if feature_columns is None:
        feature_columns = ['engine_cap', 'curb_weight', 'road_tax']
    
    print(f"Starting power imputation")
    print(f"Using features: {feature_columns}")
    print(f"Initial missing values: {data[target_column].isna().sum()}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 运行填充
    filled_data, stats = impute_power_values(
        data=data,
        target_column=target_column,
        feature_columns=feature_columns,
        model_dir=model_dir,
        device=device
    )
    
    print("\nPower imputation completed!")
    print(f"Final number of missing values: {filled_data[target_column].isna().sum()}")
    
    return filled_data, stats