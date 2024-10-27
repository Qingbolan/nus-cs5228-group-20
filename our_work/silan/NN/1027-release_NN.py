import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import pickle
import time
from typing import Tuple, List, Dict, Any, Optional
import os
import logging
import warnings
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class CarPriceDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values) if y is not None else None  # 修复：使用.values转换pandas Series
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class PricePredictor(nn.Module):
    """神经网络模型"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x).squeeze()

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """加载和初始预处理数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    return X, y

class DataPreprocessor:
    """数据预处理类"""
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        # 分离数值和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 处理数值特征
        X_num = self.num_imputer.fit_transform(X[numeric_features])
        X_num = self.scaler.fit_transform(X_num)
        
        # 处理分类特征
        X_cat = X[categorical_features].copy()
        for col in categorical_features:
            X_cat[col] = self.cat_imputer.fit_transform(X_cat[[col]])
            self.label_encoders[col] = LabelEncoder()
            X_cat[col] = self.label_encoders[col].fit_transform(X_cat[col])
        
        # 合并所有特征
        X_processed = np.hstack([X_num, X_cat.values])
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # 分离数值和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 处理数值特征
        X_num = self.num_imputer.transform(X[numeric_features])
        X_num = self.scaler.transform(X_num)
        
        # 处理分类特征
        X_cat = X[categorical_features].copy()
        for col in categorical_features:
            X_cat[col] = self.cat_imputer.transform(X_cat[[col]])
            # 处理测试集中的新类别
            X_cat[col] = X_cat[col].map(lambda x: -1 if x not in self.label_encoders[col].classes_ 
                                      else self.label_encoders[col].transform([x])[0])
        
        # 合并所有特征
        X_processed = np.hstack([X_num, X_cat.values])
        return X_processed

# 2. 恢复原来的聚类代码
def find_optimal_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 10,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> int:
    """Find optimal number of clusters using silhouette score."""
    logging.info("Starting cluster optimization")
    
    missing_features = [f for f in features_for_clustering if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing clustering features: {missing_features}")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.4f}")
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Selected optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def create_price_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: int,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """Create price-based clusters using KMeans."""
    logging.info(f"Creating {n_clusters} price clusters")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(cluster_features_clean, np.linspace(0, 100, n_clusters), axis=0)
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=3, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    price_clusters = kmeans.fit_predict(cluster_features)
    
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("\nCluster Statistics:")
    logging.info(cluster_df)
    
    kmeans.feature_imputer = imputer
    return kmeans, price_clusters, cluster_df

def predict_cluster(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    kmeans_model: KMeans,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> np.ndarray:
    """Predict clusters for new data."""
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    
    dummy_y = np.zeros(len(X)) if y is None else np.log1p(y)
    cluster_features = np.column_stack([dummy_y, cluster_features_clean])
    
    return kmeans_model.predict(cluster_features)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float, float]:
    """验证模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    val_loss = total_loss / len(val_loader)
    val_rmse = np.sqrt(mean_squared_error(targets, predictions))
    val_r2 = r2_score(targets, predictions)
    
    return val_loss, val_rmse, val_r2

def train_model(
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hidden_dims: List[int] = [256, 128, 64],
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """训练模型"""
    model = PricePredictor(input_dim, hidden_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val RMSE: {val_rmse:.4f}, "
            f"Val R2: {val_r2:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, history

def main():
    """主执行函数"""
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # 加载数据
        X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
        
        # 对目标变量进行对数变换
        y_log = np.log1p(y)
        
        # 聚类
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters,
            features_for_clustering=features_for_clustering
        )
        
        # 模型参数
        nn_params = {
            'hidden_dims': [256, 128, 64],
            'epochs': 100,
            'patience': 10,
            'learning_rate': 1e-3,
            'batch_size': 64
        }
        
        # 准备交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(X))
        models = []
        
        start_time = time.time()
        
        # 对每个聚类训练模型
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            cluster_mask = price_clusters == cluster
            X_cluster = X[cluster_mask]
            y_cluster = y_log[cluster_mask]
            
            cluster_models = []
            
            for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
                logging.info(f"Processing fold {fold}")
                
                # 准备数据
                X_train, X_val = X_cluster.iloc[train_index], X_cluster.iloc[val_index]
                y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
                
                # 预处理
                preprocessor = DataPreprocessor()
                X_train_processed = preprocessor.fit_transform(X_train)
                X_val_processed = preprocessor.transform(X_val)
                
                # 创建数据加载器
                train_dataset = CarPriceDataset(X_train_processed, y_train)
                val_dataset = CarPriceDataset(X_val_processed, y_val)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=nn_params['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=nn_params['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                # 训练模型
                input_dim = X_train_processed.shape[1]
                model, history = train_model(
                    input_dim=input_dim,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    hidden_dims=nn_params['hidden_dims'],
                    epochs=nn_params['epochs'],
                    patience=nn_params['patience'],
                    learning_rate=nn_params['learning_rate']
                )
                
                # 生成验证集预测
                model.eval()
                with torch.no_grad():
                    val_dataset = CarPriceDataset(X_val_processed)
                    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
                    val_predictions = []
                    
                    for X_batch in val_loader:
                        X_batch = X_batch.to(device)
                        outputs = model(X_batch)
                        val_predictions.extend(outputs.cpu().numpy())
                
                # 转换回原始尺度
                val_predictions = np.expm1(val_predictions)
                oof_predictions[cluster_mask][val_index] = val_predictions
                
                # 保存模型和预处理器
                cluster_models.append({
                    'model': model,
                    'preprocessor': preprocessor,
                    'history': history
                })
            
            models.append(cluster_models)
        
        # 训练完成统计
        elapsed_time = time.time() - start_time
        logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
        
        # 评估OOF预测
        oof_mse = mean_squared_error(y, oof_predictions)
        oof_r2 = r2_score(y, oof_predictions)
        logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
        logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
        
        # 保存模型
        model_save_path = 'nn_clustered_models.pkl'
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'kmeans_model': kmeans_model,
                'cluster_info': cluster_info,
                'nn_params': nn_params
            }, f)
        logging.info(f"Models saved to {model_save_path}")
        
        # 生成测试集预测
        X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
        
        # 预测测试集的聚类
        dummy_y_test = np.zeros(len(X_test))
        test_clusters = predict_cluster(X_test, dummy_y_test, kmeans_model, features_for_clustering)
        
        # 对每个聚类进行预测
        final_predictions = np.zeros(len(X_test))
        
        for cluster in range(len(cluster_info)):
            cluster_mask = test_clusters == cluster
            X_test_cluster = X_test[cluster_mask]
            
            if len(X_test_cluster) == 0:
                logging.warning(f"No samples in test data for cluster {cluster}")
                continue
            
            cluster_predictions = np.zeros((len(X_test_cluster), len(models[cluster])))
            
            for i, model_dict in enumerate(models[cluster]):
                try:
                    model = model_dict['model']
                    preprocessor = model_dict['preprocessor']
                    
                    # 预处理测试数据
                    X_test_processed = preprocessor.transform(X_test_cluster)
                    test_dataset = CarPriceDataset(X_test_processed)
                    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
                    
                    # 生成预测
                    model.eval()
                    predictions = []
                    with torch.no_grad():
                        for X_batch in test_loader:
                            X_batch = X_batch.to(device)
                            outputs = model(X_batch)
                            predictions.extend(outputs.cpu().numpy())
                    
                    # 转换回原始尺度
                    cluster_predictions[:, i] = np.expm1(predictions)
                    
                except Exception as e:
                    logging.error(f"Error predicting cluster {cluster}, model {i}: {str(e)}")
                    continue
            
            final_predictions[cluster_mask] = np.mean(cluster_predictions, axis=1)
        
        # 后处理预测结果
        min_price = 700
        max_price = 2900000
        final_predictions = np.clip(final_predictions, min_price, max_price)
        
        # 保存预测结果
        submission = pd.DataFrame({
            'Id': range(len(final_predictions)),
            'Predicted': np.round(final_predictions).astype(int)
        })
        
        submission_path = '10-27-release_nn_clustered.csv'
        submission.to_csv(submission_path, index=False)
        logging.info(f"Predictions saved to {submission_path}")
        
        # 预测统计
        logging.info("\nPrediction statistics:")
        logging.info(f"Minimum: {final_predictions.min():.2f}")
        logging.info(f"Maximum: {final_predictions.max():.2f}")
        logging.info(f"Mean: {final_predictions.mean():.2f}")
        logging.info(f"Median: {np.median(final_predictions):.2f}")
        
        # 聚类预测统计
        logging.info("\nCluster-wise prediction statistics:")
        for cluster in range(len(cluster_info)):
            cluster_mask = test_clusters == cluster
            if np.any(cluster_mask):
                cluster_preds = final_predictions[cluster_mask]
                logging.info(f"\nCluster {cluster} statistics:")
                logging.info(f"Count: {len(cluster_preds)}")
                logging.info(f"Mean: {cluster_preds.mean():.2f}")
                logging.info(f"Median: {np.median(cluster_preds):.2f}")
                logging.info(f"Min: {cluster_preds.min():.2f}")
                logging.info(f"Max: {cluster_preds.max():.2f}")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()