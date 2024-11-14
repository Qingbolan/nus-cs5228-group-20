import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import logging
import time
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """加载并准备数据"""

    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns}")
    
    return X, y

def preprocess_features(X: pd.DataFrame, 
                       encoders: Dict = None, 
                       fit: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """特征预处理"""
    X = X.copy()
    
    if encoders is None:
        encoders = {
            'num_imputer': SimpleImputer(strategy='median'),
            'robust_scaler': RobustScaler(),
            'standard_scaler': StandardScaler(),
        }
    
    # 数值型特征
    numeric_cols = [
        'curb_weight', 'power', 'engine_cap', 'no_of_owners',
        'depreciation', 'coe', 'road_tax', 'dereg_value', 'omv',
        'arf', 'vehicle_age'
    ]
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    
    # 标准化特征
    scale_cols = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    scale_cols = [col for col in scale_cols if col in X.columns]
    
    # 鲁棒缩放特征
    robust_cols = ['coe', 'dereg_value', 'omv', 'arf']
    robust_cols = [col for col in robust_cols if col in X.columns]
    
    if fit:
        X[numeric_cols] = encoders['num_imputer'].fit_transform(X[numeric_cols])
        X[scale_cols] = encoders['standard_scaler'].fit_transform(X[scale_cols].values)
        X[robust_cols] = encoders['robust_scaler'].fit_transform(X[robust_cols].values)
    else:
        X[numeric_cols] = encoders['num_imputer'].transform(X[numeric_cols])
        X[scale_cols] = encoders['standard_scaler'].transform(X[scale_cols].values)
        X[robust_cols] = encoders['robust_scaler'].transform(X[robust_cols].values)
    
    return X, encoders

def create_svr_model(cluster_stats: Dict) -> SVR:
    """创建优化的SVR模型"""
    price_std = cluster_stats['std']
    price_range = cluster_stats['max'] - cluster_stats['min']
    sample_count = cluster_stats['count']
    
    # 动态参数调整
    base_epsilon = (price_std / price_range) * 0.05
    base_C = np.log1p(sample_count) * 20
    
    if sample_count < 1000:
        base_C *= 1.2
        base_epsilon *= 0.8
    
    return SVR(
        kernel='rbf',
        C=base_C,
        epsilon=base_epsilon,
        gamma='scale',
        cache_size=2000,
        tol=0.001
    )

def train_cluster_models(X: pd.DataFrame, 
                        y: pd.Series, 
                        clusters: np.ndarray,
                        n_splits: int = 5) -> List[Dict]:
    """训练聚类模型"""
    unique_clusters = np.unique(clusters)
    models = []
    
    for cluster in unique_clusters:
        logging.info(f"\nTraining models for Cluster {cluster}")
        mask = clusters == cluster
        X_cluster = X[mask]
        y_cluster = y[mask]
        
        cluster_stats = {
            'min': y_cluster.min(),
            'max': y_cluster.max(),
            'median': y_cluster.median(),
            'count': len(y_cluster),
            'std': y_cluster.std()
        }
        logging.info(f"Cluster stats: {cluster_stats}")
        
        # 创建交叉验证折叠
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Training fold {fold}")
            
            # 准备数据
            X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
            y_train, y_val = y_cluster.iloc[train_idx], y_cluster.iloc[val_idx]
            
            # 特征预处理
            X_train_processed, encoders = preprocess_features(X_train, fit=True)
            X_val_processed, _ = preprocess_features(X_val, encoders=encoders)
            
            # 目标值转换
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # 训练模型
            model = create_svr_model(cluster_stats)
            model.fit(X_train_processed, y_train_scaled)
            
            # 验证
            y_val_pred = y_scaler.inverse_transform(
                model.predict(X_val_processed).reshape(-1, 1)
            ).ravel()
            
            # 评估
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            r2 = r2_score(y_val, y_val_pred)
            
            logging.info(f"Fold {fold} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            fold_models.append({
                'model': model,
                'encoders': encoders,
                'y_scaler': y_scaler,
                'performance': {'rmse': rmse, 'r2': r2}
            })
        
        models.append({
            'cluster': cluster,
            'models': fold_models,
            'stats': cluster_stats
        })
    
    return models

def predict_prices(X: pd.DataFrame, 
                  models: List[Dict], 
                  kmeans: KMeans) -> np.ndarray:
    """预测价格"""
    # 预测聚类
    cluster_features = [
        'depreciation', 'coe', 'dereg_value', 'omv', 'arf',
        'curb_weight', 'engine_cap', 'vehicle_age'
    ]
    cluster_features = [f for f in cluster_features if f in X.columns]
    
    # 准备聚类特征
    scaler = RobustScaler()
    X_cluster = scaler.fit_transform(X[cluster_features])
    dummy_price = np.zeros((len(X), 1))
    cluster_data = np.hstack([X_cluster, dummy_price])
    
    clusters = kmeans.predict(cluster_data)
    predictions = np.zeros(len(X))
    
    # 对每个簇进行预测
    for cluster_dict in models:
        cluster = cluster_dict['cluster']
        mask = clusters == cluster
        if not any(mask):
            continue
        
        cluster_predictions = []
        weights = []
        
        # 使用每个折叠的模型进行预测
        for model_dict in cluster_dict['models']:
            X_processed, _ = preprocess_features(
                X[mask],
                encoders=model_dict['encoders']
            )
            
            pred_scaled = model_dict['model'].predict(X_processed)
            pred = model_dict['y_scaler'].inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).ravel()
            
            cluster_predictions.append(pred)
            weights.append(model_dict['performance']['r2'])
        
        # 加权平均
        weights = np.array(weights) / sum(weights)
        cluster_pred = np.average(
            cluster_predictions,
            axis=0,
            weights=weights
        )
        
        # 应用聚类特定的约束
        cluster_stats = cluster_dict['stats']
        min_price = max(700, cluster_stats['median'] - 3 * cluster_stats['std'])
        max_price = min(2900000, cluster_stats['median'] + 3 * cluster_stats['std'])
        cluster_pred = np.clip(cluster_pred, min_price, max_price)
        
        predictions[mask] = cluster_pred
    
    return np.round(predictions).astype(int)

def create_clusters(X: pd.DataFrame, 
                   y: pd.Series, 
                   n_clusters: int = 3) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """改进的聚类方法"""
    # 聚类特征
    cluster_features = [
        'depreciation', 'coe', 'dereg_value', 'omv', 'arf',
        'curb_weight', 'engine_cap', 'vehicle_age'
    ]
    cluster_features = [f for f in cluster_features if f in X.columns]
    
    # 特征预处理
    scaler = RobustScaler()
    X_cluster = scaler.fit_transform(X[cluster_features])
    
    # 添加价格信息
    price_log = np.log1p(y)
    price_scaled = scaler.fit_transform(price_log.values.reshape(-1, 1))
    cluster_data = np.hstack([X_cluster, price_scaled])
    
    # 修复：确保生成正确数量的初始中心点
    percentiles = np.linspace(0, 100, n_clusters + 1)[1:-1]
    price_percentiles = np.percentile(y, percentiles)
    initial_centers = []
    
    last_threshold = 0
    for i, threshold in enumerate(price_percentiles):
        mask = (y > last_threshold) & (y <= threshold)
        if sum(mask) > 0:  # 确保有足够的样本
            center_features = cluster_data[mask].mean(axis=0)
        else:  # 如果没有样本，使用相邻区间的平均值
            lower_mask = y <= threshold
            upper_mask = y > last_threshold
            center_features = (cluster_data[lower_mask].mean(axis=0) + 
                             cluster_data[upper_mask].mean(axis=0)) / 2
        initial_centers.append(center_features)
        last_threshold = threshold
    
    # 添加最后一个中心点
    mask = y > price_percentiles[-1]
    if sum(mask) > 0:
        center_features = cluster_data[mask].mean(axis=0)
    else:
        center_features = cluster_data[y > np.percentile(y, 90)].mean(axis=0)
    initial_centers.append(center_features)
    
    initial_centers = np.vstack(initial_centers)
    
    # 确保初始中心点的数量正确
    assert initial_centers.shape[0] == n_clusters, \
        f"Got {initial_centers.shape[0]} centers, expected {n_clusters}"
    
    # 执行聚类
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=initial_centers,
        n_init=1,
        random_state=42
    )
    clusters = kmeans.fit_predict(cluster_data)
    
    # 收集聚类信息
    cluster_info = []
    for cluster in range(n_clusters):
        mask = clusters == cluster
        cluster_prices = y[mask]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices),
            'std': cluster_prices.std()
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Cluster Information:")
    logging.info(cluster_df)
    
    return kmeans, clusters, cluster_df

def main():
    """主函数"""
    try:
        start_time = time.time()
        np.random.seed(42)
        
        # 加载数据
        X_train, y_train = load_and_prepare_data('l2_train.csv')
        X_test, _ = load_and_prepare_data('l2_test.csv')
        
        # 创建聚类
        n_clusters = 3  # 明确指定聚类数量
        kmeans, clusters, cluster_info = create_clusters(X_train, y_train, n_clusters=n_clusters)
        
        # 训练模型
        models = train_cluster_models(X_train, y_train, clusters)
        
        # 预测
        predictions = predict_prices(X_test, models, kmeans)
        
        # 保存结果
        submission = pd.DataFrame({
            'Id': range(len(predictions)),
            'Predicted': predictions
        })
        submission.to_csv('predictions.csv', index=False)
        
        # 输出训练信息
        duration = (time.time() - start_time) / 60
        logging.info(f"\nTraining completed in {duration:.2f} minutes")
        logging.info(f"Predictions summary:")
        logging.info(f"Min: {predictions.min()}")
        logging.info(f"Max: {predictions.max()}")
        logging.info(f"Mean: {predictions.mean():.2f}")
        logging.info(f"Median: {np.median(predictions):.2f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()