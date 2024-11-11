import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from typing import Tuple, List, Dict, Any, Optional
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load and perform initial preprocessing of data."""
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

def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Create a preprocessing pipeline using ColumnTransformer."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def optimize_svr_params(price_range: str) -> Dict[str, Any]:
    """为不同价格范围优化SVR参数"""
    if price_range == 'low':
        return {
            'kernel': 'rbf',
            'C': 150,        # 增加C值以更好拟合低价区间
            'epsilon': 0.05,  # 降低epsilon以提高精度
            'gamma': 'scale',
            'cache_size': 2000,
        }
    elif price_range == 'medium':
        return {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale',
            'cache_size': 2000,
        }
    else:  # high
        return {
            'kernel': 'rbf',
            'C': 80,         # 降低C值以避免高价区间过拟合
            'epsilon': 0.15,  # 增加epsilon以提高泛化能力
            'gamma': 'scale',
            'cache_size': 2000,
        }

def find_optimal_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 3,
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
    
    scaler = RobustScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
    
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
    """Create price-based clusters."""
    logging.info(f"Creating {n_clusters} price clusters")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    scaler = RobustScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    
    # 计算每个聚类的初始中心
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(cluster_features_scaled, np.linspace(0, 100, n_clusters), axis=0)
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=3, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
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
    kmeans.feature_scaler = scaler
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
    cluster_features_scaled = kmeans_model.feature_scaler.transform(cluster_features_clean)
    
    if y is None:
        # 对于测试集，我们使用一个虚拟的价格值
        dummy_price = np.zeros(len(X))
        cluster_features = np.column_stack([dummy_price, cluster_features_scaled])
    else:
        cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
    
    return kmeans_model.predict(cluster_features)

def main():
    """Main execution function with optimized SVR training."""
    try:
        np.random.seed(42)
        
        # 加载数据
        X, y = load_and_preprocess_data('l2_train.csv')
        
        # 识别特征类型
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # 聚类
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, 
                                              features_for_clustering=features_for_clustering)
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters,
            features_for_clustering=features_for_clustering
        )
        
        # 为每个聚类确定价格范围
        price_ranges = ['low', 'medium', 'high']
        cluster_models = []
        
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            mask = price_clusters == cluster
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            # 获取优化的SVR参数
            svr_params = optimize_svr_params(price_ranges[cluster])
            
            # 创建预处理器和模型
            preprocessor = create_preprocessor(numeric_features, categorical_features)
            model = SVR(**svr_params)
            
            # 创建pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('svr', model)
            ])
            
            # 对目标值进行对数变换
            y_cluster_log = np.log1p(y_cluster)
            
            # 训练模型
            pipeline.fit(X_cluster, y_cluster_log)
            
            cluster_models.append(pipeline)
        
        # 保存模型
        with open('optimized_svr_models.pkl', 'wb') as f:
            pickle.dump({
                'models': cluster_models,
                'kmeans': kmeans_model,
                'cluster_info': cluster_info
            }, f)
        
        # 生成测试集预测
        X_test, _ = load_and_preprocess_data('l2_test.csv')
        
        # 预测测试集的聚类
        test_clusters = predict_cluster(X_test, None, kmeans_model, features_for_clustering)
        final_predictions = np.zeros(len(X_test))
        
        for cluster in range(len(cluster_info)):
            cluster_mask = test_clusters == cluster
            if not any(cluster_mask):
                continue
                
            X_test_cluster = X_test[cluster_mask]
            
            # 使用对应的模型进行预测
            model = cluster_models[cluster]
            predictions_log = model.predict(X_test_cluster)
            
            # 转换回原始尺度
            predictions = np.expm1(predictions_log)
            final_predictions[cluster_mask] = predictions
        
        # 后处理预测结果
        final_predictions = np.clip(final_predictions, 700, 2900000)
        
        # 保存预测结果
        submission = pd.DataFrame({
            'Id': range(len(final_predictions)),
            'Predicted': np.round(final_predictions).astype(int)
        })
        
        submission.to_csv('optimized_svr_predictions.csv', index=False)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()