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
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler  # 使用RobustScaler来处理异常值

# (test)RMSE Score: 33214.4162

# 设置日志
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
    
    logging.info(f"Features shape: {X.shape}")
    if y is not None:
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Price range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y

def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Create a preprocessing pipeline using ColumnTransformer."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # 使用RobustScaler来处理异常值
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
    
    # 为聚类标准化特征
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
    """Create price-based clusters using KMeans."""
    logging.info(f"Creating {n_clusters} price clusters")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    # 为聚类标准化特征
    scaler = RobustScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    
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
    
    dummy_y = np.zeros(len(X)) if y is None else np.log1p(y)
    cluster_features = np.column_stack([dummy_y, cluster_features_scaled])
    
    return kmeans_model.predict(cluster_features)

def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any]
) -> SVR:
    """Train SVR model."""
    model = SVR(**params)
    model.fit(X_train, y_train)
    
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    logging.info(f"Validation RMSE: {np.sqrt(val_mse):.4f}")
    logging.info(f"Validation R2: {val_r2:.4f}")
    
    return model

def main():
    """Main execution function."""
    try:
        np.random.seed(42)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # 加载数据
        X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
        
        # 识别特征类型
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # SVR参数
        svr_params = {
            'kernel': 'rbf',  
            'C': 100,        # 正则化参数
            'epsilon': 0.1,   # epsilon-tube
            'gamma': 'scale', # RBF核参数
            'cache_size': 2000,  # 缓存大小(MB)
            'verbose': False
        }
        
        # 聚类
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters,
            features_for_clustering=features_for_clustering
        )
        
        # 准备交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 减少折数以加快训练
        oof_predictions = np.zeros(len(X))
        models = []
        
        start_time = time.time()
        
        # 对每个聚类训练模型
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            X_cluster = X[price_clusters == cluster]
            y_cluster = y[price_clusters == cluster]
            
            if len(X_cluster) > 10000:  # 如果数据量太大，进行随机采样
                sample_idx = np.random.choice(len(X_cluster), 10000, replace=False)
                X_cluster = X_cluster.iloc[sample_idx]
                y_cluster = y_cluster.iloc[sample_idx]
            
            cluster_models = []
            
            for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
                logging.info(f"Processing fold {fold}")
                
                X_train, X_val = X_cluster.iloc[train_index], X_cluster.iloc[val_index]
                y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
                
                # 创建预处理器
                preprocessor = create_preprocessor(numeric_features, categorical_features)
                
                # 预处理数据
                X_train_processed = preprocessor.fit_transform(X_train)
                X_val_processed = preprocessor.transform(X_val)
                
                # 对目标变量进行对数变换
                y_train_log = np.log1p(y_train)
                y_val_log = np.log1p(y_val)
                
                # 训练模型
                model = train_svr(
                    X_train_processed,
                    y_train_log,
                    X_val_processed,
                    y_val_log,
                    svr_params
                )
                
                # 生成预测并转换回原始尺度
                val_predictions = np.expm1(model.predict(X_val_processed))
                oof_predictions[price_clusters == cluster][val_index] = val_predictions
                
                # 保存模型和预处理器
                cluster_models.append({
                    'model': model,
                    'preprocessor': preprocessor
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
        model_save_path = 'svr_clustered_models.pkl'
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'kmeans_model': kmeans_model,
                'cluster_info': cluster_info
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
                    
                    X_test_processed = preprocessor.transform(X_test_cluster)
                    # 预测并转换回原始尺度
                    cluster_predictions[:, i] = np.expm1(model.predict(X_test_processed))
                    
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
        
        submission_path = '10-27-release_svr_clustered.csv'
        submission.to_csv(submission_path, index=False)
        logging.info(f"Predictions saved to {submission_path}")
        
        # 预测统计
        logging.info("\nOverall prediction statistics:")
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

def tune_svr_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    为不同规模的数据调整SVR参数
    """
    n_samples = len(X_train)
    
    if n_samples < 1000:
        # 小数据集：可以使用更复杂的模型
        return {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale',
            'cache_size': 2000,
            'verbose': False
        }
    elif n_samples < 5000:
        # 中等数据集：平衡复杂度
        return {
            'kernel': 'rbf',
            'C': 50,
            'epsilon': 0.2,
            'gamma': 'scale',
            'cache_size': 2000,
            'verbose': False
        }
    else:
        # 大数据集：使用更简单的模型
        return {
            'kernel': 'linear',  # 线性核以提高速度
            'C': 10,
            'epsilon': 0.5,
            'cache_size': 2000,
            'verbose': False
        }

def verify_predictions(predictions: np.ndarray, cluster_info: pd.DataFrame) -> bool:
    """
    验证预测结果的合理性
    """
    # 检查基本范围
    if np.any(predictions < 0):
        logging.error("Found negative predictions")
        return False
    
    # 检查离群值
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = np.sum((predictions < lower_bound) | (predictions > upper_bound))
    if outliers > len(predictions) * 0.01:  # 如果离群值超过1%
        logging.warning(f"Found {outliers} outliers in predictions")
    
    # 检查与聚类统计的一致性
    for cluster in cluster_info.itertuples():
        cluster_min = cluster.min * 0.5  # 允许一定的浮动范围
        cluster_max = cluster.max * 1.5
        
        if np.any(predictions < cluster_min) or np.any(predictions > cluster_max):
            logging.warning(f"Predictions for cluster {cluster.cluster} exceed historical range")
    
    return True

if __name__ == '__main__':
    main()