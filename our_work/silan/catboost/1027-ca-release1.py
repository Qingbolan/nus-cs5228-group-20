import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import time
from typing import Tuple, List, Dict, Any, Optional
import os
import logging
import warnings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

# 保留原有的数据加载和聚类函数
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

def preprocess_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    num_imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    categorical_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, SimpleImputer, StandardScaler, List[str]]:
    """Feature preprocessing optimized for CatBoost."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    X = X.copy()
    logging.info("Starting feature preprocessing")
    
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")
    
    if len(numeric_features) > 0:
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
        else:
            X[numeric_features] = num_imputer.transform(X[numeric_features])
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in numeric_features]
    
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
        else:
            X[columns_to_standardize] = scaler.transform(X[columns_to_standardize])
    
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            X[cat_feature] = X[cat_feature].fillna('unknown')
            X[cat_feature] = X[cat_feature].astype(str)
    
    logging.info(f"Completed preprocessing. Final shape: {X.shape}")
    return X, num_imputer, scaler, categorical_features

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    categorical_features: List[str]
) -> CatBoostRegressor:
    """Train CatBoost model."""
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=categorical_features,
        use_best_model=True
    )
    
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
        
        logging.info("\nTarget variable (price) statistics:")
        logging.info(y.describe())
        
        # 聚类
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters,
            features_for_clustering=features_for_clustering
        )
        
        # CatBoost参数
        cb_params = {
            'iterations': 3000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': 42,
            'verbose': False
        }
        
        # 准备交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(X))
        feature_importance_list = []
        models = []
        
        start_time = time.time()
        
        # 对每个聚类训练模型
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            X_cluster = X[price_clusters == cluster]
            y_cluster = y[price_clusters == cluster]
            
            cluster_models = []
            
            for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
                logging.info(f"Processing fold {fold}")
                
                X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
                y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
                
                X_train_processed, num_imputer, scaler, categorical_features = preprocess_features(X_train, y_train)
                X_val_processed, _, _, _ = preprocess_features(
                    X_val,
                    num_imputer=num_imputer,
                    scaler=scaler,
                    categorical_features=categorical_features
                )
                
                model = train_catboost(
                    X_train_processed,
                    y_train,
                    X_val_processed,
                    y_val,
                    cb_params,
                    categorical_features
                )
                
                # 生成预测
                fold_predictions = model.predict(X_val_processed)
                oof_predictions[price_clusters == cluster][val_index] = fold_predictions
                
                # 特征重要性
                importance = model.get_feature_importance()
                feature_importance = pd.DataFrame({
                    'feature': X_train_processed.columns,
                    'importance': importance
                })
                feature_importance_list.append(feature_importance)
                
                # 保存模型和预处理器
                cluster_models.append({
                    'model': model,
                    'num_imputer': num_imputer,
                    'scaler': scaler,
                    'categorical_features': categorical_features
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
        
        # 特征重要性分析
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean()
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        logging.info("\nTop 10 important features:")
        logging.info(feature_importance.head(10))
        
        # 保存模型
        model_save_path = 'catboost_clustered_models.pkl'
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'kmeans_model': kmeans_model,
                'cluster_info': cluster_info,
                'feature_importance': feature_importance
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
                    X_test_processed, _, _, _ = preprocess_features(
                        X_test_cluster,
                        num_imputer=model_dict['num_imputer'],
                        scaler=model_dict['scaler'],
                        categorical_features=model_dict['categorical_features']
                    )
                    
                    cluster_predictions[:, i] = model.predict(X_test_processed)
                    
                except Exception as e:
                    logging.error(f"Error predicting cluster {cluster}, model {i}: {str(e)}")
                    logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                    logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
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
        
        submission_path = '10-27-release_catboost_clustered.csv'
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