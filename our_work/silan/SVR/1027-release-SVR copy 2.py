import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
import pickle
import logging
import time
import os
from typing import Tuple, List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns}")
    
    return X, y

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                        target_encoder=None, scaler=None, 
                        target_encode_cols=[], 

                        encoding_smoothing=1.0):
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    

    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)
        else:
            X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)


    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        else:
            X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        
        if target_encode_features:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = pd.DataFrame(target_encoder.fit_transform(X[target_encode_features], y), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
            else:
                X[target_encode_features] = pd.DataFrame(target_encoder.transform(X[target_encode_features]), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
        
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def create_price_clusters(X, y, n_clusters, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(X[features_for_clustering], np.linspace(0, 100, n_clusters), axis=0)
    ])

    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=10, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])

    price_clusters = kmeans.fit_predict(cluster_features)
    
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_mask = price_clusters == cluster
        cluster_prices = y[cluster_mask]

        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    return kmeans, price_clusters, cluster_df

def find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    from sklearn.metrics import silhouette_score
    cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters


def predict_cluster(X, kmeans_model, y=None, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """预测聚类"""
    if y is None:
        # 对于测试集，使用虚拟的价格值
        dummy_y = np.zeros(len(X))
        cluster_features = np.column_stack([dummy_y, X[features_for_clustering]])
    else:
        cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])
    
    return kmeans_model.predict(cluster_features)
def create_model_ensemble(cluster_stats: Dict) -> List[tuple]:
    """创建优化的SVR模型"""
    # 动态调整SVR参数
    price_range = cluster_stats['max'] - cluster_stats['min']
    sample_count = cluster_stats['count']
    median_price = cluster_stats['median']
    
    # 基于价格范围和样本量动态调整参数
    if median_price < 100000:  # 低价区间
        svr_params = {
            'C': 150,
            'epsilon': 0.05,
            'kernel': 'rbf',
            'gamma': 'scale',
            'cache_size': 2000,
            'tol': 0.001
        }
    elif median_price < 1500000:  # 中价区间
        svr_params = {
            'C': 100,
            'epsilon': 0.1,
            'kernel': 'rbf',
            'gamma': 'scale',
            'cache_size': 2000,
            'tol': 0.001
        }
    else:  # 高价区间
        svr_params = {
            'C': 80,
            'epsilon': 0.15,
            'kernel': 'rbf',
            'gamma': 'scale',
            'cache_size': 2000,
            'tol': 0.001
        }
    
    # 样本量调整
    if sample_count < 1000:
        svr_params['C'] *= 1.2
        svr_params['epsilon'] *= 0.8
    
    # 创建模型列表（只包含SVR）
    models = [('svr', SVR(**svr_params))]
    
    return models

def ensemble_predict(models, X, target_encode_cols):
    """集成预测（简化版）"""
    all_predictions = []
    
    for fold_models in models:
        fold_predictions = []
        for model_dict in fold_models:
            model = model_dict['model']
            preprocessors = model_dict['preprocessors']
            
            # 预处理特征
            X_processed, _, _, _, _ = preprocess_features(
                X, 
                num_imputer=preprocessors['num_imputer'],
                cat_imputer=preprocessors['cat_imputer'],
                target_encoder=preprocessors['target_encoder'],
                scaler=preprocessors['scaler'],
                target_encode_cols=target_encode_cols
            )
            
            # 预测并反转缩放
            pred_scaled = model.predict(X_processed)
            pred = preprocessors['y_scaler'].inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).ravel()
            
            # 确保预测值在合理范围内
            pred = np.clip(pred, 700, 2900000)
            
            fold_predictions.append(pred)
        
        # 计算这个fold的平均预测
        fold_mean = np.mean(fold_predictions, axis=0)
        all_predictions.append(fold_mean)
    
    # 使用中位数集成
    final_predictions = np.median(all_predictions, axis=0)
    return final_predictions

def post_process_predictions(predictions: np.ndarray, cluster_stats: Dict) -> np.ndarray:
    """改进的后处理策略"""
    # 基于聚类统计的合理范围
    min_price = max(700, cluster_stats['min'] * 0.9)
    max_price = min(2900000, cluster_stats['max'] * 1.1)
    median_price = cluster_stats['median']
    
    # 平滑处理极端值
    for i in range(len(predictions)):
        if predictions[i] < min_price:
            predictions[i] = min_price + (predictions[i] - min_price) * 0.5
        elif predictions[i] > max_price:
            predictions[i] = max_price - (max_price - predictions[i]) * 0.5
        
        # 价格合理性检查
        if abs(predictions[i] - median_price) > 2 * (max_price - min_price):
            predictions[i] = median_price + (predictions[i] - median_price) * 0.3
    
    # 最终裁剪
    predictions = np.clip(predictions, 700, 2900000)
    return np.round(predictions)

def train_cluster_models(X_cluster, y_cluster, cluster_stats):
    """改进的训练过程"""
    models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    target_encode_cols = ['manufacturer', 'model', 'type', 'transmission']
    
    # 对目标值进行缩放
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_cluster.values.reshape(-1, 1)).ravel()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster)):
        logging.info(f"Training fold {fold + 1}")
        
        X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
        y_val_original = y_cluster.iloc[val_idx]
        
        # 特征预处理
        X_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(
            X_train, y_cluster.iloc[train_idx], None, None, None, None,
            target_encode_cols=target_encode_cols
        )
        
        # 训练SVR模型
        fold_models = []
        for model_name, model in create_model_ensemble(cluster_stats):
            # 训练模型
            model.fit(X_processed, y_train)
            
            # 验证
            X_val_processed, _, _, _, _ = preprocess_features(
                X_val, y_val_original,
                num_imputer=num_imputer,
                cat_imputer=cat_imputer,
                target_encoder=target_encoder,
                scaler=scaler,
                target_encode_cols=target_encode_cols
            )
            
            # 预测并评估
            y_val_pred = y_scaler.inverse_transform(
                model.predict(X_val_processed).reshape(-1, 1)
            ).ravel()
            
            y_val_pred = np.clip(y_val_pred, cluster_stats['min'], cluster_stats['max'])
            
            rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
            r2 = r2_score(y_val_original, y_val_pred)
            
            logging.info(f"Fold {fold + 1} - {model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            fold_models.append({
                'model': model,

                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler,
                    'y_scaler': y_scaler
                },
                'performance': {
                    'rmse': rmse,
                    'r2': r2
                }
            })
        
        models.append(fold_models)
    
    return models
def main():
    try:
        np.random.seed(42)
        start_time = time.time()
        
        # 加载数据
        X_train, y_train = load_and_preprocess_data('l2_train.csv')
        X_test, _ = load_and_preprocess_data('l2_test.csv')
        
        # 寻找最优聚类数
        optimal_clusters = find_optimal_clusters(X_train, y_train)
        
        # 创建聚类
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X_train, y_train, n_clusters=optimal_clusters
        )
        
        # 训练模型
        cluster_models = []
        target_encode_cols = ['manufacturer', 'model', 'type', 'transmission']
        
        for cluster in range(optimal_clusters):
            mask = price_clusters == cluster
            X_cluster = X_train[mask]
            y_cluster = y_train[mask]
            
            cluster_stats = {
                'min': y_cluster.min(),
                'max': y_cluster.max(),
                'median': y_cluster.median(),
                'count': len(y_cluster)
            }
            
            logging.info(f"\nTraining models for Cluster {cluster}")
            logging.info(f"Cluster stats: {cluster_stats}")
            
            models = train_cluster_models(X_cluster, y_cluster, cluster_stats)
            cluster_models.append(models)
        
        # 保存模型
        with open('ensemble_models_svr_mlp.pkl', 'wb') as f:
            pickle.dump({
                'models': cluster_models,
                'kmeans': kmeans_model,
                'cluster_info': cluster_info
            }, f)
        
        # 预测测试集
        test_clusters = predict_cluster(X_test, kmeans_model)
        final_predictions = np.zeros(len(X_test))
        
        for cluster in range(optimal_clusters):
            cluster_mask = test_clusters == cluster
            if not any(cluster_mask):
                continue
            
            X_test_cluster = X_test[cluster_mask]
            predictions = ensemble_predict(
                cluster_models[cluster], 
                X_test_cluster,
                target_encode_cols
            )
            
            # 后处理预测结果
            cluster_stats = cluster_info.iloc[cluster]
            min_price = max(700, cluster_stats['min'] * 0.8)
            max_price = min(2900000, cluster_stats['max'] * 1.2)
            predictions = np.clip(predictions, min_price, max_price)
            
            final_predictions[cluster_mask] = predictions
        
        # 最终的后处理
        final_predictions = np.clip(final_predictions, 700, 2900000)
        final_predictions = np.round(final_predictions).astype(int)
        
        # 保存预测结果
        submission = pd.DataFrame({
            'Id': range(len(final_predictions)),
            'Predicted': final_predictions
        })
        
        submission.to_csv('ensemble_svr_mlp_predictions.csv', index=False)
        
        # 打印训练时间和预测统计
        end_time = time.time()
        logging.info("\nTraining and prediction completed!")
        logging.info(f"Total time taken: {(end_time - start_time)/60:.2f} minutes")
        logging.info(f"Prediction statistics:")
        logging.info(f"Min: {final_predictions.min()}")
        logging.info(f"Max: {final_predictions.max()}")
        logging.info(f"Mean: {final_predictions.mean():.2f}")
        logging.info(f"Median: {np.median(final_predictions):.2f}")
        
        # 保存完整的运行配置和结果
        run_info = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_time': end_time - start_time,
            'optimal_clusters': optimal_clusters,
            'cluster_info': cluster_info.to_dict('records'),
            'prediction_stats': {
                'min': float(final_predictions.min()),
                'max': float(final_predictions.max()),
                'mean': float(final_predictions.mean()),
                'median': float(np.median(final_predictions))
            }
        }
        
        with open('run_info_svr_mlp.json', 'w') as f:
            import json
            json.dump(run_info, f, indent=4)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()