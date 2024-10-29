import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
from typing import List
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path):
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
            scaler = RobustScaler()
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
    """Create price-based clusters."""
    logging.info(f"Creating {n_clusters} price clusters")

    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
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

def find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
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

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # verbose_eval=False,  # 设置为False以减少输出
    )
    
    return model

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    cluster_features_scaled = kmeans_model.feature_scaler.transform(cluster_features_clean)
    
    if y is None:
        dummy_price = np.zeros(len(X))
        cluster_features = np.column_stack([dummy_price, cluster_features_scaled])
    else:
        cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
    
    return kmeans_model.predict(cluster_features)

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    oof_mse = []
    oof_r2 = []
    feature_importance_list = []
    models = []
    
    # 定义各模型的超参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'cat_smooth': 10,
        'cat_l2': 10,
    }
    
    # DART 变体的超参数
    dart_params = {
        'boosting_type': 'dart',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
    }
    
    # GOSS 变体的超参数
    goss_params = {
        'boosting_type': 'goss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'top_rate': 0.2,
        'other_rate': 0.1,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
    }
    
    # GradientBoostingRegressor 的超参数
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 15,
        'loss': 'huber',
        'random_state': 42
    }
    
    # CatBoostRegressor 的超参数
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
        # 'verbose': False
    }
    
    start_time = time.time()
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\nTraining models for Cluster {cluster}")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Cluster {cluster} - Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            # 训练 LightGBM（基准模型）
            model_lgb = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, lgb_params)
            
            # 训练 LightGBM DART 变体
            model_dart = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, dart_params)
            
            # 训练 LightGBM GOSS 变体
            model_goss = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, goss_params)
            
            # 训练 GradientBoostingRegressor
            model_gb = GradientBoostingRegressor(**gb_params)
            model_gb.fit(X_train_processed, np.log1p(y_train))
            
            # 训练 CatBoostRegressor
            model_cb = CatBoostRegressor(**cb_params)
            model_cb.fit(X_train_processed, np.log1p(y_train), eval_set=(X_val_processed, np.log1p(y_val)))
            
            # 预测
            preds_lgb = np.expm1(model_lgb.predict(X_val_processed, num_iteration=model_lgb.best_iteration))
            preds_dart = np.expm1(model_dart.predict(X_val_processed, num_iteration=model_dart.best_iteration))
            preds_goss = np.expm1(model_goss.predict(X_val_processed, num_iteration=model_goss.best_iteration))
            preds_gb = np.expm1(model_gb.predict(X_val_processed))
            preds_cb = np.expm1(model_cb.predict(X_val_processed))
            
            # 集成预测（简单平均）
            preds_ensemble = (preds_lgb + preds_dart + preds_goss + preds_gb + preds_cb) / 5
            
            oof_predictions[val_index] = preds_ensemble
            rmse = np.sqrt(mean_squared_error(y_val, preds_ensemble))
            r2 = r2_score(y_val, preds_ensemble)
            oof_mse.append(rmse ** 2)
            oof_r2.append(r2)
            logging.info(f"Cluster {cluster} - Fold {fold} RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            # 保存 LightGBM 的特征重要性
            importance = model_lgb.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            # 保存模型及其预处理器
            cluster_models.append({
                'lightgbm': model_lgb,
                'lightgbm_dart': model_dart,
                'lightgbm_goss': model_goss,
                'gradient_boosting': model_gb,
                'catboost': model_cb,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
        
        models.append(cluster_models)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse_total = mean_squared_error(y, oof_predictions)
    oof_r2_total = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse_total):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")
    
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    with open('ensemble_clustered_models_lgb_variants_avg.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 预测测试数据
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = predict_cluster(X_test, None, kmeans_model, models[0][0]['preprocessors'], features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        logging.info(f"Predicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_models = models[cluster]
        cluster_predictions = np.zeros(len(X_test_cluster))
        num_models = 0
        
        for model_dict in cluster_models:
            model_lgb = model_dict['lightgbm']
            model_dart = model_dict['lightgbm_dart']
            model_goss = model_dict['lightgbm_goss']
            model_gb = model_dict['gradient_boosting']
            model_cb = model_dict['catboost']
            preprocessors = model_dict['preprocessors']
            
            try:
                X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **preprocessors)
                
                preds_lgb = np.expm1(model_lgb.predict(X_test_processed, num_iteration=model_lgb.best_iteration))
                preds_dart = np.expm1(model_dart.predict(X_test_processed, num_iteration=model_dart.best_iteration))
                preds_goss = np.expm1(model_goss.predict(X_test_processed, num_iteration=model_goss.best_iteration))
                preds_gb = np.expm1(model_gb.predict(X_test_processed))
                preds_cb = np.expm1(model_cb.predict(X_test_processed))
                
                preds = (preds_lgb + preds_dart + preds_goss + preds_gb + preds_cb) / 5
                cluster_predictions += preds
                num_models += 1
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster} with one of the models: {str(e)}")
                logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                continue
        
        if num_models > 0:
            final_predictions[cluster_mask] += cluster_predictions / num_models
        else:
            logging.warning(f"No successful model predictions for cluster {cluster}.")
    
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./submission_ensemble_clustered_lgb_variants_avg.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'submission_ensemble_clustered_lgb_variants_avg.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()