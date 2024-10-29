import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
# import shap

# (test)RMSE Score: 19905.6043
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        cluster_prices = y[price_clusters == cluster]
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

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
    )
    
    return model

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    X_processed, _, _, _, _ = preprocess_features(X, y, **preprocessors)
    cluster_features = np.column_stack([np.log1p(y) if y is not None else np.zeros(len(X)), X_processed[features_for_clustering]])
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
    
    # GradientBoostingRegressor 的超参数
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'max_features': 'sqrt',
        'min_samples_split': 20,
        'min_samples_leaf': 15,
        'loss': 'huber',
        'random_state': 42
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
            
            # 训练 LightGBM
            model_lgb = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, lgb_params)
            
            # 训练 GradientBoostingRegressor
            model_gb = GradientBoostingRegressor(**gb_params)
            model_gb.fit(X_train_processed, np.log1p(y_train))
            
            # 预测
            preds_lgb = np.expm1(model_lgb.predict(X_val_processed, num_iteration=model_lgb.best_iteration))
            preds_gb = np.expm1(model_gb.predict(X_val_processed))
            
            # 集成预测（平均）
            preds_ensemble = (preds_lgb + preds_gb) / 2.0
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
                'gradient_boosting': model_gb,
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
    
    with open('ensemble_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 预测测试数据
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = predict_cluster(X_test, dummy_y_test, kmeans_model, models[0][0]['preprocessors'], features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    prediction_counts = np.zeros(len(X_test))
    
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
            model_gb = model_dict['gradient_boosting']
            preprocessors = model_dict['preprocessors']
            
            try:
                X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **preprocessors)
                
                preds_lgb = np.expm1(model_lgb.predict(X_test_processed, num_iteration=model_lgb.best_iteration))
                preds_gb = np.expm1(model_gb.predict(X_test_processed))
                
                preds = (preds_lgb + preds_gb) / 2.0
                cluster_predictions += preds
                num_models += 1
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster} with one of the models: {str(e)}")
                logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                continue
        
        if num_models > 0:
            final_predictions[cluster_mask] += cluster_predictions
            prediction_counts[cluster_mask] += num_models
        else:
            logging.warning(f"No successful model predictions for cluster {cluster}.")
    
    # 避免除以零
    prediction_counts[prediction_counts == 0] = 1
    final_predictions /= prediction_counts
    
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./1026_ensemble_clustered_2_model.csv', index=False)
    logging.info("Predictions complete. Submission file saved as '1026_ensemble_clustered_2_model.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()