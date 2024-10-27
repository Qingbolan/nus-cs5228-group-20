import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging

# Set up logging
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
                        target_encode_cols=['make', 'model'], 
                        encoding_smoothing=1.0):
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
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

def find_optimal_clusters(X, y, max_clusters=10, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
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
        logging.info(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def create_price_clusters(X, y, n_clusters, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
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
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    kmeans.feature_imputer = imputer
    
    return kmeans, price_clusters, cluster_df

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    X_processed, _, _, _, _ = preprocess_features(X, y, **preprocessors)
    
    cluster_features_df = pd.DataFrame(X_processed[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    
    cluster_features = np.column_stack([np.log1p(y) if y is not None else np.zeros(len(X)), cluster_features_clean])
    return kmeans_model.predict(cluster_features)

def train_evaluate_models(X_train, y_train, X_val, y_val, lgb_params, cb_params, gb_params, price_range='medium'):
    """训练所有模型并返回最优权重组合"""
    # 训练LightGBM
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid']
    )
    
    # 训练CatBoost
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train, 
        np.log1p(y_train),
        eval_set=(X_val, np.log1p(y_val)),
        # verbose=False
    )
    
    # 训练GradientBoosting
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, np.log1p(y_train))
    
    # 根据价格区间调整SVR参数
    if price_range == 'high':
        svr_params = {'C': 100, 'epsilon': 0.1, 'gamma': 'scale'}
    elif price_range == 'low':
        svr_params = {'C': 50, 'epsilon': 0.2, 'gamma': 'scale'}
    else:
        svr_params = {'C': 75, 'epsilon': 0.15, 'gamma': 'scale'}
    
    svr_model = SVR(**svr_params)
    svr_model.fit(X_train, np.log1p(y_train))
    
    # 在验证集上寻找最优权重组合
    lgb_pred = np.expm1(lgb_model.predict(X_val))
    cb_pred = np.expm1(cb_model.predict(X_val))
    gb_pred = np.expm1(gb_model.predict(X_val))
    svr_pred = np.expm1(svr_model.predict(X_val))
    
    best_rmse = float('inf')
    best_weights = (0.4, 0.2, 0.2, 0.2)  # 初始权重
    
    # 网格搜索最优权重组合
    for w1 in np.arange(0.3, 0.6, 0.1):  # LightGBM权重
        for w2 in np.arange(0.1, 0.3, 0.1):  # CatBoost权重
            for w3 in np.arange(0.1, 0.3, 0.1):  # GradientBoosting权重
                w4 = 1 - w1 - w2 - w3  # SVR权重
                if w4 < 0.1:  # 确保SVR至少有10%的权重
                    continue
                    
                weighted_pred = (w1 * lgb_pred + w2 * cb_pred + 
                               w3 * gb_pred + w4 * svr_pred)
                rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = (w1, w2, w3, w4)
    
    return lgb_model, cb_model, gb_model, svr_model, best_weights

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_list = []
    models = []
    
    # LightGBM参数
    params = {
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
    
    # GradientBoosting参数
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 5,
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
        
        # 确定价格区间
        median_price = cluster_info.iloc[cluster]['median']
        if median_price < cluster_info['median'].median():
            price_range = 'low'
        elif median_price > cluster_info['median'].quantile(0.75):
            price_range = 'high'
        else:
            price_range = 'medium'
            
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            # 训练模型
            lgb_model, cb_model, gb_model, svr_model, best_weights = train_evaluate_models(
                X_train_processed, y_train,
                X_val_processed, y_val,
                params, cb_params, gb_params,
                price_range
            )
            
            # 组合预测
            lgb_pred = np.expm1(lgb_model.predict(X_val_processed))
            cb_pred = np.expm1(cb_model.predict(X_val_processed))
            gb_pred = np.expm1(gb_model.predict(X_val_processed))
            svr_pred = np.expm1(svr_model.predict(X_val_processed))
            
            fold_predictions = (best_weights[0] * lgb_pred + 
                              best_weights[1] * cb_pred +
                              best_weights[2] * gb_pred +
                              best_weights[3] * svr_pred)
            
            oof_predictions[price_clusters == cluster][val_index] = fold_predictions
            
            # 记录特征重要性
            importance = lgb_model.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            logging.info(f"Best weights for cluster {cluster}, fold {fold}: LGB={best_weights[0]:.2f}, "
                        f"CB={best_weights[1]:.2f}, GB={best_weights[2]:.2f}, SVR={best_weights[3]:.2f}")
            
            # 保存模型
            cluster_models.append({
                'lgb_model': lgb_model,
                'cb_model': cb_model,
                'gb_model': gb_model,
                'svr_model': svr_model,
                'weights': best_weights,
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
    oof_mse = mean_squared_error(y, oof_predictions)
    oof_r2 = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
    
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    # 保存模型
    with open('quad_ensemble_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 预测测试集
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = predict_cluster(X_test, dummy_y_test, kmeans_model, models[0][0]['preprocessors'], features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            continue
        
        cluster_predictions = np.zeros((len(X_test_cluster), len(models[cluster])))
        
        for i, model_dict in enumerate(models[cluster]):
            try:
                X_test_processed, _, _, _, _ = preprocess_features(
                    X_test_cluster, y=None, **model_dict['preprocessors']
                )
                
                # 组合四个模型的预测
                lgb_pred = np.expm1(model_dict['lgb_model'].predict(X_test_processed))
                cb_pred = np.expm1(model_dict['cb_model'].predict(X_test_processed))
                gb_pred = np.expm1(model_dict['gb_model'].predict(X_test_processed))
                svr_pred = np.expm1(model_dict['svr_model'].predict(X_test_processed))
                
                weights = model_dict['weights']
                cluster_predictions[:, i] = (weights[0] * lgb_pred + 
                                          weights[1] * cb_pred +
                                          weights[2] * gb_pred +
                                          weights[3] * svr_pred)
                
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}, model {i}: {str(e)}")
                continue
        
        final_predictions[cluster_mask] = np.mean(cluster_predictions, axis=1)
    
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./quad_ensemble_predictions.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'quad_ensemble_predictions.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()