import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
        # early_stopping_rounds=50,
        # verbose_eval=False
    )
    
    return model

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-27-silan/train_cleaned.csv')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
        
    # 定义目标编码的列
    target_encode_cols = ['some_categorical_column1', 'some_categorical_column2']  # 根据实际情况修改
    
    # 定义5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 初始化存储Level 1特征
    oof_lgb = np.zeros(len(X))
    oof_gb = np.zeros(len(X))
    oof_cb = np.zeros(len(X))
    
    # 初始化存储Level 1预测特征
    level1_features = pd.DataFrame(index=X.index)
    
    # 测试集的预测
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-27-silan/test_cleaned.csv')
    
    # 初始化存储测试集的预测结果
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_gb = np.zeros(len(X_test))
    test_preds_cb = np.zeros(len(X_test))
    
    # 存储每折的预处理器
    preprocessors = []
    
    # 提前处理测试集的聚类信息
    X_test_processed_full, global_num_imputer, global_cat_imputer, global_target_encoder, global_scaler = \
        preprocess_features(X_test, y=None, target_encode_cols=target_encode_cols)
    
    # 为测试集创建聚类标签
    median_price = np.full(len(X_test), np.median(y))
    test_features_for_clustering = np.column_stack([
        np.log1p(median_price),
        X_test_processed_full[features_for_clustering]
    ])
    test_clusters = kmeans_model.predict(test_features_for_clustering)
    
    # 基础模型的参数
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
        'verbose': False
    }
    
    # Store models for each cluster
    models = []
    
    start_time = time.time()
    
    # 进行KFold交叉验证
    for cluster in range(len(cluster_info)):
        logging.info(f"\nTraining models for Cluster {cluster}")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Cluster {cluster} - Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(
                X_train, y_train, target_encode_cols=target_encode_cols
            )
            X_val_processed, _, _, _, _ = preprocess_features(
                X_val, y_val,
                num_imputer=num_imputer,
                cat_imputer=cat_imputer,
                target_encoder=target_encoder,
                scaler=scaler,
                target_encode_cols=target_encode_cols
            )
            
            # 训练 LightGBM
            model_lgb = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, lgb_params)
            
            # 训练 GradientBoostingRegressor
            model_gb = GradientBoostingRegressor(**gb_params)
            model_gb.fit(X_train_processed, np.log1p(y_train))
            
            # 训练 CatBoostRegressor
            model_cb = CatBoostRegressor(**cb_params)
            model_cb.fit(X_train_processed, np.log1p(y_train), eval_set=(X_val_processed, np.log1p(y_val)))
            
            # 预测验证集
            preds_lgb = np.expm1(model_lgb.predict(X_val_processed, num_iteration=model_lgb.best_iteration))
            preds_gb = np.expm1(model_gb.predict(X_val_processed))
            preds_cb = np.expm1(model_cb.predict(X_val_processed))
            
            # 保存验证集的预测结果
            oof_lgb[val_index] = preds_lgb
            oof_gb[val_index] = preds_gb
            oof_cb[val_index] = preds_cb
            
            # 处理当前聚类的测试集样本
            X_test_cluster = X_test[test_clusters == cluster].copy()
            if len(X_test_cluster) > 0:
                X_test_processed, _, _, _, _ = preprocess_features(
                    X_test_cluster,
                    y=None,
                    num_imputer=num_imputer,
                    cat_imputer=cat_imputer,
                    target_encoder=target_encoder,
                    scaler=scaler,
                    target_encode_cols=target_encode_cols
                )
                
                # 获取当前聚类测试集样本的索引
                test_cluster_indices = np.where(test_clusters == cluster)[0]
                
                # 更新对应索引的预测结果
                test_preds_lgb[test_cluster_indices] += np.expm1(
                    model_lgb.predict(X_test_processed, num_iteration=model_lgb.best_iteration)
                ) / n_splits
                test_preds_gb[test_cluster_indices] += np.expm1(
                    model_gb.predict(X_test_processed)
                ) / n_splits
                test_preds_cb[test_cluster_indices] += np.expm1(
                    model_cb.predict(X_test_processed)
                ) / n_splits
            else:
                logging.warning(f"No test samples for Cluster {cluster} in Fold {fold}.")
            
            # 计算本折的RMSE和R2
            rmse_lgb = np.sqrt(mean_squared_error(y_val, preds_lgb))
            r2_lgb = r2_score(y_val, preds_lgb)
            rmse_gb = np.sqrt(mean_squared_error(y_val, preds_gb))
            r2_gb = r2_score(y_val, preds_gb)
            rmse_cb = np.sqrt(mean_squared_error(y_val, preds_cb))
            r2_cb = r2_score(y_val, preds_cb)
            logging.info(f"Cluster {cluster} - Fold {fold} LightGBM RMSE: {rmse_lgb:.4f}, R2: {r2_lgb:.4f}")
            logging.info(f"Cluster {cluster} - Fold {fold} GradientBoosting RMSE: {rmse_gb:.4f}, R2: {r2_gb:.4f}")
            logging.info(f"Cluster {cluster} - Fold {fold} CatBoost RMSE: {rmse_cb:.4f}, R2: {r2_cb:.4f}")
            
            # 保存模型及其预处理器
            cluster_models.append({
                'lightgbm': model_lgb,
                'catboost': model_cb,
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
    logging.info(f"\nLevel 1 Training completed in {elapsed_time/60:.2f} minutes")
    
    # 创建Level 1的特征矩阵
    level1_features['lgb'] = oof_lgb
    level1_features['gb'] = oof_gb
    level1_features['cb'] = oof_cb
    
    # Level 2 - 堆叠元模型训练
    logging.info("Training Level 2 Meta-Model (XGBoost)")
    
    meta_oof = np.zeros(len(X))
    meta_test_preds = np.zeros(len(X_test))
    
    kf_meta = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf_meta.split(level1_features), 1):
        logging.info(f"Meta-Fold {fold}")
        
        X_meta_train, X_meta_val = level1_features.iloc[train_idx].copy(), level1_features.iloc[val_idx].copy()
        y_meta_train, y_meta_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        
        # 训练XGBoost
        meta_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            verbosity=0
        )
        meta_model.fit(
            X_meta_train, np.log1p(y_meta_train),
            eval_set=[(X_meta_train, np.log1p(y_meta_train)), (X_meta_val, np.log1p(y_meta_val))],
            # early_stopping_rounds=50,
            # verbose=False
        )
        
        # 预测验证集
        preds_meta = np.expm1(meta_model.predict(X_meta_val))
        meta_oof[val_idx] = preds_meta
        
        # 预测测试集
        meta_test_preds += np.expm1(meta_model.predict(pd.DataFrame({
            'lgb': test_preds_lgb,
            'gb': test_preds_gb,
            'cb': test_preds_cb
        }))) / n_splits
        
        # 计算本折的RMSE和R2
        rmse_meta = np.sqrt(mean_squared_error(y_meta_val, preds_meta))
        r2_meta = r2_score(y_meta_val, preds_meta)
        logging.info(f"Meta-Fold {fold} - XGBoost RMSE: {rmse_meta:.4f}, R2: {r2_meta:.4f}")
    
    # 计算Level 1 + Level 2的整体性能
    final_oof = meta_oof  # 使用元模型的预测作为最终预测
    final_oof = post_process_predictions(final_oof)
    oof_mse_total = mean_squared_error(y, final_oof)
    oof_r2_total = r2_score(y, final_oof)
    logging.info(f"Out-of-fold RMSE (Stacking): {np.sqrt(oof_mse_total):.4f}")
    logging.info(f"Out-of-fold R2 (Stacking): {oof_r2_total:.4f}")
    
    # 保存所有Level 1模型、元模型及预处理器
    with open('stacking_models.pkl', 'wb') as f:
        pickle.dump({
            'base_models': models,
            'meta_model': meta_model,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 最终预测
    final_test_preds = meta_test_preds  # 使用元模型的预测作为最终预测
    final_test_preds = post_process_predictions(final_test_preds)
    
    submission = pd.DataFrame({
        'Id': range(len(final_test_preds)),
        'Predicted': np.round(final_test_preds).astype(int)
    })
    
    submission.to_csv('./submission_stacking_clustered_optimized.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'submission_stacking_clustered_optimized.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_test_preds.min()}")
    logging.info(f"Maximum: {final_test_preds.max()}")
    logging.info(f"Mean: {final_test_preds.mean()}")
    logging.info(f"Median: {np.median(final_test_preds)}")

if __name__ == '__main__':
    main()