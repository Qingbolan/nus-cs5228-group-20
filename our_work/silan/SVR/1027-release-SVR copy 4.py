import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_training.log'),
        logging.StreamHandler()
    ]
)

def create_interaction_features(X):
    """创建交互特征"""
    X = X.copy()
    
    # 基础交互特征
    X['price_efficiency'] = X['coe'] / (X['depreciation'] + 1e-5)
    X['value_ratio'] = X['dereg_value'] / (X['coe'] + 1e-5)
    X['total_cost'] = X['coe'] + X['depreciation']
    
    # 多项式特征
    for col in ['depreciation', 'coe', 'dereg_value']:
        X[f'{col}_squared'] = X[col] ** 2
        X[f'{col}_cubed'] = X[col] ** 3
        
    # 特征组合
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if col1 != col2:
                X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
                X[f'{col1}_times_{col2}'] = X[col1] * X[col2]
                X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-5)
    
    return X

def remove_outliers(X, y, n_sigma=3):
    """移除异常值"""
    X = X.copy()
    y = y.copy()
    
    # 计算z-score
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < n_sigma
    
    return X[mask], y[mask]

def analyze_model_correlation(X_val, y_val, lgb_model, cb_model, svr_model):
    """分析模型相关性"""
    lgb_pred = np.expm1(lgb_model.predict(X_val))
    cb_pred = np.expm1(cb_model.predict(X_val)) 
    svr_pred = np.expm1(svr_model.predict(X_val))
    
    preds = pd.DataFrame({
        'lgb': lgb_pred,
        'catboost': cb_pred,
        'svr': svr_pred,
        'actual': y_val
    })
    
    correlation = preds.corr()
    logging.info(f"Model predictions correlation:\n{correlation}")
    
    return correlation

def get_model_params(price_range, correlation_info=None):
    """根据价格区间和相关性信息获取优化的模型参数"""
    
    # 基础参数
    base_lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
    }
    
    base_cb_params = {
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': 42,
        'verbose': False
    }
    
    # 根据价格区间调整参数
    if price_range == 'high':
        lgb_params = {
            **base_lgb_params,
            'learning_rate': 0.03,
            'num_leaves': 41,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.8,
            'max_depth': 8,
            'min_child_samples': 30,
            'cat_smooth': 8,
            'cat_l2': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        cb_params = {
            **base_cb_params,
            'iterations': 4000,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 8,
            'min_data_in_leaf': 30,
            'random_strength': 0.8,
            'bagging_temperature': 0.3
        }
        
    elif price_range == 'low':
        lgb_params = {
            **base_lgb_params,
            'learning_rate': 0.07,
            'num_leaves': 25,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.6,
            'max_depth': 6,
            'min_child_samples': 15,
            'cat_smooth': 15,
            'cat_l2': 15,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2
        }
        
        cb_params = {
            **base_cb_params,
            'iterations': 2500,
            'learning_rate': 0.07,
            'depth': 5,
            'l2_leaf_reg': 12,
            'min_data_in_leaf': 15,
            'random_strength': 0.3,
            'bagging_temperature': 0.1
        }
        
    else:  # medium
        lgb_params = {
            **base_lgb_params,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'max_depth': 7,
            'min_child_samples': 20,
            'cat_smooth': 10,
            'cat_l2': 10,
            'reg_alpha': 0.15,
            'reg_lambda': 0.15
        }
        
        cb_params = {
            **base_cb_params,
            'iterations': 3000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2
        }
    
    # 如果有相关性信息，根据相关性调整参数
    if correlation_info is not None:
        lgb_cat_corr = abs(correlation_info.loc['lgb', 'catboost'])
        if lgb_cat_corr > 0.9:  # 相关性过高
            # 增加模型差异性
            lgb_params['feature_fraction'] *= 0.8
            cb_params['random_strength'] *= 1.5
    
    return lgb_params, cb_params

def adaptive_weighted_ensemble(predictions_dict, error_metrics, price_range):
    """自适应权重集成"""
    weights = {}
    
    # 基于误差的基础权重
    for model in predictions_dict.keys():
        error = error_metrics[model]
        weight = 1 / (error + 1e-10)
        weights[model] = weight
    
    # 根据价格区间调整权重
    if price_range == 'high':
        weights['lgb'] *= 0.8  # 降低LightGBM权重
        weights['svr'] *= 1.2  # 提高SVR权重
    elif price_range == 'low':
        weights['lgb'] *= 1.2  # 提高LightGBM权重
        weights['svr'] *= 0.8  # 降低SVR权重
    
    # 归一化权重
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # 加权组合预测
    final_pred = np.zeros_like(predictions_dict[list(predictions_dict.keys())[0]])
    for model, pred in predictions_dict.items():
        final_pred += weights[model] * pred
        
    return final_pred, weights

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    # 创建交互特征
    X = create_interaction_features(X)
    
    logging.info(f"Loaded data from {file_path}, shape after preprocessing: {X.shape}")
    
    return X, y

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                       target_encoder=None, scaler=None, 
                       target_encode_cols=['make', 'model'], 
                       encoding_smoothing=1.0):
    """特征预处理"""
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 数值特征处理
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                         columns=numeric_features, 
                                         index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                         columns=numeric_features, 
                                         index=X.index)

    # 处理分类特征
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
        
        # 目标编码
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
        
        # One-hot编码
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def train_evaluate_models(X_train, y_train, X_val, y_val, price_range='medium'):
    """训练并评估模型"""
    # 获取针对该价格区间优化的参数
    lgb_params, cb_params = get_model_params(price_range)
    
    # 训练LightGBM
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=50,
        # verbose_eval=False
    )
    
    # 训练CatBoost
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train, 
        np.log1p(y_train),
        eval_set=(X_val, np.log1p(y_val)),
        # verbose=False
    )
    
    # 设置SVR参数
    if price_range == 'high':
        svr_params = {'C': 100, 'epsilon': 0.1, 'gamma': 'scale'}
    elif price_range == 'low':
        svr_params = {'C': 50, 'epsilon': 0.2, 'gamma': 'scale'}
    else:
        svr_params = {'C': 75, 'epsilon': 0.15, 'gamma': 'scale'}
    
    svr_model = SVR(**svr_params)
    svr_model.fit(X_train, np.log1p(y_train))
    
    # 分析模型相关性
    correlation = analyze_model_correlation(X_val, y_val, lgb_model, cb_model, svr_model)
    
    # 获取各模型在验证集上的预测和误差
    lgb_pred = np.expm1(lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration))
    cb_pred = np.expm1(cb_model.predict(X_val))
    svr_pred = np.expm1(svr_model.predict(X_val))
    
    predictions_dict = {
        'lgb': lgb_pred,
        'catboost': cb_pred,
        'svr': svr_pred
    }
    
    error_metrics = {
        'lgb': mean_squared_error(y_val, lgb_pred),
        'catboost': mean_squared_error(y_val, cb_pred),
        'svr': mean_squared_error(y_val, svr_pred)
    }
    
    # 使用自适应权重集成
    final_pred, weights = adaptive_weighted_ensemble(predictions_dict, error_metrics, price_range)
    
    logging.info(f"Price range: {price_range}, Best weights: {weights}")
    
    return lgb_model, cb_model, svr_model, weights, correlation

def find_optimal_clusters(X, y, max_clusters=10, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """寻找最优聚类数量"""
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    # 添加价格信息进行聚类
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    
    silhouette_scores = []
    inertias = []  # 肘部法则
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
        
        logging.info(f"n_clusters = {n_clusters}, silhouette = {silhouette_avg:.4f}, inertia = {inertia:.4f}")
    
    # 综合考虑轮廓系数和肘部法则
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    
    return optimal_clusters

def create_price_clusters(X, y, n_clusters, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """创建价格聚类"""
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    # 使用价格分位数初始化聚类中心
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(cluster_features_clean, np.linspace(0, 100, n_clusters), axis=0)
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=3, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    price_clusters = kmeans.fit_predict(cluster_features)
    
    # 收集聚类统计信息
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'mean': cluster_prices.mean(),
            'std': cluster_prices.std(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    kmeans.feature_imputer = imputer
    
    return kmeans, price_clusters, cluster_df

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """预测聚类"""
    X_processed, _, _, _, _ = preprocess_features(X, y, **preprocessors)
    
    cluster_features_df = pd.DataFrame(X_processed[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    
    cluster_features = np.column_stack([np.log1p(y) if y is not None else np.zeros(len(X)), cluster_features_clean])
    return kmeans_model.predict(cluster_features)

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    """后处理预测结果"""
    # 处理异常值
    predictions = np.where(predictions < 0, 0, predictions)
    predictions = np.where(np.isnan(predictions), 0, predictions)
    
    # 裁剪到合理范围
    predictions = np.clip(predictions, min_price, max_price)
    
    return predictions

def main():
    """主函数"""
    np.random.seed(42)
    
    # 加载数据
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    # 移除异常值
    X, y = remove_outliers(X, y, n_sigma=3)
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最优聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    
    # 创建价格聚类
    kmeans_model, price_clusters, cluster_info = create_price_clusters(
        X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering
    )
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_list = []
    models = []
    
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
            logging.info(f"Training fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            # 数据预处理
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            # 训练模型
            lgb_model, cb_model, svr_model, weights, correlation = train_evaluate_models(
                X_train_processed, y_train,
                X_val_processed, y_val,
                price_range
            )
            
            # 记录特征重要性
            importance = lgb_model.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({
                'feature': X_train_processed.columns,
                'importance': importance
            })
            feature_importance_list.append(feature_importance)
            
            # 保存模型
            cluster_models.append({
                'lgb_model': lgb_model,
                'cb_model': cb_model,
                'svr_model': svr_model,
                'weights': weights,
                'correlation': correlation,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
            
            # 生成OOF预测
            val_preds = {
                'lgb': np.expm1(lgb_model.predict(X_val_processed)),
                'catboost': np.expm1(cb_model.predict(X_val_processed)),
                'svr': np.expm1(svr_model.predict(X_val_processed))
            }
            
            oof_predictions[price_clusters == cluster][val_index] = sum(
                w * val_preds[m] for m, w in weights.items()
            )
        
        models.append(cluster_models)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    # 评估OOF性能
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse = mean_squared_error(y, oof_predictions)
    oof_r2 = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
    
    # 特征重要性分析
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    # 保存模型
    with open('improved_ensemble_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info,
            'feature_importance': feature_importance
        }, f)
    
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
        
        cluster_predictions = []
        
        for model_dict in models[cluster]:
            try:
                X_test_processed, _, _, _, _ = preprocess_features(
                    X_test_cluster, y=None, **model_dict['preprocessors']
                )
                
                # 获取每个模型的预测
                predictions = {
                    'lgb': np.expm1(model_dict['lgb_model'].predict(X_test_processed)),
                    'catboost': np.expm1(model_dict['cb_model'].predict(X_test_processed)),
                    'svr': np.expm1(model_dict['svr_model'].predict(X_test_processed))
                }
                
                # 使用保存的权重进行加权
                weighted_pred = sum(w * predictions[m] for m, w in model_dict['weights'].items())
                cluster_predictions.append(weighted_pred)
                
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}: {str(e)}")
                continue
        
        if cluster_predictions:
            final_predictions[cluster_mask] = np.mean(cluster_predictions, axis=0)
    
    # 后处理预测结果
    final_predictions = post_process_predictions(final_predictions)
    
    # 保存预测结果
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('improved_ensemble_predictions.csv', index=False)
    
    # 输出预测统计信息
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()