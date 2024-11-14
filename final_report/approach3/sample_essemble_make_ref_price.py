import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
import logging
from joblib import Parallel, delayed
from dotenv import load_dotenv

# 确保自定义模块路径正确
from data.rmse_calculator import calculate_rmse

load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_library_versions():
    """
    记录使用的库的版本信息。
    """
    import sklearn
    import lightgbm as lgb
    import xgboost as xgb

    logging.info(f"scikit-learn version: {sklearn.__version__}")
    logging.info(f"LightGBM version: {lgb.__version__}")
    logging.info(f"XGBoost version: {xgb.__version__}")

def load_and_preprocess_data(file_path):
    """
    加载数据。
    注意：根据您的指示，不进行任何预处理。
    """
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns}")
    
    return X, y

def create_price_clusters(X, y, n_clusters, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """
    使用 KMeans 创建价格聚类。
    仅使用指定的特征进行聚类，不包含目标变量 y。
    """
    cluster_features = X[features_for_clustering].values  # 使用 NumPy 数组，避免特征名称问题
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
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

def find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """
    使用轮廓系数寻找最佳聚类数量。
    仅使用指定的特征进行聚类，不包含目标变量 y。
    """
    from sklearn.metrics import silhouette_score
    cluster_features = X[features_for_clustering].values  # 使用 NumPy 数组
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    if silhouette_scores:
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        logging.info(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    else:
        logging.warning("Could not determine optimal clusters. Defaulting to 2.")
        return 2

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    """
    训练并评估 LightGBM 模型。
    使用 scikit-learn API 的 LGBMRegressor。
    """
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, np.log1p(y_train),
        eval_set=[(X_val, np.log1p(y_val))],
        # early_stopping_rounds=50,
        # verbose=False
    )
    return model

def train_gb_regressor(X_train, y_train, **params):
    """
    训练 GradientBoostingRegressor 模型。
    """
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, np.log1p(y_train))
    return model

def train_xgboost_regressor(X_train, y_train, X_val, y_val, params):
    """
    训练并评估 XGBoost 模型。
    使用 scikit-learn API 的 XGBRegressor。
    """
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, np.log1p(y_train),
        eval_set=[(X_val, np.log1p(y_val))],
        # early_stopping_rounds=50,
        verbose=False
    )
    return model

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    """
    后处理预测结果，限制在合理范围内。
    """
    return np.clip(predictions, min_price, max_price)

def assign_test_clusters(kmeans_model, X_test, features_for_clustering):
    """
    使用训练时的 KMeans 模型为测试数据分配聚类。
    """
    cluster_features_test = X_test[features_for_clustering].values
    test_clusters = kmeans_model.predict(cluster_features_test)
    return test_clusters

def process_cluster(
    cluster: int,
    X: pd.DataFrame,
    y: pd.Series,
    price_clusters: np.ndarray,
    cluster_info: list,
    n_splits: int,
    lgb_params: dict,
    gb_params: dict,
    xgb_params: dict,
    features_for_clustering: list
) -> dict:
    """
    处理单个聚类的模型训练和评估。
    """
    logging.info(f"\nTraining models for Cluster {cluster}")
    X_cluster = X[price_clusters == cluster]
    y_cluster = y[price_clusters == cluster]
    
    if len(X_cluster) == 0:
        logging.warning(f"No data in Cluster {cluster}. Skipping.")
        return {
            'cluster': cluster,
            'models': [],
            'val_indices': [],
            'predictions': [],
            'oof_mse': [],
            'oof_r2': [],
            'feature_importance': []
        }
    
    # 在子进程内部创建 KFold 对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cluster_models = []
    val_indices = []
    predictions = []
    cluster_oof_mse = []
    cluster_oof_r2 = []
    cluster_feature_importance = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
        logging.info(f"Cluster {cluster} - Fold {fold}")
        
        X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
        y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
        
        # 并行训练三个模型
        try:
            models = Parallel(n_jobs=3, backend='loky')([
                delayed(train_evaluate_lightgbm)(X_train, y_train, X_val, y_val, lgb_params),
                delayed(train_gb_regressor)(X_train, y_train, **gb_params),
                delayed(train_xgboost_regressor)(X_train, y_train, X_val, y_val, xgb_params)
            ])
            
            model_lgb, model_gb, model_xgb = models
        except Exception as e:
            logging.error(f"Error training models for Cluster {cluster} Fold {fold}: {str(e)}")
            continue
        
        # 预测 LightGBM
        try:
            preds_lgb = np.expm1(model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration_))
        except Exception as e:
            logging.error(f"Error predicting with LightGBM for Cluster {cluster} Fold {fold}: {str(e)}")
            preds_lgb = np.zeros(len(X_val))
        
        # 预测 GradientBoostingRegressor
        try:
            preds_gb = np.expm1(model_gb.predict(X_val))
        except Exception as e:
            logging.error(f"Error predicting with GradientBoostingRegressor for Cluster {cluster} Fold {fold}: {str(e)}")
            preds_gb = np.zeros(len(X_val))
        
        # 预测 XGBoost
        try:
            preds_xgb = np.expm1(model_xgb.predict(X_val))
        except Exception as e:
            logging.error(f"Error predicting with XGBoost for Cluster {cluster} Fold {fold}: {str(e)}")
            preds_xgb = np.zeros(len(X_val))
        
        # 集成预测（平均）
        preds_ensemble = (preds_lgb + preds_gb + preds_xgb) / 3.0
        val_indices.append(val_index)
        predictions.append(preds_ensemble)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds_ensemble))
        r2 = r2_score(y_val, preds_ensemble)
        cluster_oof_mse.append(rmse ** 2)
        cluster_oof_r2.append(r2)
        logging.info(f"Cluster {cluster} - Fold {fold} RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # 保存 LightGBM 的特征重要性
        try:
            importance = model_lgb.feature_importances_
            feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': importance})
            cluster_feature_importance.append(feature_importance)
        except Exception as e:
            logging.error(f"Error extracting feature importance for Cluster {cluster} Fold {fold}: {str(e)}")
        
        # 保存模型及其预处理器（预处理器为空，因为不进行预处理）
        cluster_models.append({
            'lightgbm': model_lgb,
            'gradient_boosting': model_gb,
            'xgboost': model_xgb,
            'preprocessors': None  # 无预处理器
        })
    
    return {
        'cluster': cluster,
        'models': cluster_models,
        'val_indices': val_indices,        # 作为列表返回
        'predictions': predictions,        # 作为列表返回
        'oof_mse': cluster_oof_mse,
        'oof_r2': cluster_oof_r2,
        'feature_importance': cluster_feature_importance
    }

def main():
    np.random.seed(42)
    
    # 记录库版本
    log_library_versions()
    

    # 从环境变量中获取文件路径，提供默认值以防环境变量未设置
    train_data_path = os.getenv('FINAL_REPORT_TRAIN_DATA', 'preprocessing/2024-10-21-silan/train_cleaned.csv')
    test_data_path = os.getenv('FINAL_REPORT_TEST_DATA', 'preprocessing/2024-10-21-silan/test_cleaned.csv')
    base_models_save_path = os.getenv('FINAL_REPORT_APPROACH_Three_No_ref_Price_WEIGHT_PATH', 'final_report\\approach3\\initial_essemble_models.pkl')
    
    with_ref_price_train_data_path = os.getenv('FINAL_REPORT_APPROACH_three_With_ref_price_train_DATA', 'final_report\\approach3\\ref_price_train.csv')
    with_ref_price_test_data_path = os.getenv('FINAL_REPORT_APPROACH_three_With_ref_price_test_DATA', 'final_report\\approach3\\ref_price_test.csv')
    with_ref_models_save_path = os.getenv('FINAL_REPORT_APPROACH_three_With_ref_Price_WEIGHT_PATH', 'final_report\\approach3\\ref_price_models.pkl')
    submission_path = os.getenv('FINAL_REPORT_APPROACH_three_SUBMISSION_PATH', 'final_report\\approach3\\final_report_submission.csv')

    
    logging.info(f"Loading training data from: {train_data_path}")
    X, y = load_and_preprocess_data(train_data_path)
    
    if y is None:
        logging.error("目标变量 'price' 不存在于训练数据中。请检查数据。")
        return
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=features_for_clustering)
    
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    n_splits = 5  # 定义 K-Fold 的折数
    
    oof_predictions = np.zeros(len(X))
    oof_mse = []
    oof_r2 = []
    feature_importance_list = []
    models_dict = {cluster: [] for cluster in range(len(cluster_info))}
    
    # 定义各模型的超参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.1,          # 保持适中的学习率
        'num_leaves': 31,              # 合理的叶子数
        'max_depth': -1,               # 不限制深度
        'min_child_samples': 20,       # 防止过拟合
        'feature_fraction': 0.8,       # 每棵树使用80%的特征
        'bagging_fraction': 0.8,       # 每棵树使用80%的数据
        'bagging_freq': 5,             # 每5次迭代进行一次bagging
        'lambda_l1': 0.0,              # L1正则化项
        'lambda_l2': 10.0,             # L2正则化项
        'max_bin': 255,                # 默认值
        'min_data_in_leaf': 20,        # 与 min_child_samples 相同
        'feature_fraction_seed': 42,   # 确保可重复性
        'bagging_seed': 42,            # 确保可重复性
    }
    
    # GradientBoostingRegressor 的超参数
    gb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'loss': 'huber',
            'random_state': 42
        }
    
    # XGBoost 的超参数
    xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,    # L1正则化
            'reg_lambda': 1,    # L2正则化
            'seed': 42,
            'verbosity': 0      # 关闭XGBoost日志
        }
    
    start_time = time.time()
    
    # 使用 joblib 并行处理各个聚类
    # 这里只进行外层并行化（每个聚类一个并行任务）
    # 避免在每个聚类内再次使用 Parallel，防止资源竞争
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_cluster)(
            cluster=cluster,
            X=X,
            y=y,
            price_clusters=price_clusters,
            cluster_info=cluster_info,
            n_splits=n_splits,  # 传递折数，而不是 KFold 对象
            lgb_params=lgb_params,
            gb_params=gb_params,
            xgb_params=xgb_params,
            features_for_clustering=features_for_clustering
        )
        for cluster in range(len(cluster_info))
    )
    
    # 处理并收集并行结果
    for result in results:
        cluster = result['cluster']
        models_dict[cluster].extend(result['models'])  # 使用 extend 而不是 append
        for val_idx, preds in zip(result['val_indices'], result['predictions']):
            oof_predictions[val_idx] += preds  # 累加预测
        oof_mse.extend(result['oof_mse'])
        oof_r2.extend(result['oof_r2'])
        for fi in result['feature_importance']:
            feature_importance_list.append(fi)
    
    # 计算 OOF 预测的平均值
    fold_counts = np.zeros(len(X))
    for result in results:
        for val_idx in result['val_indices']:
            fold_counts[val_idx] += 1
    # 避免除以零
    fold_counts[fold_counts == 0] = 1
    oof_predictions /= fold_counts
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    # 检查是否有有效的 OOF 预测
    if np.all(oof_predictions == 0):
        logging.error("所有模型训练失败，未生成任何 OOF 预测。请检查日志中的错误信息。")
    else:
        oof_predictions = post_process_predictions(oof_predictions)
        oof_mse_total = mean_squared_error(y, oof_predictions)
        oof_r2_total = r2_score(y, oof_predictions)
        logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse_total):.4f}")
        logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")
    
    if feature_importance_list:
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
        logging.info("\nTop 10 important features:")
        logging.info(feature_importance.head(10))
    else:
        logging.warning("没有特征重要性信息可供显示。")
    
    # 保存模型和聚类信息
    with open(base_models_save_path, 'wb') as f:
        pickle.dump({
            'models': models_dict,  # 按集群存储
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info(f"Models and preprocessors saved to: {base_models_save_path}")
    
    # 预测测试数据
    logging.info(f"Loading test data from: {test_data_path}")
    X_test, _ = load_and_preprocess_data(test_data_path)
    
    # 分配测试数据的聚类
    test_clusters = assign_test_clusters(kmeans_model, X_test, features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    prediction_counts = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        logging.info(f"Predicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster  # 使用测试数据的聚类标签
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_models = models_dict[cluster]
        cluster_predictions = np.zeros(len(X_test_cluster))
        num_models = 0
        
        for model_dict in cluster_models:
            model_lgb = model_dict['lightgbm']
            model_gb = model_dict['gradient_boosting']
            model_xgb = model_dict['xgboost']
            preprocessors = model_dict['preprocessors']  # 这里为 None，因为不进行预处理
            
            # 由于不进行预处理，直接使用原始测试数据
            # 确保测试数据与训练数据格式一致
            
            # 预测 LightGBM
            try:
                preds_lgb = np.expm1(model_lgb.predict(X_test_cluster, num_iteration=model_lgb.best_iteration_))
            except Exception as e:
                logging.error(f"Error predicting with LightGBM for Cluster {cluster}: {str(e)}")
                preds_lgb = np.zeros(len(X_test_cluster))
            
            # 预测 GradientBoostingRegressor
            try:
                preds_gb = np.expm1(model_gb.predict(X_test_cluster))
            except Exception as e:
                logging.error(f"Error predicting with GradientBoostingRegressor for Cluster {cluster}: {str(e)}")
                preds_gb = np.zeros(len(X_test_cluster))
            
            # 预测 XGBoost
            try:
                preds_xgb = np.expm1(model_xgb.predict(X_test_cluster))
            except Exception as e:
                logging.error(f"Error predicting with XGBoost for Cluster {cluster}: {str(e)}")
                preds_xgb = np.zeros(len(X_test_cluster))
            
            # 集成预测（平均）
            preds = (preds_lgb + preds_gb + preds_xgb) / 3.0
            cluster_predictions += preds
            num_models += 1
        
        if num_models > 0:
            final_predictions[cluster_mask] += cluster_predictions
            prediction_counts[cluster_mask] += num_models
        else:
            logging.warning(f"No successful model predictions for cluster {cluster}.")
    
    # 避免除以零
    prediction_counts[prediction_counts == 0] = 1
    final_predictions /= prediction_counts
    
    final_predictions = post_process_predictions(final_predictions)
    
    # 输出预测统计信息
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min():.2f}")
    logging.info(f"Maximum: {final_predictions.max():.2f}")
    logging.info(f"Mean: {final_predictions.mean():.2f}")
    logging.info(f"Median: {np.median(final_predictions):.2f}")
    
    # 处理训练集数据
    logging.info("\nProcessing training data...")
    train_data = pd.read_csv(train_data_path)
    train_data['ref_price'] = oof_predictions  # 使用out-of-fold预测作为ref_price
    train_data.to_csv(with_ref_price_train_data_path, index=False)
    logging.info(f"Training data with ref_price saved to {with_ref_price_train_data_path}")
    
    # 处理测试集数据
    logging.info("\nProcessing test data...")
    test_data = pd.read_csv(test_data_path)
    test_data['ref_price'] = final_predictions  # 使用模型预测作为ref_price
    test_data.to_csv(with_ref_price_test_data_path, index=False)
    logging.info(f"Test data with ref_price saved to {with_ref_price_test_data_path}")
    
    # 输出ref_price的统计信息
    logging.info("\nRef_price statistics for training data:")
    logging.info(train_data['ref_price'].describe())
    logging.info("\nRef_price statistics for test data:")
    logging.info(test_data['ref_price'].describe())
    
    
    
    train_data_path = with_ref_price_train_data_path
    test_data_path = with_ref_price_test_data_path
    
    
    # 重复一遍，修改ref_price
    
    logging.info(f"Loading training data from: {train_data_path}")
    X, y = load_and_preprocess_data(train_data_path)
    
    if y is None:
        logging.error("目标变量 'price' 不存在于训练数据中。请检查数据。")
        return
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=features_for_clustering)
    
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    n_splits = 5  # 定义 K-Fold 的折数
    
    oof_predictions = np.zeros(len(X))
    oof_mse = []
    oof_r2 = []
    feature_importance_list = []
    models_dict = {cluster: [] for cluster in range(len(cluster_info))}
    
    # 定义各模型的超参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.1,          # 保持适中的学习率
        'num_leaves': 31,              # 合理的叶子数
        'max_depth': -1,               # 不限制深度
        'min_child_samples': 20,       # 防止过拟合
        'feature_fraction': 0.8,       # 每棵树使用80%的特征
        'bagging_fraction': 0.8,       # 每棵树使用80%的数据
        'bagging_freq': 5,             # 每5次迭代进行一次bagging
        'lambda_l1': 0.0,              # L1正则化项
        'lambda_l2': 10.0,             # L2正则化项
        'max_bin': 255,                # 默认值
        'min_data_in_leaf': 20,        # 与 min_child_samples 相同
        'feature_fraction_seed': 42,   # 确保可重复性
        'bagging_seed': 42,            # 确保可重复性
    }
    
    # GradientBoostingRegressor 的超参数
    gb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'loss': 'huber',
            'random_state': 42
        }
    
    # XGBoost 的超参数
    xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,    # L1正则化
            'reg_lambda': 1,    # L2正则化
            'seed': 42,
            'verbosity': 0      # 关闭XGBoost日志
        }
    
    start_time = time.time()
    
    # 使用 joblib 并行处理各个聚类
    # 这里只进行外层并行化（每个聚类一个并行任务）
    # 避免在每个聚类内再次使用 Parallel，防止资源竞争
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_cluster)(
            cluster=cluster,
            X=X,
            y=y,
            price_clusters=price_clusters,
            cluster_info=cluster_info,
            n_splits=n_splits,  # 传递折数，而不是 KFold 对象
            lgb_params=lgb_params,
            gb_params=gb_params,
            xgb_params=xgb_params,
            features_for_clustering=features_for_clustering
        )
        for cluster in range(len(cluster_info))
    )
    
    # 处理并收集并行结果
    for result in results:
        cluster = result['cluster']
        models_dict[cluster].extend(result['models'])  # 使用 extend 而不是 append
        for val_idx, preds in zip(result['val_indices'], result['predictions']):
            oof_predictions[val_idx] += preds  # 累加预测
        oof_mse.extend(result['oof_mse'])
        oof_r2.extend(result['oof_r2'])
        for fi in result['feature_importance']:
            feature_importance_list.append(fi)
    
    # 计算 OOF 预测的平均值
    fold_counts = np.zeros(len(X))
    for result in results:
        for val_idx in result['val_indices']:
            fold_counts[val_idx] += 1
    # 避免除以零
    fold_counts[fold_counts == 0] = 1
    oof_predictions /= fold_counts
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    # 检查是否有有效的 OOF 预测
    if np.all(oof_predictions == 0):
        logging.error("所有模型训练失败，未生成任何 OOF 预测。请检查日志中的错误信息。")
    else:
        oof_predictions = post_process_predictions(oof_predictions)
        oof_mse_total = mean_squared_error(y, oof_predictions)
        oof_r2_total = r2_score(y, oof_predictions)
        logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse_total):.4f}")
        logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")
    
    if feature_importance_list:
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
        logging.info("\nTop 10 important features:")
        logging.info(feature_importance.head(10))
    else:
        logging.warning("没有特征重要性信息可供显示。")
    
    # 保存模型和聚类信息
    with open(with_ref_models_save_path, 'wb') as f:
        pickle.dump({
            'models': models_dict,  # 按集群存储
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info(f"Models and preprocessors saved to: {with_ref_models_save_path}")
    
    # 预测测试数据
    logging.info(f"Loading test data from: {test_data_path}")
    X_test, _ = load_and_preprocess_data(test_data_path)
    
    # 分配测试数据的聚类
    test_clusters = assign_test_clusters(kmeans_model, X_test, features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    prediction_counts = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        logging.info(f"Predicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster  # 使用测试数据的聚类标签
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_models = models_dict[cluster]
        cluster_predictions = np.zeros(len(X_test_cluster))
        num_models = 0
        
        for model_dict in cluster_models:
            model_lgb = model_dict['lightgbm']
            model_gb = model_dict['gradient_boosting']
            model_xgb = model_dict['xgboost']
            preprocessors = model_dict['preprocessors']  # 这里为 None，因为不进行预处理
            
            # 由于不进行预处理，直接使用原始测试数据
            # 确保测试数据与训练数据格式一致
            
            # 预测 LightGBM
            try:
                preds_lgb = np.expm1(model_lgb.predict(X_test_cluster, num_iteration=model_lgb.best_iteration_))
            except Exception as e:
                logging.error(f"Error predicting with LightGBM for Cluster {cluster}: {str(e)}")
                preds_lgb = np.zeros(len(X_test_cluster))
            
            # 预测 GradientBoostingRegressor
            try:
                preds_gb = np.expm1(model_gb.predict(X_test_cluster))
            except Exception as e:
                logging.error(f"Error predicting with GradientBoostingRegressor for Cluster {cluster}: {str(e)}")
                preds_gb = np.zeros(len(X_test_cluster))
            
            # 预测 XGBoost
            try:
                preds_xgb = np.expm1(model_xgb.predict(X_test_cluster))
            except Exception as e:
                logging.error(f"Error predicting with XGBoost for Cluster {cluster}: {str(e)}")
                preds_xgb = np.zeros(len(X_test_cluster))
            
            # 集成预测（平均）
            preds = (preds_lgb + preds_gb + preds_xgb) / 3.0
            cluster_predictions += preds
            num_models += 1
        
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
    
    submission.to_csv(submission_path, index=False)
    logging.info(f"Predictions complete. Submission file saved as '{submission_path}'.")
    


if __name__ == '__main__':
    main()
