import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别
    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log', mode='w'),
        logging.StreamHandler()
    ]
)

def get_base_models():
    base_models = [
        ('lgb', lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            verbosity=-1,
            random_state=42,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.7,
            bagging_freq=5,
            max_depth=-1,
            min_child_samples=20,
            n_estimators=1000,
        )),
        ('xgb', xgb.XGBRegressor(
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.8,
            gamma=0,
            objective='reg:squarederror',
            eval_metric='rmse',
            booster='gbtree',
            random_state=42,
            verbosity=0,
            n_estimators=1000,
        )),
        ('cat', CatBoostRegressor(
            iterations=2000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=5,
            random_seed=42,
            od_type='Iter',
            od_wait=50,
            verbose=False
        )),
        ('rf', RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('etr', ExtraTreesRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('gbr', GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
    ]
    return base_models

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

    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]),
                                           columns=numeric_features,
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]),
                                           columns=numeric_features,
                                           index=X.index)

    # 自动检测需要标准化的列
    columns_to_standardize = numeric_features
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
            cat_imputer = SimpleImputer(strategy='most_frequent')
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

    return X, num_imputer, cat_imputer, target_encoder, scaler

def create_price_clusters(X, y, n_clusters, features_for_clustering):
    # 提取用于聚类的特征并进行预处理
    cluster_features = X[features_for_clustering].copy()
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    cluster_features = pd.DataFrame(num_imputer.fit_transform(cluster_features),
                                    columns=features_for_clustering,
                                    index=X.index)
    cluster_features = pd.DataFrame(scaler.fit_transform(cluster_features),
                                    columns=features_for_clustering,
                                    index=X.index)

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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

    # 保存聚类所需的预处理器
    cluster_preprocessors = {
        'num_imputer': num_imputer,
        'scaler': scaler
    }

    return kmeans, price_clusters, cluster_df, cluster_preprocessors

def find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    from sklearn.metrics import silhouette_score
    # 提取用于聚类的特征并进行预处理
    cluster_features = X[features_for_clustering].copy()
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    cluster_features = pd.DataFrame(num_imputer.fit_transform(cluster_features),
                                    columns=features_for_clustering,
                                    index=X.index)
    cluster_features = pd.DataFrame(scaler.fit_transform(cluster_features),
                                    columns=features_for_clustering,
                                    index=X.index)

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

def predict_cluster(X, kmeans_model, cluster_preprocessors, features_for_clustering):
    # 提取用于聚类的特征并进行预处理
    X_cluster_features = X[features_for_clustering].copy()
    num_imputer = cluster_preprocessors['num_imputer']
    scaler = cluster_preprocessors['scaler']

    X_cluster_features = pd.DataFrame(num_imputer.transform(X_cluster_features),
                                      columns=features_for_clustering,
                                      index=X.index)
    X_cluster_features = pd.DataFrame(scaler.transform(X_cluster_features),
                                      columns=features_for_clustering,
                                      index=X.index)

    # 使用训练阶段的聚类模型进行聚类分配
    cluster_features = X_cluster_features
    cluster_labels = kmeans_model.predict(cluster_features)
    return cluster_labels

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def train_models_for_cluster(cluster, X_cluster, y_cluster, kf):
    logging.info(f"\nTraining models for Cluster {cluster}")
    oof_predictions_cluster = np.zeros(len(X_cluster))
    models = []
    oof_mse = []
    oof_r2 = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
        logging.info(f"Cluster {cluster} - Fold {fold}")

        X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
        y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]

        # 数据预处理
        X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
        X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)

        # 获取基础模型
        base_models = get_base_models()

        # 创建元学习器
        meta_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            random_state=42,
            learning_rate=0.05,
            n_estimators=1000,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.7,
            bagging_freq=5,
        )

        # 创建堆叠模型
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            passthrough=False,
            cv=kf,
            n_jobs=-1
        )

        # 训练堆叠模型
        stacking_model.fit(X_train_processed, np.log1p(y_train))

        # 预测验证集
        preds_val = np.expm1(stacking_model.predict(X_val_processed))
        oof_predictions_cluster[val_index] = preds_val

        rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        r2 = r2_score(y_val, preds_val)
        oof_mse.append(rmse ** 2)
        oof_r2.append(r2)
        logging.info(f"Cluster {cluster} - Fold {fold} Stacking RMSE: {rmse:.4f}, R2: {r2:.4f}")

        # 保存模型和预处理器
        models.append({
            'stacking_model': stacking_model,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        })

    # 计算总体指标
    oof_rmse = np.sqrt(mean_squared_error(y_cluster, oof_predictions_cluster))
    oof_r2_total = r2_score(y_cluster, oof_predictions_cluster)
    logging.info(f"Cluster {cluster} OOF RMSE: {oof_rmse:.4f}, R2: {oof_r2_total:.4f}")

    return models, oof_predictions_cluster

def retrain_models_for_cluster(cluster, X_cluster, y_cluster):
    logging.info(f"Retraining models for Cluster {cluster} on full data")
    # 数据预处理
    X_cluster_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_cluster, y_cluster)

    # 获取基础模型
    base_models = get_base_models()

    # 创建元学习器
    meta_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        random_state=42,
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.7,
        bagging_freq=5,
    )

    # 创建堆叠模型
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=False,
        cv=5,
        n_jobs=-1
    )

    # 训练堆叠模型
    stacking_model.fit(X_cluster_processed, np.log1p(y_cluster))

    # 保存模型和预处理器
    models = {
        'stacking_model': stacking_model,
        'preprocessors': {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'target_encoder': target_encoder,
            'scaler': scaler
        }
    }

    return models

def predict_for_cluster(cluster, X_test_cluster, retrained_models):
    # 数据预处理
    X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **retrained_models['preprocessors'])

    # 预测
    stacking_model = retrained_models['stacking_model']
    preds = np.expm1(stacking_model.predict(X_test_processed))

    return preds

def main():
    np.random.seed(42)

    X, y = load_and_preprocess_data(os.getenv('train_data'))

    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())

    features_for_clustering = ['depreciation', 'vehicle_age', 'dereg_value', 'coe', 'arf', 'power']

    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=features_for_clustering)

    kmeans_model, price_clusters, cluster_info, cluster_preprocessors = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)

    oof_predictions = np.zeros(len(X))
    models = []

    start_time = time.time()

    for cluster in range(len(cluster_info)):
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]

        # 使用KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cluster_models, oof_preds_cluster = train_models_for_cluster(cluster, X_cluster, y_cluster, kf)
        models.append(cluster_models)
        oof_predictions[price_clusters == cluster] = oof_preds_cluster

    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")

    oof_predictions = post_process_predictions(oof_predictions)
    oof_rmse_total = np.sqrt(mean_squared_error(y, oof_predictions))
    oof_r2_total = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {oof_rmse_total:.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")

    # 重新训练模型并保存
    retrained_models_list = []

    for cluster in range(len(cluster_info)):
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]

        retrained_models = retrain_models_for_cluster(cluster, X_cluster, y_cluster)
        retrained_models_list.append(retrained_models)

    # 保存模型和预处理器、聚类模型
    with open('stacking_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': retrained_models_list,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info,
            'features_for_clustering': features_for_clustering,
            'cluster_preprocessors': cluster_preprocessors  # 保存用于聚类的预处理器
        }, f)
    logging.info("Models and preprocessors saved.")

    # 预测测试数据
    X_test, _ = load_and_preprocess_data(os.getenv('test_data'))

    # 使用相同的聚类模型和预处理器对测试数据进行聚类分配
    test_clusters = predict_cluster(X_test, kmeans_model, cluster_preprocessors, features_for_clustering)

    final_predictions = np.zeros(len(X_test))

    for cluster in range(len(cluster_info)):
        logging.info(f"Predicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]

        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue

        retrained_models = retrained_models_list[cluster]

        preds_cluster = predict_for_cluster(cluster, X_test_cluster, retrained_models)
        final_predictions[cluster_mask] = preds_cluster

    final_predictions = post_process_predictions(final_predictions)

    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })

    submission.to_csv(os.getenv('output_prediction'), index=False)
    logging.info(f"Predictions complete. Submission file saved as {os.getenv('output_prediction')}.")

    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

    # 处理训练集数据
    logging.info("\nProcessing training data...")
    train_data = pd.read_csv(os.getenv('train_data'))
    train_data['ref_price'] = oof_predictions  # 使用out-of-fold预测作为ref_price
    train_data.to_csv(os.getenv('ref_train'), index=False)
    logging.info(f"Training data with ref_price saved to {os.getenv('ref_train')}")

    # 处理测试集数据
    logging.info("\nProcessing test data...")
    test_data = pd.read_csv(os.getenv('test_data'))
    test_data['ref_price'] = final_predictions  # 使用模型预测作为ref_price
    test_data.to_csv(os.getenv('ref_test'), index=False)
    logging.info(f"Test data with ref_price saved to {os.getenv('ref_test')}")

    # 输出ref_price的统计信息
    logging.info("\nRef_price statistics for training data:")
    logging.info(train_data['ref_price'].describe())
    logging.info("\nRef_price statistics for test data:")
    logging.info(test_data['ref_price'].describe())

if __name__ == '__main__':
    main()
