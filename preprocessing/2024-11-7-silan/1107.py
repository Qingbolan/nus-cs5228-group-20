import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
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

class WeightedEnsembleL2:
    def __init__(self, lambda_reg=0.0001):  # 减小正则化参数
        self.weights = None
        self.lambda_reg = lambda_reg

    def fit(self, predictions, y_true):
        """
        使用L2正则化的最小二乘法来优化权重
        predictions: shape (n_samples, n_models)
        y_true: shape (n_samples,)
        """
        n_models = predictions.shape[1]

        # 添加正则化项的解析解
        A = predictions.T @ predictions + self.lambda_reg * np.eye(n_models)
        b = predictions.T @ y_true

        try:
            self.weights = np.linalg.solve(A, b)
            # 归一化权重
            self.weights = np.maximum(0, self.weights)  # 确保权重非负
            if np.sum(self.weights) > 0:
                self.weights /= np.sum(self.weights)
            else:
                self.weights = np.ones(n_models) / n_models
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用均等权重
            self.weights = np.ones(n_models) / n_models

        return self

    def predict(self, predictions):
        """
        使用学习到的权重进行预测
        predictions: shape (n_samples, n_models)
        """
        return predictions @ self.weights

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
            verbosity=0
        )),
        ('cat', CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            min_data_in_leaf=20,
            random_strength=0.5,
            bagging_temperature=0.2,
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            verbose=False
        )),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
        ('etr', ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
        ('gbr', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
        ('ada', AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.05,
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
            cat_imputer = SimpleImputer(strategy='most_frequent')  # 修改为most_frequent
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
    ensemble_weights_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
        logging.info(f"Cluster {cluster} - Fold {fold}")

        X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
        y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]

        # 数据预处理
        X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
        X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)

        # 获取基础模型
        base_models = get_base_models()

        # 训练基础模型并收集预测结果
        val_preds = np.zeros((X_val_processed.shape[0], len(base_models)))
        train_preds = np.zeros((X_train_processed.shape[0], len(base_models)))

        model_performance = []

        for i, (name, model) in enumerate(base_models):
            logging.debug(f"Training base model: {name}")

            try:
                if name == 'lgb':
                    model.fit(X_train_processed, np.log1p(y_train),
                              eval_set=[(X_val_processed, np.log1p(y_val))],)
                    preds_val = np.expm1(model.predict(X_val_processed, num_iteration=model.best_iteration_))
                    preds_train = np.expm1(model.predict(X_train_processed, num_iteration=model.best_iteration_))
                elif name == 'xgb':
                    model.fit(X_train_processed, np.log1p(y_train),
                              eval_set=[(X_val_processed, np.log1p(y_val))],
                              verbose=False)
                    preds_val = np.expm1(model.predict(X_val_processed))
                    preds_train = np.expm1(model.predict(X_train_processed))
                elif name == 'cat':
                    model.fit(X_train_processed, np.log1p(y_train),
                              eval_set=(X_val_processed, np.log1p(y_val)),
                              use_best_model=True,)
                    preds_val = np.expm1(model.predict(X_val_processed))
                    preds_train = np.expm1(model.predict(X_train_processed))
                else:
                    model.fit(X_train_processed, np.log1p(y_train))
                    preds_val = np.expm1(model.predict(X_val_processed))
                    preds_train = np.expm1(model.predict(X_train_processed))

                val_preds[:, i] = preds_val
                train_preds[:, i] = preds_train

                rmse = np.sqrt(mean_squared_error(y_val, preds_val))
                r2 = r2_score(y_val, preds_val)
                model_performance.append({'name': name, 'rmse': rmse, 'r2': r2})
                logging.debug(f"Model {name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

            except Exception as e:
                logging.error(f"Error training model {name}: {str(e)}")
                val_preds[:, i] = 0
                train_preds[:, i] = 0
                model_performance.append({'name': name, 'rmse': np.inf, 'r2': -np.inf})

        # 使用WeightedEnsembleL2进行加权集成
        ensemble = WeightedEnsembleL2(lambda_reg=0.0001)
        ensemble.fit(train_preds, y_train)
        preds_meta = ensemble.predict(val_preds)
        oof_predictions_cluster[val_index] = preds_meta

        rmse = np.sqrt(mean_squared_error(y_val, preds_meta))
        r2 = r2_score(y_val, preds_meta)
        oof_mse.append(rmse ** 2)
        oof_r2.append(r2)
        logging.info(f"Cluster {cluster} - Fold {fold} Ensemble RMSE: {rmse:.4f}, R2: {r2:.4f}")
        logging.info(f"Ensemble weights: {dict(zip([name for name, _ in base_models], ensemble.weights))}")

        # 保存模型和预处理器
        models.append({
            'base_models': base_models,
            'ensemble': ensemble,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        })

        # 保存集成权重
        ensemble_weights_list.append(ensemble.weights)

    # 计算总体指标
    oof_rmse = np.sqrt(mean_squared_error(y_cluster, oof_predictions_cluster))
    oof_r2_total = r2_score(y_cluster, oof_predictions_cluster)
    logging.info(f"Cluster {cluster} OOF RMSE: {oof_rmse:.4f}, R2: {oof_r2_total:.4f}")

    # 计算平均的集成权重
    avg_ensemble_weights = np.mean(ensemble_weights_list, axis=0)

    return models, oof_predictions_cluster, avg_ensemble_weights

def retrain_models_for_cluster(cluster, X_cluster, y_cluster):
    logging.info(f"Retraining models for Cluster {cluster} on full data")
    # 数据预处理
    X_cluster_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_cluster, y_cluster)

    # 获取基础模型
    base_models = get_base_models()

    # 训练基础模型
    for name, model in base_models:
        logging.debug(f"Retraining base model: {name}")
        if name == 'lgb':
            model.fit(X_cluster_processed, np.log1p(y_cluster))
        elif name == 'xgb':
            model.fit(X_cluster_processed, np.log1p(y_cluster),
                      verbose=False)
        elif name == 'cat':
            model.fit(X_cluster_processed, np.log1p(y_cluster),
                      use_best_model=False)
        else:
            model.fit(X_cluster_processed, np.log1p(y_cluster))

    # 保存模型和预处理器
    models = {
        'base_models': base_models,
        'preprocessors': {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'target_encoder': target_encoder,
            'scaler': scaler
        }
    }

    return models

def predict_for_cluster(cluster, X_test_cluster, retrained_models, ensemble_weights):
    # 数据预处理
    X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **retrained_models['preprocessors'])

    base_models = retrained_models['base_models']
    val_preds = np.zeros((X_test_processed.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        logging.debug(f"Predicting with base model: {name}")
        preds = np.expm1(model.predict(X_test_processed))
        val_preds[:, i] = preds

    # 使用在交叉验证中得到的平均集成权重
    final_predictions_cluster = val_preds @ ensemble_weights

    return final_predictions_cluster

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
    ensemble_weights_list = []

    start_time = time.time()

    for cluster in range(len(cluster_info)):
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]

        # 使用KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cluster_models, oof_preds_cluster, avg_ensemble_weights = train_models_for_cluster(cluster, X_cluster, y_cluster, kf)
        models.append({
            'fold_models': cluster_models,
            'avg_ensemble_weights': avg_ensemble_weights
        })
        oof_predictions[price_clusters == cluster] = oof_preds_cluster
        ensemble_weights_list.append(avg_ensemble_weights)

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
    with open('weighted_ensemble_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': retrained_models_list,
            'ensemble_weights_list': ensemble_weights_list,
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
        ensemble_weights = ensemble_weights_list[cluster]

        preds_cluster = predict_for_cluster(cluster, X_test_cluster, retrained_models, ensemble_weights)
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