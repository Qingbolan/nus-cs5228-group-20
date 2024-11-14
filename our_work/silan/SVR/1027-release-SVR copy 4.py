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
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):

    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns.tolist()}")
    
    return X, y

def encode_categorical_features(X):
    """对类别特征进行编码"""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X

def preprocess_features_fit(X, y=None, num_imputer=None, cat_imputer=None, 
                            target_encoder=None, scaler=None, 
                            target_encode_cols=['make', 'model'], 
                            encoding_smoothing=1.0):
    """
    在训练数据上拟合并应用特征预处理器。
    
    返回：
        X: 预处理后的特征数据框
        num_imputer: 数值特征的缺失值填补器
        cat_imputer: 类别特征的缺失值填补器
        target_encoder: 目标编码器
        scaler: 标准化器
        encoder: OneHotEncoder
    """

    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 数值特征缺失值填补
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    else:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
    
    # 类别特征缺失值填补
    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
        else:
            X[categorical_features] = cat_imputer.transform(X[categorical_features])
        
        # 目标编码
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        if target_encode_features:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = target_encoder.fit_transform(X[target_encode_features], y)
            else:
                X[target_encode_features] = target_encoder.transform(X[target_encode_features])
        
        # 其他类别特征一热编码

        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)
    else:
        target_encode_features = []
        encoded_feature_names = []
        encoder = None
    
    # 标准化数值特征
    if scaler is None:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
    else:
        X[numeric_features] = scaler.transform(X[numeric_features])
    
    return X, num_imputer, cat_imputer, target_encoder, scaler, encoder

def preprocess_features_transform(X, y=None, num_imputer=None, cat_imputer=None, 
                                 target_encoder=None, scaler=None, 
                                 encoder=None, target_encode_cols=['make', 'model']):
    """
    使用已拟合的预处理器转换特征，并应用已拟合的OneHotEncoder。
    """
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 数值特征缺失值填补
    if num_imputer is not None:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
    
    # 类别特征缺失值填补
    if len(categorical_features) > 0 and cat_imputer is not None:
        X[categorical_features] = cat_imputer.transform(X[categorical_features])
        
        # 目标编码
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        if target_encode_features and target_encoder is not None:
            X[target_encode_features] = target_encoder.transform(X[target_encode_features])
        
        # 其他类别特征一热编码
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0 and encoder is not None:
            encoded_features = encoder.transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)
    
    # 标准化数值特征
    if scaler is not None:
        X[numeric_features] = scaler.transform(X[numeric_features])
    
    return X

def find_optimal_clusters(X, y, max_clusters=10, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    # First ensure the clustering features have no NaN values
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    
    # Apply simple imputation for any NaN values
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    # Stack with log-transformed price
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
    # First handle NaN values in clustering features

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
    
    # 收集聚类统计信息
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices),
            'std': cluster_prices.std(),
            'log_std': np.log1p(cluster_prices).std()
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    # Store the imputer with the kmeans model for later use

    kmeans.feature_imputer = imputer
    
    return kmeans, price_clusters, cluster_df

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        # early_stopping_rounds=50,
        # verbose_eval=100
    )
    
    return model

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    X_processed, _, _, _, _, encoder = preprocess_features_fit(X, y, 
                                                               num_imputer=preprocessors['num_imputer'],
                                                               cat_imputer=preprocessors['cat_imputer'],
                                                               target_encoder=preprocessors['target_encoder'],
                                                               scaler=preprocessors['scaler'],
                                                               target_encode_cols=['make', 'model'],
                                                               encoding_smoothing=1.0)
    
    # Handle NaN in clustering features

    cluster_features_df = pd.DataFrame(X_processed[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    
    cluster_features = np.column_stack([np.log1p(y) if y is not None else np.zeros(len(X)), cluster_features_clean])
    return kmeans_model.predict(cluster_features)


def analyze_clusters(X: pd.DataFrame, y: pd.Series, clusters: np.ndarray):
    """对每个簇进行数据分布分析"""
    X_with_target = X.copy()
    X_with_target['price'] = y
    X_with_target['cluster'] = clusters

    for cluster in sorted(X_with_target['cluster'].unique()):
        cluster_data = X_with_target[X_with_target['cluster'] == cluster]
        plt.figure(figsize=(10, 6))
        sns.histplot(cluster_data['price'], bins=50, kde=True)
        plt.title(f'Price Distribution for Cluster {cluster}')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

        # 特征与目标的相关性
        plt.figure(figsize=(12, 8))
        corr = cluster_data.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title(f'Correlation Heatmap for Cluster {cluster}')
        plt.show()

def analyze_predictions(y_true: pd.Series, y_pred: np.ndarray):
    """生成预测结果的可视化分析"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('True vs Predicted Prices')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.show()

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residual')
    plt.title('Residual Distribution')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """绘制特征重要性图"""
    importances = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values(by='importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def main():
    np.random.seed(42)
    
    # 加载训练数据
    X, y = load_and_preprocess_data('preprocessing/release/ver2/train_cleaned.csv')

    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=5, features_for_clustering=features_for_clustering)
    
    # 创建价格聚类
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    # 深入分析聚类（可选择性打开）
    # analyze_clusters(X, y, price_clusters)
    
    # K-Fold 交叉验证

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_list = []
    models = []
    
    # LightGBM 参数调整或使用 Optuna 寻找最佳参数
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
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    

    start_time = time.time()
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\nTraining models for Cluster {cluster}")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train_fold, y_val_fold = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            # 预处理特征
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler, encoder = preprocess_features_fit(X_train, y_train_fold)
            # 预处理验证集特征
            try:
                X_val_processed = preprocess_features_transform(X_val, y=None, 
                                                                 num_imputer=num_imputer, 
                                                                 cat_imputer=cat_imputer, 
                                                                 target_encoder=target_encoder, 
                                                                 scaler=scaler, 
                                                                 encoder=encoder, 
                                                                 target_encode_cols=['make', 'model'])
            except NotImplementedError as e:
                logging.error(str(e))
                # 如果 OneHotEncoder 部分需要额外处理，这里可以添加代码以加载和应用已拟合的编码器
                raise
            
            # 训练模型
            model = train_evaluate_lightgbm(X_train_processed, y_train_fold, X_val_processed, y_val_fold, params)
            
            # 预测并逆转换
            y_val_pred = np.expm1(model.predict(X_val_processed, num_iteration=model.best_iteration))
            oof_predictions[price_clusters == cluster][val_index] = y_val_pred
            
            # 记录特征重要性
            importance = model.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            cluster_models.append({
                'model': model,

                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler,
                    'encoder': encoder
                }
            })
        
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
    
    with open('lightgbm_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 加载测试数据
    X_test, _ = load_and_preprocess_data('preprocessing/release/ver2/test_cleaned.csv')
    
    # 创建测试集的聚类标签（忽略 y_test，因为测试集中没有标签）
    try:
        # 预处理测试集特征
        X_test_processed, num_imputer_test, cat_imputer_test, target_encoder_test, scaler_test, encoder_test = preprocess_features_fit(X_test, y=None)
        
        # 处理聚类特征
        cluster_features_df_test = pd.DataFrame(X_test_processed[features_for_clustering])
        cluster_features_clean_test = kmeans_model.feature_imputer.transform(cluster_features_df_test)
        
        # 假设测试集没有价格信息，使用相同的聚类特征进行预测
        cluster_features_test = np.column_stack([np.zeros(len(X_test)), cluster_features_clean_test])  # 使用0作为log_price的代理
        test_clusters = kmeans_model.predict(cluster_features_test)
    except Exception as e:
        logging.error(f"Error during clustering the test set: {str(e)}")
        test_clusters = np.zeros(len(X_test))  # 默认单一簇
    
    final_predictions = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_predictions = []
        
        for model_dict in models[cluster]:
            model = model_dict['model']
            preprocessors = model_dict['preprocessors']
            
            try:
                # 重新构造预测集的预处理器（这里假设在训练时拟合的编码器已保存并能够复用）
                X_test_processed_cluster = X_test_cluster.copy()
                
                # 使用训练时的预处理器
                X_test_processed_cluster = preprocess_features_transform(X_test_processed_cluster, y=None, 
                                                                           num_imputer=preprocessors['num_imputer'], 
                                                                           cat_imputer=preprocessors['cat_imputer'], 
                                                                           target_encoder=preprocessors['target_encoder'], 
                                                                           scaler=preprocessors['scaler'], 
                                                                           encoder=preprocessors['encoder'], 
                                                                           target_encode_cols=['make', 'model'])
                
                # 预测并逆转换
                preds = np.expm1(model.predict(X_test_processed_cluster, num_iteration=model.best_iteration))
                cluster_predictions.append(preds)

            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}: {str(e)}")
                continue
        
        if len(cluster_predictions) == 0:
            logging.warning(f"No models available for cluster {cluster}. Skipping.")
            continue
        
        # 集成不同fold的预测结果（简单平均）
        cluster_predictions = np.mean(cluster_predictions, axis=0)
        final_predictions[cluster_mask] = cluster_predictions
    
    final_predictions = post_process_predictions(final_predictions)
    
    # 生成Submission文件

    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./final_submission_lightgbm.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'final_submission_lightgbm.csv'.")
    

    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()