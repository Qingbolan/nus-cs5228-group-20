# price_clustering.py

import os
import pickle
import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from dotenv import load_dotenv

# 导入已有的模型训练函数
from final_report.approach1.without_cluster.models.xgboost import train_xgboost_models
from final_report.approach1.without_cluster.models.lightgbm import train_lightgbm_models
from final_report.approach1.without_cluster.models.catboost import train_catboost_models
from final_report.approach1.without_cluster.models.gradientboost import train_gradient_boosting_models

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

load_dotenv()

def load_and_preprocess_data(file_path: str, is_train: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    加载并预处理数据
    Args:
        file_path: 数据文件路径
        is_train: 是否为训练集
    Returns:
        X: 特征数据
        y: 目标变量（仅训练集）
    """
    data = pd.read_csv(file_path)
    logging.info(f"成功加载数据文件: {file_path}. 数据形状: {data.shape}")

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
        logging.info("删除 'Unnamed: 0' 列")

    if is_train:
        X = data.drop('price', axis=1)
        y = data['price']
        logging.info(f"训练集特征形状: {X.shape}")
        logging.info(f"训练集目标变量形状: {y.shape}")
        logging.info(f"价格范围: {y.min()} 到 {y.max()}")
        logging.info("\n目标变量 (price) 统计信息:")
        logging.info(y.describe())
    else:
        X = data
        y = None
        logging.info(f"测试集特征形状: {X.shape}")

    return X, y

def preprocess_features(X: pd.DataFrame, y: Optional[pd.Series] = None,
                       num_imputer: Optional[SimpleImputer] = None,
                       cat_imputer: Optional[SimpleImputer] = None,
                       encoder: Optional[OneHotEncoder] = None,
                       scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, Any, Any, Any, Any]:
    """
    预处理特征：填充缺失值、编码、标准化
    Args:
        X: 特征数据
        y: 目标变量（仅用于训练集的目标编码）
        num_imputer: 数值特征填充器
        cat_imputer: 类别特征填充器
        encoder: OneHot编码器
        scaler: 标准化器
    Returns:
        X_processed: 预处理后的特征数据
        num_imputer, cat_imputer, encoder, scaler: 预处理器对象
    """
    X = X.copy()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logging.info(f"数值特征: {numeric_features}")
    logging.info(f"类别特征: {categorical_features}")

    # 数值特征填充
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
        logging.info("使用中位数填充数值特征缺失值")
    else:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
        logging.info("使用现有的数值特征填充器填充数值特征缺失值")

    # 类别特征填充
    if cat_imputer is None:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
        logging.info("使用常量 'unknown' 填充类别特征缺失值")
    else:
        X[categorical_features] = cat_imputer.transform(X[categorical_features])
        logging.info("使用现有的类别特征填充器填充类别特征缺失值")

    # 类别特征编码
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        if categorical_features:
            encoded = encoder.fit_transform(X[categorical_features])
            encoded_cols = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
            X = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)
            logging.info(f"对类别特征进行 OneHot 编码，生成特征数量: {encoded_df.shape[1]}")
    else:
        if categorical_features:
            encoded = encoder.transform(X[categorical_features])
            encoded_cols = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
            X = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)
            logging.info(f"使用现有的 OneHot 编码器对类别特征进行编码，生成特征数量: {encoded_df.shape[1]}")
    
    # 标准化数值特征
    if scaler is None:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        logging.info("对数值特征进行标准化")
    else:
        X[numeric_features] = scaler.transform(X[numeric_features])
        logging.info("使用现有的标准化器对数值特征进行标准化")

    return X, num_imputer, cat_imputer, encoder, scaler

def find_optimal_clusters(X: pd.DataFrame, max_clusters: int = 3,
                         features_for_clustering: list = ['depreciation', 'coe', 'dereg_value']) -> int:
    """
    使用轮廓系数找到最优聚类数量
    Args:
        X: 特征数据
        max_clusters: 最大聚类数量
        features_for_clustering: 用于聚类的特征
    Returns:
        optimal_clusters: 最优聚类数量
    """
    # 组合特征用于聚类，仅使用特征，不包含目标变量
    cluster_features = X[features_for_clustering].values
    logging.info("开始寻找最优聚类数量")

    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"n_clusters = {n_clusters}, silhouette score: {silhouette_avg:.4f}")

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    logging.info(f"最优聚类数量: {optimal_clusters}")

    return optimal_clusters

def create_price_clusters(X: pd.DataFrame, y: pd.Series, n_clusters: int,
                         features_for_clustering: list = ['depreciation', 'coe', 'dereg_value']) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    创建价格聚类
    Args:
        X: 特征数据
        y: 目标变量（价格）
        n_clusters: 聚类数量
        features_for_clustering: 用于聚类的特征
    Returns:
        kmeans: 训练好的 KMeans 模型
        price_clusters: 聚类标签
        cluster_info: 聚类信息
    """
    # 计算特征的百分位点作为初始聚类中心
    feature_percentiles = np.percentile(X[features_for_clustering], np.linspace(0, 100, n_clusters))
    
    # 计算聚类中心
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_features = X[features_for_clustering].values
    price_clusters = kmeans.fit_predict(cluster_features)

    # 统计每个聚类的信息
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_mask = price_clusters == cluster
        cluster_prices = y[cluster_mask]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices),
            'mean': cluster_prices.mean()
        })

    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)

    return kmeans, price_clusters, cluster_df

def train_models_for_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: np.ndarray,
    model_name: str,
    output_dir: str,
    params: Dict[str, Any],
    n_clusters: int
) -> Dict[int, str]:
    """
    为每个聚类训练单独的模型
    Args:
        X: 特征数据
        y: 目标变量
        clusters: 聚类标签
        model_name: 模型名称（'lightgbm', 'xgboost', 'catboost', 'gradient_boosting'）
        output_dir: 模型保存目录
        params: 模型参数
        n_clusters: 聚类数量
    Returns:
        models: 聚类ID到模型路径的映射字典
    """
    models = {}
    
    # 检查输出目录是否存在
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"模型将保存在目录: {output_dir}")

    # 验证输入数据
    if len(X) != len(y) or len(X) != len(clusters):
        raise ValueError(f"输入数据长度不匹配: X={len(X)}, y={len(y)}, clusters={len(clusters)}")

    # 记录基本信息
    logging.info(f"开始训练 {model_name} 模型")
    logging.info(f"总样本数: {len(X)}")
    logging.info(f"特征数量: {X.shape[1]}")
    logging.info(f"聚类数量: {n_clusters}")

    for cluster_id in range(n_clusters):
        logging.info(f"\n=== 训练聚类 {cluster_id} 的模型 ===")
        
        # 获取当前聚类的数据
        cluster_mask = clusters == cluster_id
        X_cluster = X[cluster_mask].copy()
        y_cluster = y[cluster_mask].copy()
        
        # 记录当前聚类的基本信息
        logging.info(f"聚类 {cluster_id} 的样本数量: {len(X_cluster)}")
        if len(y_cluster) > 0:
            logging.info(f"目标变量范围: {y_cluster.min():.2f} - {y_cluster.max():.2f}")
            logging.info(f"目标变量均值: {y_cluster.mean():.2f}")

        # 检查样本数量
        if len(X_cluster) < 10:
            logging.warning(f"聚类 {cluster_id} 的样本数量过少({len(X_cluster)}), 跳过该聚类")
            continue

        # 设置文件路径
        cluster_train_file = os.path.join(output_dir, f"train_cluster_{cluster_id}.csv")
        model_save_path = os.path.join(output_dir, f"{model_name}_model_cluster_{cluster_id}.pkl")
        prediction_output_path = os.path.join(output_dir, f"prediction_cluster_{cluster_id}.csv")

        try:
            # 保存训练数据
            logging.info(f"保存聚类 {cluster_id} 的训练数据")
            cluster_data = X_cluster.copy()
            cluster_data['price'] = y_cluster
            cluster_data.to_csv(cluster_train_file, index=False)

            # 训练模型
            logging.info(f"开始训练聚类 {cluster_id} 的模型")
            if model_name == 'lightgbm':
                import lightgbm as lgb
                train_data = lgb.Dataset(X_cluster, y_cluster)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9
                }
                model = lgb.train(params, train_data, num_boost_round=100)
                
            elif model_name == 'xgboost':
                import xgboost as xgb
                train_data = xgb.DMatrix(X_cluster, y_cluster)
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'eta': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
                model = xgb.train(params, train_data, num_boost_round=100)
                
            elif model_name == 'catboost':
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.05,
                    depth=6,
                    loss_function='RMSE',
                    verbose=False
                )
                model.fit(X_cluster, y_cluster)
                
            elif model_name == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42
                )
                model.fit(X_cluster, y_cluster)
            
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")

            # 验证模型对象
            if not hasattr(model, 'predict'):
                raise AttributeError(f"训练得到的模型对象没有 predict 方法")

            # 保存模型
            logging.info(f"保存聚类 {cluster_id} 的模型")
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)

            # 进行简单的验证预测
            if model_name == 'lightgbm':
                preds = model.predict(X_cluster)
            elif model_name == 'xgboost':
                preds = model.predict(xgb.DMatrix(X_cluster))
            else:
                preds = model.predict(X_cluster)

            # 记录验证结果
            mse = np.mean((preds - y_cluster) ** 2) ** 0.5
            logging.info(f"聚类 {cluster_id} 的训练集 RMSE: {mse:.2f}")

            # 将模型路径添加到返回字典
            models[cluster_id] = model_save_path
            logging.info(f"聚类 {cluster_id} 的模型训练完成并保存")

        except Exception as e:
            logging.error(f"训练聚类 {cluster_id} 的模型时发生错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

        finally:
            # 清理临时文件
            if os.path.exists(cluster_train_file):
                os.remove(cluster_train_file)
                logging.info(f"删除临时训练文件: {cluster_train_file}")

    # 最终统计
    logging.info("\n=== 训练完成统计 ===")
    logging.info(f"成功训练的聚类数量: {len(models)}")
    logging.info(f"失败的聚类数量: {n_clusters - len(models)}")
    
    if len(models) == 0:
        logging.warning("警告：没有任何模型被成功训练")
    
    return models

def load_models(models: Dict[int, str]) -> Dict[int, Any]:
    """
    加载训练好的模型
    Args:
        models: 聚类ID到模型路径的映射字典
    Returns:
        loaded_models: 聚类ID到加载模型对象的映射字典
    """
    loaded_models = {}
    for cluster_id, model_path in models.items():
        try:
            # 读取模型文件
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            
            # 从字典中获取实际的模型对象
            if isinstance(model_dict, dict):
                if 'model' in model_dict:
                    model = model_dict['model']
                elif 'best_model' in model_dict:
                    model = model_dict['best_model']
                else:
                    logging.error(f"模型文件 {model_path} 中没有找到模型对象")
                    continue
            else:
                model = model_dict  # 如果直接是模型对象

            loaded_models[cluster_id] = model
            logging.info(f"成功加载模型: {model_path}")
            
            # 验证模型对象
            if not hasattr(model, 'predict'):
                logging.error(f"加载的模型对象 {model_path} 没有 predict 方法")
                continue
                
        except Exception as e:
            logging.error(f"加载模型 {model_path} 时发生错误: {str(e)}")
            continue
            
    if not loaded_models:
        logging.error("没有成功加载任何模型")
        
    return loaded_models

def predict_clusters(
    X_test: pd.DataFrame,
    kmeans: KMeans,
    loaded_models: Dict[int, Any],
    features_for_clustering: list,
    model_name: str
) -> pd.DataFrame:
    """
    对测试数据进行聚类预测并生成最终预测结果
    Args:
        X_test: 测试数据
        kmeans: 训练好的 KMeans 模型
        loaded_models: 聚类ID到加载模型对象的映射字典
        features_for_clustering: 用于聚类的特征
        model_name: 模型名称
    Returns:
        submission: 包含预测结果的DataFrame
    """
    # 记录测试集的原始顺序
    X_test = X_test.reset_index(drop=True)

    # 使用训练好的 KMeans 模型进行聚类
    cluster_features_test = X_test[features_for_clustering].values
    test_clusters = kmeans.predict(cluster_features_test)
    logging.info("测试集聚类完成")

    # 添加聚类标签到测试集
    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster'] = test_clusters

    # 初始化预测数组
    final_predictions = np.zeros(len(X_test))

    # 对每个聚类进行预测
    for cluster_id, model in loaded_models.items():
        cluster_mask = X_test_with_cluster['cluster'] == cluster_id
        n_samples = cluster_mask.sum()

        if n_samples == 0:
            logging.warning(f"聚类 {cluster_id} 在测试集中没有样本")
            continue

        logging.info(f"预测聚类 {cluster_id} 的 {n_samples} 个样本")
        X_test_cluster = X_test_with_cluster[cluster_mask].drop(columns=['cluster'])

        try:
            # 直接预测，不使用对数转换
            if model_name == 'lightgbm':
                preds = model.predict(X_test_cluster)
            elif model_name in ['xgboost', 'catboost', 'gradient_boosting']:
                preds = model.predict(X_test_cluster)
            else:
                logging.error(f"不支持的模型类型: {model_name}")
                continue

            # 后处理预测值
            preds = np.clip(preds, 700.00, 2900000.00)
            
            # 记录预测统计信息
            logging.info(f"聚类 {cluster_id} 预测值统计:")
            logging.info(f"最小值: {np.min(preds):.2f}")
            logging.info(f"最大值: {np.max(preds):.2f}")
            logging.info(f"平均值: {np.mean(preds):.2f}")
            logging.info(f"中位数: {np.median(preds):.2f}")

            final_predictions[cluster_mask] = preds

        except Exception as e:
            logging.error(f"预测聚类 {cluster_id} 时发生错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    # 检查是否有未预测的样本
    missing_mask = final_predictions == 0
    if missing_mask.any():
        n_missing = missing_mask.sum()
        logging.warning(f"有 {n_missing} 个样本未被预测，使用默认值 700.00")
        final_predictions[missing_mask] = 700.00

    # 创建提交文件
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })

    return submission

def predict_clusters(
    X_test: pd.DataFrame,
    kmeans: KMeans,
    loaded_models: Dict[int, Any],
    features_for_clustering: list,
    model_name: str
) -> pd.DataFrame:
    """
    对测试数据进行聚类预测并生成最终预测结果
    Args:
        X_test: 测试数据
        kmeans: 训练好的 KMeans 模型
        loaded_models: 聚类ID到加载模型对象的映射字典
        features_for_clustering: 用于聚类的特征
        model_name: 模型名称
    Returns:
        submission: 包含预测结果的DataFrame
    """
    # 记录测试集的原始顺序
    X_test = X_test.reset_index(drop=True)

    # 使用训练好的 KMeans 模型进行聚类
    cluster_features_test = X_test[features_for_clustering].values
    test_clusters = kmeans.predict(cluster_features_test)
    logging.info("测试集聚类完成")

    # 添加聚类标签到测试集
    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster'] = test_clusters

    # 初始化预测数组
    final_predictions = np.zeros(len(X_test))

    # 对每个聚类进行预测
    for cluster_id, model in loaded_models.items():
        cluster_mask = X_test_with_cluster['cluster'] == cluster_id
        n_samples = cluster_mask.sum()

        if n_samples == 0:
            logging.warning(f"聚类 {cluster_id} 在测试集中没有样本")
            continue

        logging.info(f"预测聚类 {cluster_id} 的 {n_samples} 个样本")
        X_test_cluster = X_test_with_cluster[cluster_mask].drop(columns=['cluster'])

        try:
            # 预测
            if model_name == 'lightgbm':
                preds_log = model.predict(X_test_cluster, num_iteration=model.best_iteration)
                preds = np.expm1(preds_log)  # 还原对数变换
            elif model_name in ['xgboost', 'catboost', 'gradient_boosting']:
                preds_log = model.predict(X_test_cluster)
                preds = np.expm1(preds_log)  # 还原对数变换
            else:
                logging.error(f"不支持的模型类型: {model_name}")
                continue

            # 后处理预测值
            preds = np.clip(preds, 700.00, 2900000.00)

            final_predictions[cluster_mask] = preds

        except Exception as e:
            logging.error(f"预测聚类 {cluster_id} 时发生错误: {str(e)}")
            continue

    # 检查是否有未预测的样本
    missing_mask = final_predictions == 0
    if missing_mask.any():
        n_missing = missing_mask.sum()
        logging.warning(f"有 {n_missing} 个样本未被预测，使用默认值 700.00")
        final_predictions[missing_mask] = 700.00  # 使用最小价格作为默认值

    # 创建提交文件
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })

    return submission

def main():
    # 定义文件路径和参数
    train_file_path = os.getenv('FINAL_REPORT_TRAIN_DATA')
    test_file_path = os.getenv('FINAL_REPORT_TEST_DATA')
    submission_output_path = os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH')
    output_dir = 'final_report/approach1/clustering'
    model_name = 'lightgbm'  # 可选: 'lightgbm', 'xgboost', 'catboost', 'gradient_boosting'

    # 检查环境变量
    missing_vars = []
    if not train_file_path:
        missing_vars.append('FINAL_REPORT_TRAIN_DATA')
    if not test_file_path:
        missing_vars.append('FINAL_REPORT_TEST_DATA')
    if not submission_output_path:
        missing_vars.append('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH')

    if missing_vars:
        raise EnvironmentError(f"缺少环境变量: {', '.join(missing_vars)}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载训练数据
    X_train, y_train = load_and_preprocess_data(train_file_path, is_train=True)

    # 选择用于聚类的特征
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']

    # 寻找最优聚类数量
    optimal_clusters = find_optimal_clusters(X_train, max_clusters=5, features_for_clustering=features_for_clustering)
    # X: pd.DataFrame, max_clusters: int = 3,features_for_clustering: list = ['depreciation', 'coe', 'dereg_value']

    # 创建价格聚类
    kmeans, train_clusters, cluster_info = create_price_clusters(
        X=X_train,
        y=y_train,
        n_clusters=optimal_clusters,
        features_for_clustering=features_for_clustering
    )
# def create_price_clusters(X: pd.DataFrame, n_clusters: int,
#                          features_for_clustering: list = ['depreciation', 'coe', 'dereg_value']) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    # 保存聚类模型
    kmeans_model_path = os.path.join(output_dir, 'kmeans_model.pkl')
    with open(kmeans_model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    logging.info(f"KMeans 模型已保存到 {kmeans_model_path}")

    # 为每个聚类训练模型
    models = train_models_for_clusters(
        X=X_train,
        y=y_train,
        clusters=train_clusters,
        model_name=model_name,
        output_dir=output_dir,
        params={},  # 这里传递空字典，因为模型训练函数已经定义了参数
        n_clusters=optimal_clusters
    )

    if not models:
        logging.error("没有任何模型被训练。退出程序。")
        return

    # 加载测试数据
    X_test, _ = load_and_preprocess_data(test_file_path, is_train=False)

    # 加载所有训练好的模型
    loaded_models = load_models(models)

    # 预测测试集
    submission = predict_clusters(
        X_test=X_test,
        kmeans=kmeans,
        loaded_models=loaded_models,
        features_for_clustering=features_for_clustering,
        model_name=model_name
    )

    # 保存预测结果
    submission.to_csv(submission_output_path, index=False)
    logging.info(f"预测结果已保存到 {submission_output_path}")

    # 输出预测统计信息
    logging.info("\n预测统计信息:")
    logging.info(f"最小值: {submission['Predicted'].min():.2f}")
    logging.info(f"最大值: {submission['Predicted'].max():.2f}")
    logging.info(f"均值: {submission['Predicted'].mean():.2f}")
    logging.info(f"中位数: {submission['Predicted'].median():.2f}")

    logging.info("模型训练和预测流程完成")

if __name__ == '__main__':
    main()
