# error_analysis.py
def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from joblib import Parallel, delayed

    # 获取环境变量
    train_file_path = os.getenv('FINAL_REPORT_KANGLE_TRAIN_DATA')
    output_dir = 'final_report/approach1/without_cluster/error_analysis'
    
    model_configs = [
        {
            'model_save_path': os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_GRADIENT_BOOSTING_WEIGHT_PATH'),
            'train_file_path': train_file_path,
            'output_dir': os.path.join(output_dir, 'mistake_analysis_gradient_boosting'),
            'heatmap_dir': os.path.join(output_dir, 'mistake_analysis_gradient_boosting', 'heatmaps_gradient_boosting'),
            'cluster_range': (2, 10),
            'random_state': 42,
            'model_name': 'gradient_boosting'
        },
        {
            'model_save_path': os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_LightGBM_WEIGHT_PATH'),
            'train_file_path': train_file_path,
            'output_dir': os.path.join(output_dir, 'mistake_analysis_lightgbm'),
            'heatmap_dir': os.path.join(output_dir, 'mistake_analysis_lightgbm', 'heatmaps_lightgbm'),
            'cluster_range': (2, 10),
            'random_state': 42,
            'model_name': 'lightgbm'
        },
        {
            'model_save_path': os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_XGBoost_WEIGHT_PATH'),
            'train_file_path': train_file_path,
            'output_dir': os.path.join(output_dir, 'mistake_analysis_xgboost'),
            'heatmap_dir': os.path.join(output_dir, 'mistake_analysis_xgboost', 'heatmaps_xgboost'),
            'cluster_range': (2, 10),
            'random_state': 42,
            'model_name': 'xgboost'
        },
        {
            'model_save_path': os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_CATBOOST_WEIGHT_PATH'),
            'train_file_path': train_file_path,
            'output_dir': os.path.join(output_dir, 'mistake_analysis_catboost'),
            'heatmap_dir': os.path.join(output_dir, 'mistake_analysis_catboost', 'heatmaps_catboost'),
            'cluster_range': (2, 10),
            'random_state': 42,
            'model_name': 'catboost'
        },
        # 添加更多模型配置
    ]

    # 并行处理多个模型的误差分析
    Parallel(n_jobs=-1)(
        delayed(perform_error_analysis)(
            model_save_path=config['model_save_path'],
            train_file_path=config['train_file_path'],
            output_dir=config['output_dir'],
            heatmap_dir=config['heatmap_dir'],
            cluster_range=config['cluster_range'],
            random_state=config['random_state'],
            model_name=config['model_name']
        )
        for config in model_configs
    )




import pandas as pd
import numpy as np
import pickle
import logging
import os
import json
from typing import Any, Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from final_report.approach1.without_cluster.models.common_utils import (
    load_data,
    preprocess_features,
    post_process_predictions,
    verify_saved_model,
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error_analysis.log'),
        logging.StreamHandler()
    ]
)

def convert_numpy_to_python(obj):
    """
    将包含 numpy 数据类型的对象转换为原生 Python 数据类型。
    
    Args:
        obj: 需要转换的对象。
    
    Returns:
        转换后的对象。
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj

def load_model(model_save_path: str) -> Dict[str, Any]:
    """
    加载保存的模型。

    Args:
        model_save_path (str): 模型保存路径。

    Returns:
        Dict[str, Any]: 加载的模型字典。
    """
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"模型文件未找到: {model_save_path}")
    
    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)
    
    if not verify_saved_model(model_save_path):
        raise ValueError("加载的模型不符合预期结构。")
    
    logging.info(f"成功加载模型: {model_save_path}")
    return model

def perform_inference(
    model: Dict[str, Any],
    train_file_path: str,
    model_type: str = 'lightgbm'  # 根据不同模型类型调整预测方法
) -> pd.DataFrame:
    """
    使用加载的模型对指定的数据集进行推理，生成inference_price和mistake列。

    Args:
        model (Dict[str, Any]): 加载的模型字典。
        train_file_path (str): 数据集文件路径（包含price，但推理时忽略）。
        model_type (str): 模型类型，用于选择预测方法。

    Returns:
        pd.DataFrame: 包含inference_price和mistake列的数据框。
    """
    X, y = load_data(train_file_path)  # y将被忽略
    logging.info(f"开始对数据集 {train_file_path} 进行推理。数据形状: {X.shape}")
    
    final_predictions = np.zeros(len(X))
    
    for i, model_dict in enumerate(model['models'], 1):
        logging.info(f"使用模型 {i} 进行推理")
        trained_model = model_dict['model']
        preprocessors = model_dict['preprocessors']
        
        # 预处理
        X_processed, _, _ = preprocess_features(
            X,
            y=None,
            num_imputer=preprocessors['num_imputer'],
            scaler=preprocessors['scaler']
        )
        
        # 预测
        if model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_processed)
            preds_log = trained_model.predict(dtest)
            preds = np.expm1(preds_log)
        elif model_type == 'lightgbm':
            preds_log = trained_model.predict(X_processed, num_iteration=trained_model.best_iteration)
            preds = np.expm1(preds_log)
        elif model_type == 'gradient_boosting':
            preds = trained_model.predict(X_processed)
        elif model_type == 'catboost':
            preds = trained_model.predict(X_processed)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        final_predictions += preds
    
    # 取平均
    final_predictions /= len(model['models'])
    
    # 后处理
    final_predictions = post_process_predictions(final_predictions)
    
    # 生成结果数据框
    results = X.copy()
    results['inference_price'] = final_predictions
    if y is not None:
        results['price'] = y
        results['mistake'] = results['price'] - results['inference_price']
    else:
        logging.warning("没有目标变量 'price'，无法计算 'mistake'")
        results['mistake'] = None
    
    logging.info("推理完成，生成inference_price和mistake列。")
    return results

def cluster_errors(
    data: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    对mistake进行聚类，选择最佳聚类数量（基于轮廓系数）。

    Args:
        data (pd.DataFrame): 包含mistake列的数据框。
        min_clusters (int): 最小聚类数。
        max_clusters (int): 最大聚类数。
        random_state (int): 随机种子。

    Returns:
        Tuple[pd.DataFrame, int]: 原数据框添加了cluster列，以及最佳聚类数。
    """
    scaler = StandardScaler()
    mistakes_scaled = scaler.fit_transform(data[['mistake']])
    
    best_score = -1
    best_k = min_clusters
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        clusters = kmeans.fit_predict(mistakes_scaled)
        score = silhouette_score(mistakes_scaled, clusters)
        logging.info(f"聚类数: {k}, 轮廓系数: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    logging.info(f"选择聚类数: {best_k}，轮廓系数: {best_score:.4f}")
    
    # 最佳聚类数
    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    clusters = kmeans.fit_predict(mistakes_scaled)
    data['cluster'] = clusters
    
    return data, best_k

def analyze_cluster(
    data: pd.DataFrame,
    cluster_id: int,
    heatmap_dir: str,
    model_name: str
) -> Dict[str, Any]:
    """
    分析单个聚类，找出与误差关系最大的特征，并生成热力图。

    Args:
        data (pd.DataFrame): 包含误差和特征的数据框。
        cluster_id (int): 当前聚类的ID。
        heatmap_dir (str): 热力图保存目录。
        model_name (str): 模型名称，用于命名文件。

    Returns:
        Dict[str, Any]: 分析结果，包括特征重要性、RMSE、统计信息和热力图路径。
    """
    cluster_data = data[data['cluster'] == cluster_id]
    logging.info(f"开始分析聚类 {cluster_id}，样本数量: {len(cluster_data)}")
    
    # 计算每个特征与误差的相关性
    correlation = cluster_data.drop(columns=['price', 'inference_price', 'mistake', 'cluster']).corrwith(cluster_data['mistake']).abs()
    top_features = correlation.sort_values(ascending=False).head(5).index.tolist()
    
    logging.info(f"聚类 {cluster_id} 与误差关系最强的特征: {top_features}")
    
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(cluster_data['price'], cluster_data['inference_price']))
    
    # 计算统计信息
    count = len(cluster_data)
    proportion = count / len(data)
    mean_mistake = cluster_data['mistake'].mean()
    max_mistake = cluster_data['mistake'].max()
    min_mistake = cluster_data['mistake'].min()
    
    logging.info(f"聚类 {cluster_id} 的 RMSE: {rmse:.4f}, Count: {count}, Proportion: {proportion:.2%}, Mean: {mean_mistake:.4f}, Max: {max_mistake:.4f}, Min: {min_mistake:.4f}")
    
    # 生成热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cluster_data[top_features + ['mistake']].corr(), annot=True, cmap='coolwarm')
    plt.title(f'Cluster {cluster_id} Feature vs Mistake Correlation Heatmap')
    
    heatmap_path = os.path.join(heatmap_dir, f'cluster_{cluster_id}_heatmap_{model_name}.png')
    os.makedirs(heatmap_dir, exist_ok=True)
    plt.savefig(heatmap_path)
    plt.close()
    
    logging.info(f"生成聚类 {cluster_id} 的热力图: {heatmap_path}")
    
    return {
        'cluster_id': int(cluster_id),  # 转换为原生 int 类型
        'rmse': rmse,
        'count': int(count),
        'proportion': proportion,
        'mean_mistake': mean_mistake,
        'max_mistake': max_mistake,
        'min_mistake': min_mistake,
        'top_features': top_features,
        'heatmap_path': heatmap_path
    }

def plot_mistake_distribution_with_clusters(
    data: pd.DataFrame,
    cluster_range: Tuple[int, int],
    output_dir: str,
    model_name: str
) -> str:
    """
    生成误差分布的统计图，并标明误差聚类得到的区间。

    Args:
        data (pd.DataFrame): 包含mistake和cluster列的数据框。
        cluster_range (Tuple[int, int]): 聚类数范围（最小聚类数, 最大聚类数）。
        output_dir (str): 图像保存目录。
        model_name (str): 模型名称，用于命名文件。

    Returns:
        str: 统计图的保存路径。
    """
    plt.figure(figsize=(10,6))
    sns.histplot(data['mistake'], bins=50, kde=True, color='skyblue', label='Mistake Distribution')
    
    # 获取聚类中心
    cluster_centers = data.groupby('cluster')['mistake'].mean().sort_values().values
    for idx, center in enumerate(cluster_centers):
        plt.axvline(center, color=f'C{idx+1}', linestyle='--', label=f'Cluster {idx} Center')
    
    # 添加统计信息
    stats = data.groupby('cluster')['mistake'].agg(['count', 'mean', 'max', 'min'])
    total = len(data)
    for idx, row in stats.iterrows():
        proportion = row['count'] / total
        plt.text(row['mean'], plt.ylim()[1]*0.9 - idx*plt.ylim()[1]*0.05,
                 f'Cluster {idx}:\nCount: {row["count"]}\nProportion: {proportion:.2%}\nMean: {row["mean"]:.2f}\nMax: {row["max"]:.2f}\nMin: {row["min"]:.2f}',
                 color=f'C{idx+1}',
                 bbox=dict(facecolor='white', alpha=0.6))
    
    plt.title(f'Mistake Distribution with Cluster Centers - {model_name}')
    plt.xlabel('Mistake')
    plt.ylabel('Frequency')
    plt.legend()
    
    distribution_plot_path = os.path.join(output_dir, f'mistake_distribution_{model_name}.png')
    plt.savefig(distribution_plot_path)
    plt.close()
    
    logging.info(f"生成误差分布图: {distribution_plot_path}")
    
    return distribution_plot_path

def generate_overall_heatmap(
    data: pd.DataFrame,
    heatmap_dir: str,
    model_name: str
) -> str:
    """
    生成整体特征与预测误差的热力图。

    Args:
        data (pd.DataFrame): 包含误差和特征的数据框。
        heatmap_dir (str): 热力图保存目录。
        model_name (str): 模型名称，用于命名文件。

    Returns:
        str: 热力图的保存路径。
    """
    logging.info("开始生成整体特征与误差的热力图。")
    
    correlation = data.drop(columns=['price', 'inference_price', 'mistake', 'cluster']).corrwith(data['mistake']).abs()
    top_features = correlation.sort_values(ascending=False).head(10).index.tolist()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(data[top_features + ['mistake']].corr(), annot=True, cmap='coolwarm')
    plt.title('Overall Feature vs Mistake Correlation Heatmap')
    
    heatmap_path = os.path.join(heatmap_dir, f'overall_heatmap_{model_name}.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    logging.info(f"生成整体热力图: {heatmap_path}")
    
    return heatmap_path

def analyze_cluster_features(
    data: pd.DataFrame,
    model_name: str,
    heatmap_dir: str
) -> Dict[str, Any]:
    """
    对所有聚类进行分析，并生成相应的热力图。

    Args:
        data (pd.DataFrame): 包含误差和聚类标签的数据框。
        model_name (str): 模型名称，用于命名文件。
        heatmap_dir (str): 热力图保存目录。

    Returns:
        Dict[str, Any]: 包含所有聚类分析结果和整体热力图路径的字典。
    """
    analysis_results = {}
    for cluster_id in data['cluster'].unique():
        analysis = analyze_cluster(data, cluster_id, heatmap_dir, model_name)
        analysis_results[f'cluster_{cluster_id}'] = analysis
    
    # 生成整体热力图
    overall_heatmap_path = generate_overall_heatmap(data, heatmap_dir, model_name)
    analysis_results['overall'] = {
        'heatmap_path': overall_heatmap_path
    }
    
    return analysis_results

def save_analysis_results(
    analysis_results: Dict[str, Any],
    output_dir: str,
    model_name: str
):
    """
    保存分析结果为JSON文件。

    Args:
        analysis_results (Dict[str, Any]): 分析结果字典。
        output_dir (str): 分析结果保存目录。
        model_name (str): 模型名称，用于命名文件。
    """
    # 转换所有 numpy 类型为原生 Python 类型
    analysis_results = convert_numpy_to_python(analysis_results)
    
    analysis_file = os.path.join(output_dir, f'{model_name}_error_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    logging.info(f"误差分析完成，分析结果已保存为 {analysis_file}")

def generate_summary_table(models_analysis_dir: str, summary_output_path: str):
    """
    生成一个汇总表，展示每个聚类中各模型的 RMSE，并标注出最佳模型。

    Args:
        models_analysis_dir (str): 各模型分析结果的根目录。
        summary_output_path (str): 汇总表的保存路径（CSV 格式）。
    """
    summary_data = {}
    
    for model_name in os.listdir(models_analysis_dir):
        model_dir = os.path.join(models_analysis_dir, model_name)
        analysis_file = os.path.join(model_dir, f'{model_name}_error_analysis.json')
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            for key, value in analysis.items():
                if key.startswith('cluster_'):
                    cluster_id = key.split('_')[1]
                    rmse = value.get('rmse')
                    if cluster_id not in summary_data:
                        summary_data[cluster_id] = {}
                    summary_data[cluster_id][model_name] = rmse
    
    # 转换为 DataFrame
    summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
    summary_df.index.name = 'cluster_id'
    summary_df.reset_index(inplace=True)
    
    # 标注最佳模型
    best_models = summary_df.set_index('cluster_id').idxmin(axis=1).rename('best_model').reset_index()
    summary_df = summary_df.merge(best_models, on='cluster_id')
    
    # 保存汇总表
    summary_df.to_csv(summary_output_path, index=False)
    logging.info(f"汇总表已保存为 {summary_output_path}")

def plot_rmse_summary(
    summary_df: pd.DataFrame,
    output_dir: str
):
    """
    绘制每个聚类中各模型的 RMSE，标注最佳模型。

    Args:
        summary_df (pd.DataFrame): 汇总表数据框。
        output_dir (str): 图像保存目录。
    """
    plt.figure(figsize=(12, 8))
    models = summary_df.columns.drop(['cluster_id', 'best_model'])
    for model in models:
        plt.plot(summary_df['cluster_id'], summary_df[model], marker='o', label=model)
    
    # 标注最佳模型
    for idx, row in summary_df.iterrows():
        best_model = row['best_model']
        plt.scatter(row['cluster_id'], row[best_model], color='red', s=100, edgecolor='k', zorder=5)
        plt.text(row['cluster_id'], row[best_model], ' Best', color='red', fontsize=9, verticalalignment='bottom')
    
    plt.title('RMSE per Cluster per Model')
    plt.xlabel('Cluster ID')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'rmse_summary_per_cluster.png')
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"RMSE 汇总图已保存为 {plot_path}")

def analyze_errors(
    inference_results: pd.DataFrame,
    model_type: str,
    output_dir: str,
    heatmap_dir: str,
    cluster_range: Tuple[int, int],
    random_state: int = 42
) -> None:
    """
    对推理结果进行误差分析，生成分析报告和热力图。

    Args:
        inference_results (pd.DataFrame): 包含inference_price和mistake列的数据框。
        model_type (str): 模型类型，用于命名分析文件。
        output_dir (str): 误差分析结果保存目录。
        heatmap_dir (str): 热力图保存目录。
        cluster_range (Tuple[int, int]): 聚类数范围。
        random_state (int): 随机种子。
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 检查 'price' 列是否存在
    if 'price' not in inference_results.columns:
        logging.error("'price' 列缺失，无法计算 'mistake'")
        return
    
    # 聚类误差
    min_clusters, max_clusters = cluster_range
    clustered_data, best_k = cluster_errors(inference_results, min_clusters=min_clusters, max_clusters=max_clusters, random_state=random_state)
    
    # 保存聚类结果
    clustered_file = os.path.join(output_dir, f'{model_type}_clustered_results.csv')
    clustered_data.to_csv(clustered_file, index=False)
    logging.info(f"聚类结果已保存为 {clustered_file}")
    
    # 生成误差分布图
    distribution_plot_path = plot_mistake_distribution_with_clusters(clustered_data, cluster_range, output_dir, model_type)
    
    # 进行误差分析
    analysis_results = analyze_cluster_features(clustered_data, model_type, heatmap_dir)
    
    # 保存分析结果为JSON
    save_analysis_results(analysis_results, output_dir, model_type)

def perform_error_analysis(
    model_save_path: str,
    train_file_path: str,
    output_dir: str,
    heatmap_dir: str,
    cluster_range: Tuple[int, int],
    random_state: int,
    model_name: str
):
    """
    执行单个模型的错误分析流程。

    Args:
        model_save_path (str): 保存的模型路径。
        train_file_path (str): 训练数据路径（包含price，但推理时忽略）。
        output_dir (str): 误差分析结果保存目录。
        heatmap_dir (str): 热力图保存目录。
        cluster_range (Tuple[int, int]): 聚类数范围（最小聚类数, 最大聚类数）。
        random_state (int): 随机种子。
        model_name (str): 模型名称。
    """
    try:
        logging.info(f"开始错误分析流程，模型: {model_name}")
        
        # 加载模型
        model = load_model(model_save_path)
        
        # 进行推理并计算误差
        inference_results = perform_inference(model, train_file_path, model_type=model_name)
        
        # 保存推理结果
        inference_file = os.path.join(output_dir, f'{model_name}_inference_results.csv')
        os.makedirs(output_dir, exist_ok=True)
        inference_results.to_csv(inference_file, index=False)
        logging.info(f"推理结果已保存为 {inference_file}")
        
        # 进行误差分析
        analyze_errors(
            inference_results=inference_results,
            model_type=model_name,
            output_dir=output_dir,
            heatmap_dir=heatmap_dir,
            cluster_range=cluster_range,
            random_state=random_state
        )
        
    except Exception as e:
        logging.error(f"错误分析过程中出现错误（模型: {model_name}）: {str(e)}")
        raise

if __name__ == '__main__':
    main()
