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
    
    # 生成汇总表
    summary_output_path = os.path.join(output_dir, 'rmse_summary_per_category.csv')
    generate_summary_table(models_analysis_dir=output_dir, summary_output_path=summary_output_path)
    
    # 绘制汇总图
    summary_df = pd.read_csv(summary_output_path)
    plot_rmse_summary(summary_df, output_dir=output_dir)
    plot_proportion_summary(summary_df, output_dir=output_dir)




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
    verify_saved_model
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

def define_mistake_categories() -> List[Tuple[int, int]]:
    """
    定义误差绝对值的分类区间。

    Returns:
        List[Tuple[int, int]]: 分类区间列表。
    """
    # 定义分类区间，确保覆盖所有可能的误差值
    categories = [
        (0, 2000),
        (2001, 3000),
        (3001, 5000),
        (5001, 8000),
        (8001, 10000),
        (10001, 12000),
        (12001, 14000),
        (14001, 16000),
        (16001, 20000)
    ]
    return categories

def assign_mistake_category(mistake: float, categories: List[Tuple[int, int]]) -> int:
    """
    根据误差的绝对值将其分配到相应的分类区间。

    Args:
        mistake (float): 预测误差。
        categories (List[Tuple[int, int]]): 分类区间列表。

    Returns:
        int: 分类ID。
    """
    abs_mistake = abs(mistake)
    for idx, (lower, upper) in enumerate(categories):
        if lower <= abs_mistake <= upper:
            return idx
    return len(categories)  # 超出定义区间的分类

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
    features = cluster_data.drop(columns=['price', 'inference_price', 'mistake', 'cluster', 'mistake_category'])
    correlation = features.corrwith(cluster_data['mistake']).abs()
    top_features = correlation.sort_values(ascending=False).head(5).index.tolist()
    
    logging.info(f"聚类 {cluster_id} 与误差关系最强的特征: {top_features}")
    
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(cluster_data['price'], cluster_data['inference_price']))
    
    # 计算统计信息
    count = len(cluster_data)
    proportion = count / len(data)
    mean_mistake = cluster_data['mistake'].abs().mean()
    max_mistake = cluster_data['mistake'].abs().max()
    min_mistake = cluster_data['mistake'].abs().min()
    
    logging.info(f"聚类 {cluster_id} 的 RMSE: {rmse:.4f}, Count: {count}, Proportion: {proportion:.2%}, "
                 f"Mean Absolute Mistake: {mean_mistake:.4f}, Max Mistake: {max_mistake:.4f}, "
                 f"Min Mistake: {min_mistake:.4f}")
    
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
        'mean_absolute_mistake': mean_mistake,
        'max_mistake': max_mistake,
        'min_mistake': min_mistake,
        'top_features': top_features,
        'heatmap_path': heatmap_path
    }

def plot_mistake_distribution_with_categories(
    data: pd.DataFrame,
    categories: List[Tuple[int, int]],
    output_dir: str,
    model_name: str
) -> str:
    """
    生成误差分布的统计图，并标明误差类别。改进了文本标注的布局以避免重叠。

    Args:
        data (pd.DataFrame): 包含mistake和mistake_category列的数据框。
        categories (List[Tuple[int, int]]): 误差分类区间列表。
        output_dir (str): 图像保存目录。
        model_name (str): 模型名称，用于命名文件。

    Returns:
        str: 统计图的保存路径。
    """
    plt.figure(figsize=(15, 10))  # 增加图像尺寸
    
    # 创建主图和用于文本的子图
    gs = plt.GridSpec(1, 5)
    ax1 = plt.subplot(gs[0, :3])  # 主分布图使用左边3/5
    ax2 = plt.subplot(gs[0, 3:])  # 文本注释使用右边2/5
    
    # 在左侧子图绘制分布
    sns.histplot(data=data, x='mistake_abs', bins=50, kde=True, color='skyblue', 
                label='Mistake Distribution', ax=ax1)
    
    # 获取类别中心
    category_centers = data.groupby('mistake_category')['mistake_abs'].mean().sort_values().values
    for idx, center in enumerate(category_centers):
        ax1.axvline(center, color=f'C{idx+1}', linestyle='--', 
                   label=f'Category {idx} Center')
    
    ax1.set_title('Mistake Distribution')
    ax1.set_xlabel('Absolute Mistake')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 在右侧子图添加文本统计信息
    ax2.axis('off')  # 关闭坐标轴
    
    # 计算统计信息
    stats = data.groupby('mistake_category')['mistake_abs'].agg(['count', 'mean', 'max', 'min'])
    total = len(data)
    
    # 创建文本内容
    text_content = 'Category Statistics:\n\n'
    for idx, row in stats.iterrows():
        proportion = row['count'] / total
        text_content += (f'Category {idx}:\n'
                        f'Count: {row["count"]}\n'
                        f'Proportion: {proportion:.2%}\n'
                        f'Mean: {row["mean"]:.2f}\n'
                        f'Max: {row["max"]:.2f}\n'
                        f'Min: {row["min"]:.2f}\n\n')
    
    # 添加文本框
    ax2.text(0, 0.95, text_content,
             transform=ax2.transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle(f'Mistake Distribution Analysis - {model_name}')
    
    # 保存图像
    distribution_plot_path = os.path.join(output_dir, f'mistake_distribution_{model_name}.png')
    plt.savefig(distribution_plot_path, bbox_inches='tight', dpi=300)
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
    
    # 移除非特征列
    non_feature_cols = ['price', 'inference_price', 'mistake', 'mistake_category', 'mistake_abs']
    correlation = data.drop(columns=non_feature_cols).corrwith(data['mistake']).abs()
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
    for cluster_id in sorted(data['mistake_category'].unique()):
        analysis = analyze_cluster(data, cluster_id, heatmap_dir, model_name)
        analysis_results[f'category_{cluster_id}'] = analysis
    
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
    生成一个汇总表，展示每个分类中各模型的 RMSE，并标注出最佳模型。

    Args:
        models_analysis_dir (str): 各模型分析结果的根目录。
        summary_output_path (str): 汇总表的保存路径（CSV 格式）。
    """
    summary_data = {}
    
    # 遍历每个模型目录
    for model_dir in os.listdir(models_analysis_dir):
        full_model_dir = os.path.join(models_analysis_dir, model_dir)
        if not os.path.isdir(full_model_dir):
            continue
            
        model_name = model_dir.replace('mistake_analysis_', '')
        analysis_file = os.path.join(full_model_dir, f'{model_name}_error_analysis.json')
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
                
            for key, value in analysis.items():
                if key.startswith('category_'):
                    category_id = key.split('_')[1]
                    rmse = value.get('rmse')
                    if category_id not in summary_data:
                        summary_data[category_id] = {}
                    summary_data[category_id][model_name] = rmse
    
    if not summary_data:
        logging.warning("没有找到任何模型分析结果")
        return
    
    # 转换为 DataFrame
    summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
    summary_df.index.name = 'category_id'
    summary_df.reset_index(inplace=True)
    
    # 确保category_id是数值类型
    summary_df['category_id'] = pd.to_numeric(summary_df['category_id'])
    
    # 对每个分类找出最佳模型（RMSE最低的模型）
    best_models = []
    for _, row in summary_df.iterrows():
        model_rmses = {col: row[col] for col in summary_df.columns if col != 'category_id'}
        best_model = min(model_rmses.items(), key=lambda x: float(x[1]) if x[1] is not None else float('inf'))[0]
        best_models.append(best_model)
    
    summary_df['best_model'] = best_models
    
    # 按category_id排序
    summary_df.sort_values('category_id', inplace=True)
    
    # 保存汇总表
    summary_df.to_csv(summary_output_path, index=False)
    logging.info(f"汇总表已保存为 {summary_output_path}")
    
    return summary_df


def plot_rmse_summary(
    summary_df: pd.DataFrame,
    output_dir: str
):
    """
    绘制每个分类中各模型的 RMSE，标注最佳模型。

    Args:
        summary_df (pd.DataFrame): 汇总表数据框。
        output_dir (str): 图像保存目录。
    """
    plt.figure(figsize=(14, 10))
    
    # 获取模型列（排除category_id和best_model列）
    model_columns = [col for col in summary_df.columns if col not in ['category_id', 'best_model']]
    
    # 确保category_id是数值类型
    summary_df['category_id'] = pd.to_numeric(summary_df['category_id'])
    
    # 绘制每个模型的RMSE线
    for model in model_columns:
        # 确保RMSE值是数值类型
        rmse_values = pd.to_numeric(summary_df[model], errors='coerce')
        plt.plot(summary_df['category_id'], rmse_values, 
                marker='o', label=model, linewidth=2, markersize=8)
    
    # 标注最佳模型
    for idx, row in summary_df.iterrows():
        best_model = row['best_model']
        if best_model in model_columns:  # 确保best_model在数据中
            best_value = pd.to_numeric(row[best_model], errors='coerce')
            if pd.notnull(best_value):  # 只标注有效的值
                plt.scatter(row['category_id'], best_value, 
                          color='red', s=100, edgecolor='k', zorder=5)
                plt.text(row['category_id'], best_value, ' Best', 
                        color='red', fontsize=9, 
                        verticalalignment='bottom')
    
    plt.title('RMSE per Category per Model')
    plt.xlabel('Category ID')
    plt.ylabel('RMSE')
    
    # 设置x轴刻度为整数
    category_ids = sorted(summary_df['category_id'].unique())
    plt.xticks(category_ids, [str(int(x)) for x in category_ids])
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局以确保图例完全显示
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, 'rmse_summary_per_category.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"RMSE 汇总图已保存为 {plot_path}")


def plot_proportion_summary(
    summary_df: pd.DataFrame,
    output_dir: str
):
    """
    基于误差分析JSON结果绘制各个模型在不同误差类别的样本比例分布和累计比例，
    在横坐标显示误差阈值。

    Args:
        summary_df (pd.DataFrame): 汇总表数据框。
        output_dir (str): 图像保存目录。
    """
    plt.figure(figsize=(14, 10))
    
    # 定义每个类别的最大误差值
    error_thresholds = [2000, 3000, 5000, 8000, 10000, 12000, 14000, 16000, 20000, 99999]
    
    # 获取所有分析结果文件
    model_results = {}
    models_dir = output_dir
    
    for model_dir in os.listdir(models_dir):
        if not model_dir.startswith('mistake_analysis_'):
            continue
            
        model_name = model_dir.replace('mistake_analysis_', '')
        analysis_file = os.path.join(models_dir, model_dir, f'{model_name}_error_analysis.json')
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
                
            # 提取每个类别的proportion
            proportions = []
            categories = []
            for key, value in analysis.items():
                if key.startswith('category_'):
                    category_id = int(key.split('_')[1])
                    proportion = value.get('proportion', 0) * 100  # 转换为百分比
                    proportions.append(proportion)
                    categories.append(category_id)
            
            # 按category_id排序
            sorted_indices = np.argsort(categories)
            categories = np.array(categories)[sorted_indices]
            proportions = np.array(proportions)[sorted_indices]
            
            # 计算累计比例
            cumulative_proportions = np.cumsum(proportions)
            
            model_results[model_name] = {
                'categories': categories,
                'proportions': proportions,
                'cumulative_proportions': cumulative_proportions
            }
    
    # 为累计线和普通线使用不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    # 绘制每个模型的proportion线和累计线
    for (model_name, data), color in zip(model_results.items(), colors):
        # 绘制普通比例线
        plt.plot(data['categories'], data['proportions'],
                marker='o', label=f'{model_name} (Individual)',
                linewidth=2, markersize=8, color=color)
        
        # 绘制累计比例线
        plt.plot(data['categories'], data['cumulative_proportions'],
                marker='s', label=f'{model_name} (Cumulative)',
                linewidth=2, markersize=8, color=color,
                linestyle='--', alpha=0.7)
        
        # 为每个点添加标签
        for x, y in zip(data['categories'], data['proportions']):
            plt.text(x, y, f'{y:.1f}%',
                    ha='center', va='bottom', color=color)
        
        # 为累计线添加标签
        for x, y in zip(data['categories'], data['cumulative_proportions']):
            plt.text(x, y, f'{y:.1f}%',
                    ha='center', va='top', color=color)
    
    plt.title('Sample Proportion per Error Category (Individual and Cumulative)')
    plt.xlabel('Error Category (Max Error in $)')
    plt.ylabel('Proportion of Samples (%)')
    
    # 设置x轴刻度和标签
    x_ticks = sorted(set(sum([list(data['categories']) for data in model_results.values()], [])))
    plt.xticks(x_ticks, [f'{x}\n(${error_thresholds[x]:,})' for x in x_ticks])
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              ncol=1, borderaxespad=0.)
    
    # 调整y轴范围，留出空间显示标签
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, max(105, ymax * 1.1))  # 确保y轴最大值至少到100%
    
    # 添加100%的参考线
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    
    # 调整布局以确保图例完全显示
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, 'proportion_summary_per_category.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"比例汇总图已保存为 {plot_path}")
    
def analyze_category(
    data: pd.DataFrame,
    category_id: int,
    heatmap_dir: str,
    model_name: str
) -> Dict[str, Any]:
    """
    分析单个误差分类，找出与误差关系最大的特征，并生成热力图。

    Args:
        data (pd.DataFrame): 包含误差和特征的数据框。
        category_id (int): 当前分类的ID。
        heatmap_dir (str): 热力图保存目录。
        model_name (str): 模型名称，用于命名文件。

    Returns:
        Dict[str, Any]: 分析结果，包括特征重要性、RMSE、统计信息和热力图路径。
    """
    category_data = data[data['mistake_category'] == category_id]
    logging.info(f"开始分析分类 {category_id}，样本数量: {len(category_data)}")
    
    # 计算每个特征与误差的相关性
    features = category_data.drop(columns=['price', 'inference_price', 'mistake', 'mistake_category', 'mistake_abs'])
    correlation = features.corrwith(category_data['mistake']).abs()
    top_features = correlation.sort_values(ascending=False).head(5).index.tolist()
    
    logging.info(f"分类 {category_id} 与误差关系最强的特征: {top_features}")
    
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(category_data['price'], category_data['inference_price']))
    
    # 计算统计信息
    count = len(category_data)
    proportion = count / len(data)
    mean_mistake = category_data['mistake'].abs().mean()
    max_mistake = category_data['mistake'].abs().max()
    min_mistake = category_data['mistake'].abs().min()
    
    logging.info(f"分类 {category_id} 的 RMSE: {rmse:.4f}, Count: {count}, Proportion: {proportion:.2%}, "
                 f"Mean Absolute Mistake: {mean_mistake:.4f}, Max Mistake: {max_mistake:.4f}, "
                 f"Min Mistake: {min_mistake:.4f}")
    
    # 生成热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(category_data[top_features + ['mistake']].corr(), annot=True, cmap='coolwarm')
    plt.title(f'Category {category_id} Feature vs Mistake Correlation Heatmap')
    
    heatmap_path = os.path.join(heatmap_dir, f'category_{category_id}_heatmap_{model_name}.png')
    os.makedirs(heatmap_dir, exist_ok=True)
    plt.savefig(heatmap_path)
    plt.close()
    
    logging.info(f"生成分类 {category_id} 的热力图: {heatmap_path}")
    
    return {
        'category_id': int(category_id),
        'rmse': rmse,
        'count': int(count),
        'proportion': proportion,
        'mean_absolute_mistake': mean_mistake,
        'max_mistake': max_mistake,
        'min_mistake': min_mistake,
        'top_features': top_features,
        'heatmap_path': heatmap_path
    }

def analyze_category_features(
    data: pd.DataFrame,
    model_name: str,
    heatmap_dir: str
) -> Dict[str, Any]:
    """
    对所有误差分类进行分析，并生成相应的热力图。

    Args:
        data (pd.DataFrame): 包含误差和分类标签的数据框。
        model_name (str): 模型名称，用于命名文件。
        heatmap_dir (str): 热力图保存目录。

    Returns:
        Dict[str, Any]: 包含所有分类分析结果和整体热力图路径的字典。
    """
    analysis_results = {}
    for category_id in sorted(data['mistake_category'].unique()):
        analysis = analyze_category(data, category_id, heatmap_dir, model_name)
        analysis_results[f'category_{category_id}'] = analysis
    
    # 生成整体热力图
    overall_heatmap_path = generate_overall_heatmap(data, heatmap_dir, model_name)
    analysis_results['overall'] = {
        'heatmap_path': overall_heatmap_path
    }
    
    return analysis_results

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
        cluster_range (Tuple[int, int]): 聚类数范围（不再使用）。
        random_state (int): 随机种子。
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 检查 'price' 列是否存在
    if 'price' not in inference_results.columns:
        logging.error("'price' 列缺失，无法计算 'mistake'")
        return
    
    # 计算误差的绝对值
    inference_results['mistake_abs'] = inference_results['mistake'].abs()
    
    # 定义误差分类区间
    categories = define_mistake_categories()
    inference_results['mistake_category'] = inference_results['mistake'].apply(lambda x: assign_mistake_category(x, categories))
    
    # 保存分类结果
    categorized_file = os.path.join(output_dir, f'{model_type}_categorized_results.csv')
    inference_results.to_csv(categorized_file, index=False)
    logging.info(f"分类结果已保存为 {categorized_file}")
    
    # 生成误差分布图
    distribution_plot_path = plot_mistake_distribution_with_categories(inference_results, categories, output_dir, model_type)
    
    # 进行误差分析
    analysis_results = analyze_category_features(inference_results, model_type, heatmap_dir)
    
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
        cluster_range (Tuple[int, int]): 聚类数范围（不再使用）。
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
