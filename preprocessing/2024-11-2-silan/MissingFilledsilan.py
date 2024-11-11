import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Union, Dict, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression

def impute_with_auto_sklearn_auto_features(
    data: pd.DataFrame,
    target_columns: Union[List[str], str],
    possible_estimation_features: Optional[List[str]] = None,  # 新增参数，限定估算特征的选择范围
    feature_selection_method: str = 'mutual_info',  # 'mutual_info', 'select_kbest', 'random_forest'
    k_feature_ratio: float = 0.2,  # 新增参数，特征选择的比例，默认 20%
    include_features: Optional[List[str]] = None,  # 可选，指定包含哪些特征
    exclude_features: Optional[List[str]] = None,  # 可选，指定排除哪些特征
    time_left_for_this_task: int = 120,  # 每个目标列的总时间（秒）
    per_run_time_limit: int = 30,  # 每个模型的训练时间限制（秒）
    n_jobs: int = -1,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    使用 auto-sklearn 和自动化特征选择对缺失值进行插补，无需手动传递估算特征。
    
    参数:
    - data: pandas.DataFrame, 包含数据的DataFrame。
    - target_columns: list 或 str, 需要插补的目标列。
    - possible_estimation_features: list 或 None, 可选，限定估算特征的选择范围。如果为 None，默认使用所有非目标列。
    - feature_selection_method: str, 特征选择方法，可选 'mutual_info', 'select_kbest', 'random_forest'。
    - k_feature_ratio: float, 特征选择的比例，介于 0 和 1 之间，默认 0.2（即 20%）。
    - include_features: list 或 None, 可选，指定包含哪些特征作为估算特征的一部分。
    - exclude_features: list 或 None, 可选，指定排除哪些特征。
    - time_left_for_this_task: int, auto-sklearn 搜索时间总量（秒）。
    - per_run_time_limit: int, 每个模型训练的时间限制（秒）。
    - n_jobs: int, 并行处理的作业数，-1表示使用所有CPU核心。
    - random_state: int, 随机种子，保证结果可重复。
    
    返回:
    - data: pandas.DataFrame, 插补后的DataFrame。
    - imputation_stats: dict, 插补统计信息。
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 确保 target_columns 是列表
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # 数据验证
    available_target_columns = [col for col in target_columns if col in data.columns]
    if not available_target_columns:
        raise ValueError(f"目标列不存在于数据中: {target_columns}")

    # 自动选择估算特征
    all_features = data.columns.tolist()
    
    if possible_estimation_features is not None:
        # 如果用户提供了限定的估算特征列表，首先确保这些特征存在于数据中
        missing_features = [feat for feat in possible_estimation_features if feat not in all_features]
        if missing_features:
            raise ValueError(f"指定的估算特征不存在于数据中: {missing_features}")
        estimation_features = possible_estimation_features.copy()
    else:
        # 默认使用所有非目标列作为估算特征
        estimation_features = [col for col in all_features if col not in available_target_columns]

    # 可选的包含或排除特征
    if include_features:
        # 确保包含的特征存在于数据中
        missing_include = [feat for feat in include_features if feat not in all_features]
        if missing_include:
            raise ValueError(f"包含的特征不存在于数据中: {missing_include}")
        estimation_features = list(set(estimation_features + include_features))
    if exclude_features:
        # 排除指定的特征
        estimation_features = [col for col in estimation_features if col not in exclude_features]

    if not estimation_features:
        raise ValueError("未找到任何估算特征。请检查数据或提供包含/排除特征。")

    logger.info(f"自动选择的估算特征: {estimation_features}")

    # 处理分类变量的缺失值并进行独热编码
    categorical_features = data[estimation_features].select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        logger.info(f"处理分类特征: {categorical_features}")
        # 填充分类特征中的缺失值
        data[categorical_features] = data[categorical_features].fillna('Missing')

        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # 确保使用正确参数
        encoded_features = encoder.fit_transform(data[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
        data = pd.concat([data, encoded_df], axis=1)
        # 更新估算特征列表，移除原始分类特征，添加编码后的特征
        estimation_features = [feat for feat in estimation_features if feat not in categorical_features] + list(encoded_feature_names)

    # 处理数值特征中的缺失值（如果有）
    numerical_features = data[estimation_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numerical_features:
        logger.info(f"处理数值特征: {numerical_features}")
        # 使用中位数填充数值特征中的缺失值
        data[numerical_features] = data[numerical_features].fillna(data[numerical_features].median())

    # 确保所有估算特征中不再有缺失值
    remaining_missing = data[estimation_features].isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"在估算特征中仍存在 {remaining_missing} 个缺失值。将这些缺失值填充为 'Missing' 或中位数。")
        # 对于分类特征，再次填充缺失值
        for col in categorical_features:
            data[col] = data[col].fillna('Missing')
        # 对于数值特征，再次填充缺失值
        for col in numerical_features:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())

    # 初始化统计信息
    imputation_stats = {col: {
        'initial_missing': data[col].isnull().sum(),
        'final_missing': 0,
        'filled_values': 0,
        'errors': []
    } for col in available_target_columns}

    def select_features(X: pd.DataFrame, y: pd.Series) -> List[str]:
        """根据指定的方法选择特征，并自动确定 k_features。"""
        if feature_selection_method == 'mutual_info':
            mi_scores = mutual_info_regression(X, y, random_state=random_state)
            feature_scores = pd.Series(mi_scores, index=X.columns)
            # 自动确定 k_features，取总特征数的 k_feature_ratio，至少1个
            k_features_auto = max(int(k_feature_ratio * len(X.columns)), 1)
            selected = feature_scores.sort_values(ascending=False).head(k_features_auto).index.tolist()
        elif feature_selection_method == 'select_kbest':
            from sklearn.feature_selection import SelectKBest, f_regression
            # 自动确定 k_features，取总特征数的 k_feature_ratio，至少1个
            k_features_auto = max(int(k_feature_ratio * len(X.columns)), 1)
            selector = SelectKBest(score_func=f_regression, k=k_features_auto)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        elif feature_selection_method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            # 自动确定 k_features，取总特征数的 k_feature_ratio，至少1个
            k_features_auto = max(int(k_feature_ratio * len(X.columns)), 1)
            selected = importances.sort_values(ascending=False).head(k_features_auto).index.tolist()
        else:
            raise ValueError(f"未知的特征选择方法: {feature_selection_method}")
        logger.info(f"选择的特征: {selected}")
        return selected

    def process_column(
        target_column: str
    ) -> Tuple[str, pd.Series, Dict]:
        stats = {
            'initial_missing': 0,
            'final_missing': 0,
            'filled_values': 0,
            'errors': []
        }
        try:
            # 检查是否为数值型
            if not pd.api.types.is_numeric_dtype(data[target_column]):
                msg = f"目标列 '{target_column}' 不是数值型。跳过该列。"
                logger.warning(msg)
                stats['errors'].append(msg)
                return target_column, data[target_column], stats

            # 分离缺失和非缺失数据
            not_null_data = data[data[target_column].notnull()]
            null_data = data[data[target_column].isnull()]

            stats['initial_missing'] = null_data.shape[0]

            if not null_data.empty:
                X = not_null_data[estimation_features]
                y = not_null_data[target_column]
                X_pred = null_data[estimation_features]

                # 特征选择
                selected_features = select_features(X, y)
                if not selected_features:
                    msg = f"目标列 '{target_column}' 没有选择到任何特征。使用全局中位数填补。"
                    logger.warning(msg)
                    stats['errors'].append(msg)
                    median_value = data[target_column].median()
                    data[target_column].fillna(median_value, inplace=True)
                    stats['final_missing'] = data[target_column].isnull().sum()
                    stats['filled_values'] = stats['initial_missing'] - stats['final_missing']
                    return target_column, data[target_column], stats

                X_selected = X[selected_features]
                X_pred_selected = X_pred[selected_features]

                # 训练 auto-sklearn 回归模型
                automl = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=time_left_for_this_task,
                    per_run_time_limit=per_run_time_limit,
                    n_jobs=1,  # auto-sklearn 内部已使用多线程
                    seed=random_state,
                    resampling_strategy='holdout',
                    resampling_strategy_arguments={'train_size': 0.75}
                )
                automl.fit(X_selected, y)

                # 预测缺失值
                y_pred = automl.predict(X_pred_selected)

                # 将预测值插补到原数据中
                data.loc[data[target_column].isnull(), target_column] = y_pred

                stats['filled_values'] = len(y_pred)

            stats['final_missing'] = data[target_column].isnull().sum()

            return target_column, data[target_column], stats

        except Exception as e:
            error_msg = f"处理列 '{target_column}' 时发生错误: {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            return target_column, data[target_column], stats

    # 并行处理每个目标列
    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        future_to_column = {
            executor.submit(process_column, target_column): target_column for target_column in available_target_columns
        }

        for future in as_completed(future_to_column):
            target_column = future_to_column[future]
            try:
                col, imputed_series, stats = future.result()
                data[col] = imputed_series
                imputation_stats[col].update(stats)
                logger.info(f"完成插补目标列: {col}")
                logger.info(f"初始缺失值: {stats['initial_missing']}, 填补值: {stats['filled_values']}, 处理后缺失值: {stats['final_missing']}")
            except Exception as e:
                error_msg = f"处理列 '{target_column}' 时发生错误: {str(e)}"
                logger.error(error_msg)
                imputation_stats[target_column]['errors'].append(error_msg)

    logger.info("缺失值插补完成")
    return data, imputation_stats

if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'make': np.random.choice(['Toyota', 'Ford', 'Honda', None], size=100),
        'model': np.random.choice(['ModelA', 'ModelB', 'ModelC', None], size=100),
        'type_of_vehicle': np.random.choice(['SUV', 'Sedan', 'Truck', None], size=100),
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric', None], size=100),
        'opc_scheme': np.random.choice(['Scheme1', 'Scheme2', None], size=100),
        'lifespan': np.random.choice(['Short', 'Medium', 'Long', None], size=100),
        'eco_category': np.random.choice(['Eco1', 'Eco2', 'Eco3', None], size=100),
        'features': np.random.choice(['Feature1', 'Feature2', None], size=100),
        'accessories': np.random.choice(['Accessory1', 'Accessory2', None], size=100),
        'power': np.random.randn(100)
    })

    # 引入缺失值
    df.loc[df.sample(frac=0.1, random_state=42).index, 'power'] = np.nan

    print("原始数据示例:")
    print(df.head())

    # 定义目标列
    targets = ['power']

    # 定义可能的估算特征（可选）
    possible_estimators = None  # 或者指定特定的特征列表，例如 ['make', 'model', ...]

    # 插补缺失值
    imputed_df, stats = impute_with_auto_sklearn_auto_features(
        data=df,
        target_columns=targets,
        possible_estimation_features=possible_estimators,  # 新增参数
        feature_selection_method='mutual_info',  # 可选 'mutual_info', 'select_kbest', 'random_forest'
        k_feature_ratio=0.2,  # 选择特征的比例，默认为20%
        include_features=None,  # 可选，指定包含哪些特征
        exclude_features=None,  # 可选，指定排除哪些特征
        time_left_for_this_task=60,  # 为每个目标列分配60秒
        per_run_time_limit=15,
        n_jobs=2,  # 并行处理两个目标列
        random_state=42
    )

    print("\n插补后的数据示例:")
    print(imputed_df.head())

    print("\n插补统计信息:")
    for col, stat in stats.items():
        print(f"\n{col}:")
        for key, value in stat.items():
            print(f"  {key}: {value}")
