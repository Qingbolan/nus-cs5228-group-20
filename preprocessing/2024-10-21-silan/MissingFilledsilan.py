import pandas as pd
import numpy as np
# import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
# from sklearn.linear_model import LinearRegression


def impute_missing_values_1(data, target_columns, estimation_features, n_neighbors=5, z_threshold=3, min_features=3):
    # 确保 target_columns 是列表
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # 检查哪些目标列实际存在于数据中
    available_target_columns = [col for col in target_columns if col in data.columns]
    if not available_target_columns:
        raise ValueError(f"None of the specified target columns {target_columns} are present in the data. Available columns are: {data.columns.tolist()}")
    
    print(f"Available target columns: {available_target_columns}")
    print(f"Missing target columns: {set(target_columns) - set(available_target_columns)}")

    # 检查哪些估算特征实际存在于数据中
    available_estimation_features = [col for col in estimation_features if col in data.columns]
    if not available_estimation_features:
        raise ValueError(f"None of the specified estimation features are present in the data. Available columns are: {data.columns.tolist()}")
    
    print(f"Available estimation features: {available_estimation_features}")
    print(f"Missing estimation features: {set(estimation_features) - set(available_estimation_features)}")

    imputation_stats = {col: {
        'initial_missing': 0,
        'final_missing': 0,
        'filled_values': 0,
        'outliers_detected': 0,
        'outliers_replaced': 0,
        'small_values_adjusted': 0
    } for col in available_target_columns}

    try:
        data = data.copy()

        # 对分类变量进行独热编码
        categorical_features = data[available_estimation_features].select_dtypes(include=['object', 'category']).columns
        if not categorical_features.empty:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            available_estimation_features = list(set(available_estimation_features) - set(categorical_features)) + list(encoded_feature_names)

        for target_column in available_target_columns:
            print(f"\nProcessing {target_column}")
            imputation_stats[target_column]['initial_missing'] = data[target_column].isnull().sum()
            print(f"Initial missing {target_column} values: {imputation_stats[target_column]['initial_missing']}")

            if not pd.api.types.is_numeric_dtype(data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Skipping.")
                continue

            # 确定目标列的合理最小值
            valid_values = data[target_column].dropna()
            min_valid_value = valid_values.min()
            reasonable_min = max(min_valid_value * 0.5, 1)  # 使用有效值的一半或1，取较大者
            print(f"Determined reasonable minimum value for {target_column}: {reasonable_min}")

            # 特征选择
            numeric_features = data[available_estimation_features].select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_features) > 0:
                # 去除包含 NaN 的行用于计算互信息
                valid_data = data.dropna(subset=[target_column] + list(numeric_features))
                if len(valid_data) == 0:
                    print(f"Warning: No valid data for mutual information calculation for {target_column}. Using all numeric features.")
                    selected_features = list(numeric_features)
                else:
                    mi_scores = mutual_info_regression(valid_data[numeric_features], valid_data[target_column])
                    selected_features = [feature for feature, score in zip(numeric_features, mi_scores) if score > 0]
                selected_features = selected_features[:min_features] if len(selected_features) > min_features else selected_features
            else:
                selected_features = []
            
            selected_features.extend([f for f in available_estimation_features if f not in selected_features])
            print(f"Selected features for estimation: {selected_features}")

            # KNN估算
            impute_data = data[selected_features + [target_column]].copy()
            non_missing_count = impute_data[target_column].notna().sum()
            if non_missing_count > n_neighbors:
                imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                imputed_values = imputer.fit_transform(impute_data)
                
                mask = data[target_column].isnull()
                estimated_values = imputed_values[mask, -1]
                small_estimates = estimated_values < reasonable_min

                if small_estimates.any():
                    print(f"Warning: {small_estimates.sum()} values were estimated below {reasonable_min}. Adjusting these values.")
                    estimated_values[small_estimates] = reasonable_min
                    imputation_stats[target_column]['small_values_adjusted'] = small_estimates.sum()

                data.loc[mask, target_column] = estimated_values
            else:
                print(f"Not enough non-missing values for KNN imputation. Using median imputation.")
                data[target_column].fillna(max(data[target_column].median(), reasonable_min), inplace=True)

            # 处理可能的无效值
            # data[target_column] = data[target_column].replace([np.inf, -np.inf], np.nan)
            # if data[target_column].isnull().sum() > 0:
            #     print(f"Warning: {data[target_column].isnull().sum()} invalid values detected after imputation. Filling with median.")
            #     data[target_column].fillna(max(data[target_column].median(), reasonable_min), inplace=True)

            # # 异常值检测和处理
            # z_scores = np.abs(scipy_stats.zscore(data[target_column]))
            # outliers = z_scores > z_threshold
            # imputation_stats[target_column]['outliers_detected'] = outliers.sum()
            # print(f"Detected {imputation_stats[target_column]['outliers_detected']} outliers in {target_column}")

            if imputation_stats[target_column]['outliers_detected'] > 0:
                lower_bound = data[target_column].quantile(0.25)
                upper_bound = data[target_column].quantile(0.75)
                iqr = upper_bound - lower_bound
                smart_lower = max(lower_bound - 1.5 * iqr, reasonable_min)
                smart_upper = upper_bound + 1.5 * iqr
                
                # data.loc[outliers & (data[target_column] < smart_lower), target_column] = smart_lower
                # data.loc[outliers & (data[target_column] > smart_upper), target_column] = smart_upper
                
                imputation_stats[target_column]['outliers_replaced'] = imputation_stats[target_column]['outliers_detected']
                print(f"Replaced outliers with values between {smart_lower} and {smart_upper}")

            # 最终检查，确保没有不合理的小值
            small_values = data[target_column] < reasonable_min
            if small_values.any():
                print(f"Warning: {small_values.sum()} values are still below {reasonable_min}. Adjusting these values.")
                data.loc[small_values, target_column] = reasonable_min
                imputation_stats[target_column]['small_values_adjusted'] += small_values.sum()

            imputation_stats[target_column]['final_missing'] = data[target_column].isnull().sum()
            imputation_stats[target_column]['filled_values'] = imputation_stats[target_column]['initial_missing'] - imputation_stats[target_column]['final_missing']
            print(f"After processing, missing {target_column} values: {imputation_stats[target_column]['final_missing']}")
            print(f"Filled missing values: {imputation_stats[target_column]['filled_values']}")
            print(f"Small values adjusted: {imputation_stats[target_column]['small_values_adjusted']}")

            print(f"\n{target_column} statistics:")
            print(data[target_column].describe())

        return data, imputation_stats
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        for col in available_target_columns:
            imputation_stats[col]['error'] = str(e)
        return data, imputation_stats
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import LocalOutlierFactor


def impute_missing_values_optimized(data, target_columns, estimation_features, n_neighbors=5, min_features=3):
    """
    优化后的缺失值插补函数，支持多种填补策略、异常值处理和特征选择。

    参数:
    - data: pandas.DataFrame, 包含数据的DataFrame。
    - target_columns: list 或 str, 需要插补的目标列。
    - estimation_features: list, 用于估算缺失值的特征列。
    - n_neighbors: int, KNN插补的邻居数量。
    - min_features: int, 特征选择时最小特征数量。

    返回:
    - data: pandas.DataFrame, 插补后的DataFrame。
    - imputation_stats: dict, 插补统计信息。
    """

    # 确保 target_columns 是列表
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # 检查目标列是否存在
    available_target_columns = [col for col in target_columns if col in data.columns]
    if not available_target_columns:
        raise ValueError(f"指定的目标列 {target_columns} 都不存在于数据中。可用列: {data.columns.tolist()}")

    print(f"可用目标列: {available_target_columns}")
    print(f"缺失的目标列: {set(target_columns) - set(available_target_columns)}")

    # 检查估算特征是否存在
    available_estimation_features = [col for col in estimation_features if col in data.columns]
    if not available_estimation_features:
        raise ValueError(f"指定的估算特征 {estimation_features} 都不存在于数据中。可用列: {data.columns.tolist()}")

    print(f"可用估算特征: {available_estimation_features}")
    print(f"缺失的估算特征: {set(estimation_features) - set(available_estimation_features)}")

    # 初始化插补统计信息
    imputation_stats = {col: {
        'initial_missing': data[col].isnull().sum(),
        'final_missing': 0,
        'filled_values': 0,
        'outliers_detected': 0,
        'outliers_replaced': 0,
        'small_values_adjusted': 0
    } for col in available_target_columns}

    try:
        data = data.copy()

        # 处理分类变量的独热编码
        categorical_features = data[available_estimation_features].select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            available_estimation_features = [feat for feat in available_estimation_features if
                                             feat not in categorical_features] + list(encoded_feature_names)

        # 遍历每个目标列进行插补
        for target_column in available_target_columns:
            print(f"\n正在处理目标列: {target_column}")
            imputation_stats[target_column]['initial_missing'] = data[target_column].isnull().sum()
            print(f"初始缺失值数量: {imputation_stats[target_column]['initial_missing']}")

            if not pd.api.types.is_numeric_dtype(data[target_column]):
                print(f"警告: 目标列 '{target_column}' 不是数值型。跳过该列。")
                continue

            # 确定合理的最小值
            valid_values = data[target_column].dropna()
            min_valid_value = valid_values.min()
            reasonable_min = max(min_valid_value * 0.5, 1)  # 使用有效值的一半或1，取较大者
            print(f"为 {target_column} 确定的合理最小值: {reasonable_min}")

            # 特征选择
            numeric_features = data[available_estimation_features].select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_features) > 0:
                # 使用SelectKBest进行特征选择，确保至少有 min_features 个特征
                selector = SelectKBest(score_func=f_regression, k=min(min_features, len(numeric_features)))
                valid_data = data.dropna(subset=[target_column] + list(numeric_features))
                if len(valid_data) > 0:
                    selector.fit(valid_data[numeric_features], valid_data[target_column])
                    selected_features = [feature for feature, selected in zip(numeric_features, selector.get_support())
                                         if selected]
                else:
                    selected_features = numeric_features.tolist()

                # 保证至少有 min_features 个特征
                if len(selected_features) < min_features:
                    additional_features = [f for f in numeric_features if f not in selected_features]
                    selected_features += additional_features[:min_features - len(selected_features)]

                print(f"选择的特征: {selected_features}")
            else:
                selected_features = []

            # 使用选定特征进行KNN插补
            impute_data = data[selected_features + [target_column]].copy()
            non_missing_count = impute_data[target_column].notna().sum()
            if non_missing_count > n_neighbors:
                imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                imputed_values = imputer.fit_transform(impute_data)

                mask = data[target_column].isnull()
                estimated_values = imputed_values[mask, -1]

                # 检测并调整小于合理最小值的估计值
                small_estimates = estimated_values < reasonable_min
                if small_estimates.any():
                    print(f"警告: {small_estimates.sum()} 个估计值低于 {reasonable_min}，将进行调整。")
                    estimated_values[small_estimates] = reasonable_min
                    imputation_stats[target_column]['small_values_adjusted'] += small_estimates.sum()

                data.loc[mask, target_column] = estimated_values
            else:
                print(f"缺少足够的非缺失值进行KNN插补。使用中位数填补。")
                median_value = max(data[target_column].median(), reasonable_min)
                data[target_column].fillna(median_value, inplace=True)

            # 使用IQR方法检测和处理异常值
            Q1 = data[target_column].quantile(0.25)
            Q3 = data[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(Q1 - 1.5 * IQR, reasonable_min)
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[target_column] < lower_bound) | (data[target_column] > upper_bound)
            imputation_stats[target_column]['outliers_detected'] = outliers.sum()
            data.loc[outliers, target_column] = np.nan  # 将异常值设为NaN以便后续插补

            # 再次进行插补以处理异常值
            if outliers.sum() > 0:
                if non_missing_count > n_neighbors:
                    imputed_values = imputer.fit_transform(impute_data)
                    mask = outliers
                    estimated_values = imputed_values[mask, -1]
                    # 调整小值
                    small_estimates = estimated_values < reasonable_min
                    if small_estimates.any():
                        print(f"警告: {small_estimates.sum()} 个异常估计值低于 {reasonable_min}，将进行调整。")
                        estimated_values[small_estimates] = reasonable_min
                        imputation_stats[target_column]['small_values_adjusted'] += small_estimates.sum()
                    data.loc[mask, target_column] = estimated_values
                else:
                    print(f"缺少足够的非缺失值进行异常值插补。使用中位数填补。")
                    median_value = max(data[target_column].median(), reasonable_min)
                    data[target_column].fillna(median_value, inplace=True)

                imputation_stats[target_column]['outliers_replaced'] = outliers.sum()

            # 最终缺失值统计
            imputation_stats[target_column]['final_missing'] = data[target_column].isnull().sum()
            imputation_stats[target_column]['filled_values'] = imputation_stats[target_column]['initial_missing'] - \
                                                               imputation_stats[target_column]['final_missing']
            print(f"处理后缺失值数量: {imputation_stats[target_column]['final_missing']}")
            print(f"填补的缺失值数量: {imputation_stats[target_column]['filled_values']}")
            print(f"调整的小值数量: {imputation_stats[target_column]['small_values_adjusted']}")

            # 显示统计信息
            print(f"\n{target_column} 的统计信息:")
            print(data[target_column].describe())

        return data, imputation_stats

    except Exception as e:
        print(f"发生错误: {str(e)}")
        for col in available_target_columns:
            if 'error' not in imputation_stats[col]:
                imputation_stats[col]['error'] = str(e)
        return data, imputation_stats

    
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Union, Dict, Tuple, Optional

def process_single_column(
    target_column: str,
    data: pd.DataFrame,
    available_estimation_features: List[str],
    use_rf_importance: bool,
    n_neighbors: int,
    min_features: int,
    random_state: Optional[int] = 42
) -> Tuple[str, pd.Series, Dict]:
    logger = logging.getLogger(__name__)
    stats = {
        'initial_missing': 0,
        'final_missing': 0,
        'filled_values': 0,
        'errors': []
    }

    try:
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            msg = f"目标列 '{target_column}' 不是数值型。跳过该列。"
            logger.warning(msg)
            stats['errors'].append(msg)
            return target_column, data[target_column], stats

        stats['initial_missing'] = data[target_column].isnull().sum()

        # 特征选择
        numeric_features = data[available_estimation_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = []

        if numeric_features:
            if use_rf_importance:
                valid_data = data.dropna(subset=[target_column] + numeric_features)
                if not valid_data.empty:
                    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
                    rf.fit(valid_data[numeric_features], valid_data[target_column])
                    importance = pd.Series(rf.feature_importances_, index=numeric_features)
                    selected_features = importance.nlargest(min_features).index.tolist()
                    logger.info(f"目标列 '{target_column}' 使用随机森林选择的特征: {selected_features}")
            if not selected_features:
                selector = SelectKBest(score_func=f_regression, k=min(min_features, len(numeric_features)))
                valid_data = data.dropna(subset=[target_column] + numeric_features)
                if not valid_data.empty:
                    selector.fit(valid_data[numeric_features], valid_data[target_column])
                    selected_features = [feature for feature, selected in 
                                         zip(numeric_features, selector.get_support()) if selected]
                    logger.info(f"目标列 '{target_column}' 使用SelectKBest选择的特征: {selected_features}")

            # 确保至少有 min_features 个特征
            if len(selected_features) < min_features and numeric_features:
                additional_features = [f for f in numeric_features if f not in selected_features]
                needed = min_features - len(selected_features)
                selected_features += additional_features[:needed]
                logger.info(f"目标列 '{target_column}' 补充选择的特征: {selected_features[:needed]}")

        if not selected_features:
            msg = f"目标列 '{target_column}' 没有足够的数值特征用于插补。使用全局中位数填补。"
            logger.warning(msg)
            stats['errors'].append(msg)
            median_value = data[target_column].median()
            data[target_column].fillna(median_value, inplace=True)
            stats['final_missing'] = data[target_column].isnull().sum()
            stats['filled_values'] = stats['initial_missing'] - stats['final_missing']
            return target_column, data[target_column], stats

        # 使用选定特征进行KNN插补
        impute_features = selected_features + [target_column]
        impute_data = data[impute_features].copy()

        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_array = imputer.fit_transform(impute_data)
        imputed_df = pd.DataFrame(imputed_array, columns=impute_features, index=data.index)

        # 更新缺失值
        mask = data[target_column].isnull()
        filled_values = imputed_df.loc[mask, target_column]
        data.loc[mask, target_column] = filled_values
        stats['filled_values'] = filled_values.notna().sum()

        # 最终缺失值统计
        stats['final_missing'] = data[target_column].isnull().sum()

        return target_column, data[target_column], stats

    except Exception as e:
        error_msg = f"处理列 '{target_column}' 时发生错误: {str(e)}"
        logger.error(error_msg)
        stats['errors'].append(error_msg)
        return target_column, data[target_column], stats

def impute_missing_values_optimized_2(
    data: pd.DataFrame,
    target_columns: Union[List[str], str],
    estimation_features: List[str],
    n_neighbors: int = 5,
    min_features: int = 3,
    n_jobs: int = -1,
    use_rf_importance: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    优化后的缺失值插补函数，去除了特征缩放和异常值检测，提升性能和可靠性。

    参数:
    - data: pandas.DataFrame, 包含数据的DataFrame。
    - target_columns: list 或 str, 需要插补的目标列。
    - estimation_features: list, 用于估算缺失值的特征列。
    - n_neighbors: int, KNN插补的邻居数量。
    - min_features: int, 特征选择时最小特征数量。
    - n_jobs: int, 并行处理的作业数，-1表示使用所有CPU核心。
    - use_rf_importance: bool, 是否使用随机森林进行特征重要性评估。
    - random_state: int, 随机种子，保证结果可重复。

    返回:
    - data: pandas.DataFrame, 插补后的DataFrame。
    - imputation_stats: dict, 插补统计信息。
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 主处理流程
    try:
        data = data.copy()
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        # 数据验证
        available_target_columns = [col for col in target_columns if col in data.columns]
        if not available_target_columns:
            raise ValueError(f"目标列不存在于数据中: {target_columns}")

        available_estimation_features = [col for col in estimation_features if col in data.columns]
        if not available_estimation_features:
            raise ValueError(f"估算特征不存在于数据中: {estimation_features}")

        # 处理分类变量的独热编码
        categorical_features = data[available_estimation_features].select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            logger.info(f"处理分类特征: {categorical_features}")
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data, encoded_df], axis=1)
            available_estimation_features = [feat for feat in available_estimation_features 
                                             if feat not in categorical_features] + list(encoded_feature_names)

        # 初始化统计信息
        imputation_stats = {col: {
            'initial_missing': data[col].isnull().sum(),
            'final_missing': 0,
            'filled_values': 0,
            'errors': []
        } for col in available_target_columns}

        # 并行处理每个目标列
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            future_to_column = {
                executor.submit(
                    process_single_column,
                    target_column,
                    data,
                    available_estimation_features,
                    use_rf_importance,
                    n_neighbors,
                    min_features,
                    random_state
                ): target_column for target_column in available_target_columns
            }

            for future in as_completed(future_to_column):
                target_column = future_to_column[future]
                try:
                    col, imputed_series, stats = future.result()
                    data[col] = imputed_series
                    imputation_stats[col].update(stats)
                except Exception as e:
                    error_msg = f"处理列 '{target_column}' 时发生错误: {str(e)}"
                    logger.error(error_msg)
                    imputation_stats[target_column]['errors'].append(error_msg)

        # 更新最终统计信息
        for col in available_target_columns:
            if 'final_missing' not in imputation_stats[col]:
                imputation_stats[col]['final_missing'] = data[col].isnull().sum()
                imputation_stats[col]['filled_values'] = (
                    imputation_stats[col]['initial_missing'] - 
                    imputation_stats[col]['final_missing']
                )

        logger.info("缺失值插补完成")
        return data, imputation_stats

    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        return data, {'error': str(e)}


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Required import
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import logging
from typing import List, Union, Dict, Tuple, Optional

def process_cluster(
    cluster_data: pd.DataFrame,
    target_columns: List[str],
    selected_features: List[str],
    imputation_method: str,
    n_neighbors: int,
    random_state: Optional[int]
) -> Tuple[pd.DataFrame, Dict]:
    """Process a single cluster's data for imputation."""
    stats = {}
    try:
        for target in target_columns:
            missing_before = cluster_data[target].isnull().sum()
            if missing_before == 0:
                continue

            features = selected_features + [target]
            impute_subset = cluster_data[features].copy()

            if imputation_method == 'knn':
                # Adjust n_neighbors if it's larger than the available samples
                adjusted_n_neighbors = min(n_neighbors, len(impute_subset) - 1)
                if adjusted_n_neighbors < 1:
                    adjusted_n_neighbors = 1
                imputer = KNNImputer(n_neighbors=adjusted_n_neighbors, weights='distance')
            elif imputation_method == 'iterative':
                imputer = IterativeImputer(random_state=random_state)
            else:
                raise ValueError(f"Unknown imputation method: {imputation_method}")

            imputed_array = imputer.fit_transform(impute_subset)
            imputed_df = pd.DataFrame(imputed_array, columns=features, index=cluster_data.index)

            mask = cluster_data[target].isnull()
            filled_values = imputed_df.loc[mask, target]
            cluster_data.loc[mask, target] = filled_values

            filled = filled_values.notna().sum()
            remaining = cluster_data[target].isnull().sum()
            stats[target] = {
                'filled': filled,
                'remaining': remaining
            }

    except Exception as e:
        logging.error(f"Error processing cluster: {str(e)}")
        for target in target_columns:
            stats[target] = {
                'filled': 0,
                'remaining': cluster_data[target].isnull().sum(),
                'error': str(e)
            }

    return cluster_data, stats

def impute_missing_values_with_clustering(
    data: pd.DataFrame,
    target_columns: Union[List[str], str],
    estimation_features: List[str],
    n_clusters: int = 5,
    imputation_method: str = 'iterative',
    n_neighbors: int = 5,
    min_features: Optional[int] = None,  # Changed to Optional
    n_jobs: int = -1,
    use_rf_importance: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cluster-based missing value imputation with dynamic feature selection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing the data
    target_columns : Union[List[str], str]
        Column(s) containing missing values to impute
    estimation_features : List[str]
        Features to use for estimation
    n_clusters : int, default=5
        Number of clusters for KMeans
    imputation_method : str, default='iterative'
        Method for imputation ('knn' or 'iterative')
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation
    min_features : Optional[int], default=None
        Minimum number of features to select. If None, automatically determined
    n_jobs : int, default=-1
        Number of parallel jobs
    use_rf_importance : bool, default=True
        Whether to use Random Forest for feature importance
    random_state : Optional[int], default=42
        Random state for reproducibility
    """
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Convert target_columns to list if string
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # Validate columns existence
    available_target_columns = [col for col in target_columns if col in data.columns]
    if not available_target_columns:
        raise ValueError(f"Target columns not found in data: {target_columns}")

    available_estimation_features = [col for col in estimation_features if col in data.columns]
    if not available_estimation_features:
        raise ValueError(f"Estimation features not found in data: {estimation_features}")

    data = data.copy()

    # Handle categorical features
    categorical_features = data[available_estimation_features].select_dtypes(
        include=['object', 'category']).columns.tolist()
    if categorical_features:
        logger.info(f"Processing categorical features: {categorical_features}")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(data[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
        data = pd.concat([data, encoded_df], axis=1)
        available_estimation_features = [feat for feat in available_estimation_features 
                                      if feat not in categorical_features] + list(encoded_feature_names)

    # Select numeric features
    numeric_estimation_features = data[available_estimation_features].select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    if not numeric_estimation_features:
        raise ValueError("No numeric estimation features available.")

    # Dynamically determine min_features if not specified
    if min_features is None:
        min_features = min(3, len(numeric_estimation_features))
    else:
        min_features = min(min_features, len(numeric_estimation_features))

    # Feature selection
    selected_features = []
    if use_rf_importance and len(numeric_estimation_features) > 1:
        logger.info("Using Random Forest for feature importance")
        valid_rows = data.dropna(subset=numeric_estimation_features + available_target_columns)
        if not valid_rows.empty:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
            rf.fit(valid_rows[numeric_estimation_features], valid_rows[available_target_columns])
            feature_importances = pd.DataFrame({
                'feature': numeric_estimation_features,
                'importance': rf.feature_importances_
            }).groupby('feature').mean()
            selected_features = feature_importances.sort_values(
                by='importance', ascending=False).head(min_features).index.tolist()
            logger.info(f"Selected features (Random Forest): {selected_features}")

    if len(selected_features) < min_features:
        logger.info("Using SelectKBest for feature selection")
        k = min(min_features, len(numeric_estimation_features))
        selector = SelectKBest(score_func=f_regression, k=k)
        valid_rows = data.dropna(subset=numeric_estimation_features + available_target_columns)
        if not valid_rows.empty:
            selector.fit(valid_rows[numeric_estimation_features], valid_rows[available_target_columns])
            selected_features = [feature for feature, selected in 
                               zip(numeric_estimation_features, selector.get_support()) if selected]
            logger.info(f"Selected features (SelectKBest): {selected_features}")

    # Ensure we have at least one feature
    if not selected_features and numeric_estimation_features:
        selected_features = numeric_estimation_features[:min_features]
        logger.info(f"Using all available features: {selected_features}")

    if not selected_features:
        raise ValueError("No features available for imputation after selection process.")

    # Initialize statistics
    imputation_stats = {col: {
        'initial_missing': data[col].isnull().sum(),
        'final_missing': 0,
        'filled_values': 0,
        'errors': []
    } for col in available_target_columns}

    # Initial imputation for estimation features
    logger.info("Performing initial imputation on estimation features")
    estimator_imputer = SimpleImputer(strategy='median')
    data[numeric_estimation_features] = estimator_imputer.fit_transform(data[numeric_estimation_features])

    # Adjust n_clusters based on data size
    n_clusters = min(n_clusters, len(data) // 2)
    if n_clusters < 2:
        n_clusters = 2
    logger.info(f"Clustering data into {n_clusters} clusters")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data[selected_features])
    data['cluster'] = cluster_labels

    # Process clusters
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cluster)(
            data[data['cluster'] == cluster_id].copy(),
            available_target_columns,
            selected_features,
            imputation_method,
            n_neighbors,
            random_state
        ) for cluster_id in range(n_clusters)
    )

    # Merge results
    for cluster_data, cluster_stats in results:
        data.loc[cluster_data.index, available_target_columns] = cluster_data[available_target_columns]
        for target, stats_dict in cluster_stats.items():
            if target in imputation_stats:
                imputation_stats[target]['filled_values'] += stats_dict.get('filled', 0)
                imputation_stats[target]['final_missing'] += stats_dict.get('remaining', 0)
                if 'error' in stats_dict:
                    imputation_stats[target]['errors'].append(stats_dict['error'])

    # Clean up
    data.drop('cluster', axis=1, inplace=True)
    
    logger.info("Imputation completed")
    return data, imputation_stats
