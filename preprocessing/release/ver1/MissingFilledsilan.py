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
