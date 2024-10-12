import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_regression

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