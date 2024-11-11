def main():
    preprocess('data/for-experiment-raw/for_test.csv', 'test_cleaned.csv')
    preprocess('data/for-experiment-raw/for_train.csv', 'train_cleaned.csv')

def preprocess(input_path, output_path):
    # %% [markdown]
    # ## Data Preprocessing for Car Price Prediction

    # %%
    import os
    import re
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    # from sklearn.ensemble import RandomForestRegressor  # 添加了缺失的导入
    import logging
    from typing import List, Union, Dict, Tuple, Optional
    from rapidfuzz import process
    import warnings

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # %% [markdown]
    # ### Load Data

    # %%
    # Load the dataset
    data = pd.read_csv(input_path)
    logger.info(f"Data loaded from {input_path}. Shape: {data.shape}")

    # %% [markdown]
    # ### Function Definitions

    # %%
    def clean_title(title):
        coe_match = re.search(r'\(COE till (\d{2}/\d{4})\)', title)
        coe_expiry = coe_match.group(1) if coe_match else None
        title = re.sub(r'\(COE till \d{2}/\d{4}\)', '', title)
        title = re.sub(r'\b\d{4}\b|\d+(\.\d+)?(cc|l|litre|liter)|[0-9]+\s*(km|kms)', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\b(auto|manual|diesel|petrol)\b', '', title, flags=re.IGNORECASE)
        return ' '.join(title.split())

    def find_best_match(x, choices, cutoff=80):
        best_match = process.extractOne(x, choices, score_cutoff=cutoff)
        return best_match[0] if best_match else x

    def clean_string(s):
        if isinstance(s, str):
            s = s.lower()
            s = s.replace('/', '_')
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', '_', s)
            return s.strip()
        else:
            return ''

    def make_unique_columns(columns: List[str]) -> List[str]:
        """
        Make column names unique by appending suffixes to duplicates.
        """
        seen = {}
        unique_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)
        return unique_columns

    def impute_missing_values_serial(
        data: pd.DataFrame,
        target_columns: Union[List[str], str],
        estimation_features: List[str],
        n_neighbors: int = 5,
        min_features: int = 3,
        use_rf_importance: bool = True,
        random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, Dict]:
        try:
            if isinstance(target_columns, str):
                target_columns = [target_columns]

            available_target_columns = [col for col in target_columns if col in data.columns]
            available_estimation_features = [col for col in estimation_features if col in data.columns]
            available_estimation_features = list(set(available_estimation_features) - set(target_columns))

            imputation_stats = {}

            for target_column in available_target_columns:
                stats = {
                    'initial_missing': data[target_column].isnull().sum(),
                    'final_missing': 0,
                    'filled_values': 0,
                    'errors': []
                }

                try:
                    numeric_features = data[available_estimation_features].select_dtypes(include=[np.number]).columns.tolist()
                    selected_features = []

                    if numeric_features:
                        valid_data = data.dropna(subset=[target_column] + numeric_features)
                        if not valid_data.empty and use_rf_importance:
                            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
                            rf.fit(valid_data[numeric_features], valid_data[target_column])
                            importance = pd.Series(rf.feature_importances_, index=numeric_features)
                            selected_features = importance.nlargest(min_features + 1).index.tolist()
                        else:
                            selected_features = numeric_features[:min_features + 1]

                    # Ensure target_column is not in selected_features
                    selected_features = [feat for feat in selected_features if feat != target_column]

                    if len(selected_features) < min_features:
                        msg = f"Target column '{target_column}' has insufficient numeric features for imputation. Filling with median."
                        stats['errors'].append(msg)
                        data[target_column].fillna(data[target_column].median(), inplace=True)
                    else:
                        impute_features = selected_features + [target_column]
                        impute_data = data[impute_features].copy()

                        scaler = StandardScaler()
                        impute_data[selected_features] = scaler.fit_transform(impute_data[selected_features])

                        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                        imputed_array = imputer.fit_transform(impute_data)
                        imputed_df = pd.DataFrame(imputed_array, columns=impute_features, index=data.index)

                        imputed_df[selected_features] = scaler.inverse_transform(imputed_df[selected_features])

                        mask = data[target_column].isnull()
                        data.loc[mask, target_column] = imputed_df.loc[mask, target_column]

                    stats['final_missing'] = data[target_column].isnull().sum()
                    stats['filled_values'] = stats['initial_missing'] - stats['final_missing']

                    logger.info(f"Processed column '{target_column}': filled {stats['filled_values']} missing values.")

                except Exception as e:
                    error_msg = f"Error processing column '{target_column}': {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)

                imputation_stats[target_column] = stats

            logger.info("Missing value imputation completed.")
            return data, imputation_stats

        except Exception as e:
            logger.error(f"An error occurred during imputation: {str(e)}")
            return data, {'error': str(e)}

    # %% [markdown]
    # ### Data Preprocessing Steps

    # %%
    logger.info("Starting data preprocessing...")

    # Log all column names
    logger.info(f"All column names: {data.columns.tolist()}")

    # %% [markdown]
    # #### 1. Clean 'title' Column

    # %%
    data['cleaned_title'] = data['title'].apply(clean_title)
    data.drop('title', axis=1, inplace=True)
    logger.info("'title' column cleaned and dropped.")

    # %% [markdown]
    # #### 2. Process 'make' Column

    # %%
    missing_make_before = data['make'].isnull().sum()
    logger.info(f"Missing 'make' values before cleaning: {missing_make_before}")
    known_makes = data['make'].dropna().unique()
    data.loc[data['make'].isnull(), 'make'] = data.loc[data['make'].isnull(), 'cleaned_title'].apply(
        lambda x: find_best_match(x.split()[0], known_makes)
    )
    missing_make_after = data['make'].isnull().sum()
    logger.info(f"Missing 'make' values after cleaning: {missing_make_after}")

    # %% [markdown]
    # #### 3. Process 'model' Column

    # %%
    missing_model_before = data['model'].isnull().sum()
    logger.info(f"Missing 'model' values before cleaning: {missing_model_before}")
    data['model'] = data['model'].fillna(data['cleaned_title'])
    missing_model_after = data['model'].isnull().sum()
    logger.info(f"Missing 'model' values after cleaning: {missing_model_after}")
    data.drop('cleaned_title', axis=1, inplace=True)

    # %% [markdown]
    # #### 4. Drop 'description' Column

    # %%
    if 'description' in data.columns:
        data.drop('description', axis=1, inplace=True)
        logger.info("'description' column dropped.")

    # %% [markdown]
    # #### 5. Process 'manufactured' Column

    # %%
    data['manufactured'] = pd.to_numeric(data['manufactured'], errors='coerce')
    missing_manufactured_before = data['manufactured'].isnull().sum()
    logger.info(f"Missing 'manufactured' values before processing: {missing_manufactured_before}")
    mean_manufactured = data.groupby('type_of_vehicle')['manufactured'].transform('mean')
    data['manufactured'] = data['manufactured'].fillna(mean_manufactured)
    data['manufactured'] = data['manufactured'].fillna(data['manufactured'].mean())
    data['manufactured'] = data['manufactured'].astype(int)
    missing_manufactured_after = data['manufactured'].isnull().sum()
    logger.info(f"Missing 'manufactured' values after processing: {missing_manufactured_after}")

    # %% [markdown]
    # #### 6. Drop 'original_reg_date' Column

    # %%
    if 'original_reg_date' in data.columns:
        data.drop('original_reg_date', axis=1, inplace=True)
        logger.info("'original_reg_date' column dropped.")

    # %% [markdown]
    # #### 7. Process 'reg_date' and Create 'vehicle_age'

    # %%
    data['reg_date'] = pd.to_datetime(data['reg_date'], errors='coerce')
    missing_reg_date = data['reg_date'].isnull().sum()
    logger.info(f"Missing 'reg_date' values: {missing_reg_date}")
    current_date = pd.Timestamp.now()
    data['vehicle_age'] = (current_date - data['reg_date']).dt.days / 365
    data.drop('reg_date', axis=1, inplace=True)
    logger.info("'vehicle_age' column created and 'reg_date' column dropped.")

    # %% [markdown]
    # #### 8. Process 'type_of_vehicle' Column

    # %%
    data['type_of_vehicle'] = data['type_of_vehicle'].apply(clean_string)
    data['type_of_vehicle'] = data['type_of_vehicle'].str.split()
    mlb_vehicle_type = MultiLabelBinarizer()
    vehicle_type_encoded = mlb_vehicle_type.fit_transform(data['type_of_vehicle'])
    vehicle_type_df = pd.DataFrame(vehicle_type_encoded, columns=[f"vehicle_type_{clean_string(c)}" for c in mlb_vehicle_type.classes_])
    data = pd.concat([data.drop('type_of_vehicle', axis=1).reset_index(drop=True), vehicle_type_df], axis=1)
    logger.info("'type_of_vehicle' column processed.")

    # Ensure unique column names
    data.columns = make_unique_columns(data.columns.tolist())

    # %% [markdown]
    # #### 9. Process 'category' Column

    # %%
    data['category'] = data['category'].replace('-', '')
    categories_split = data['category'].apply(lambda x: [clean_string(item.strip()) for item in x.split(',')] if isinstance(x, str) and x else [])
    mlb = MultiLabelBinarizer()
    category_encoded = mlb.fit_transform(categories_split)
    category_df = pd.DataFrame(category_encoded, columns=[f"category_{clean_string(str(c))}" for c in mlb.classes_])
    data = pd.concat([data.drop('category', axis=1).reset_index(drop=True), category_df], axis=1)
    logger.info("'category' column processed.")

    # Ensure unique column names
    data.columns = make_unique_columns(data.columns.tolist())

    # %% [markdown]
    # #### 10. Process 'transmission' Column

    # %%
    data_transmission_encoded = pd.get_dummies(data['transmission'], prefix='transmission_type').astype(int)
    col_idx = data.columns.get_loc('transmission')

    # Insert the one-hot encoded columns
    for col in data_transmission_encoded.columns:
        data.insert(col_idx, col, data_transmission_encoded[col])

    # Ensure unique column names
    data.columns = make_unique_columns(data.columns.tolist())

    # Drop the original 'transmission' column
    data.drop('transmission', axis=1, inplace=True)
    logger.info("'transmission' column processed.")

    # %% [markdown]
    # #### 11. Drop or Combine Highly Correlated Features

    # %%
    # Remove 'road_tax' and 'arf' due to high correlation with other features
    columns_to_drop = ['road_tax', 'arf']
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_columns_to_drop, axis=1, inplace=True)
    logger.info(f"Dropped highly correlated columns: {existing_columns_to_drop}")

    # Combine 'omv', 'arf', 'dereg_value' into a single feature (e.g., average)
    if all(col in data.columns for col in ['omv', 'dereg_value']):
        data['value_mean'] = data[['omv', 'dereg_value']].mean(axis=1)
        logger.info("Created 'value_mean' feature by averaging 'omv' and 'dereg_value'.")

    # Drop 'omv' and 'dereg_value' if necessary (optional)
    # data.drop(['omv', 'dereg_value'], axis=1, inplace=True)

    # %% [markdown]
    # #### 12. Encode 'make' and 'model' Columns

    # %%
    for col in ['make', 'model']:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            logger.info(f"'{col}' column encoded.")

    # %% [markdown]
    # #### 13. Create New Features

    # %%
    # Create 'power_to_weight' feature
    if 'power' in data.columns and 'curb_weight' in data.columns:
        data['power_to_weight'] = data['power'] / data['curb_weight']
        logger.info("Created 'power_to_weight' feature.")

    # %% [markdown]
    # #### 14. Bin Weakly Correlated Features

    # %%
    # Bin 'no_of_owners' if it exists
    if 'no_of_owners' in data.columns:
        data['no_of_owners_bin'] = pd.qcut(data['no_of_owners'], q=3, labels=False, duplicates='drop')
        logger.info("Binned 'no_of_owners' into 'no_of_owners_bin'.")

    # Drop original 'no_of_owners' column if necessary
    # data.drop('no_of_owners', axis=1, inplace=True)

    # %% [markdown]
    # #### 15. Drop Irrelevant or Redundant Columns

    # %%
    columns_to_drop = [
        "fuel_type", "mileage", "opc_scheme", "lifespan", "eco_category",
        "features", "accessories", "indicative_price"
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_columns_to_drop, axis=1, inplace=True)
    logger.info(f"Dropped irrelevant or redundant columns: {existing_columns_to_drop}")

    # Ensure unique column names
    data.columns = make_unique_columns(data.columns.tolist())

    # %% [markdown]
    # #### 16. Clean Column Names

    # %%
    data.columns = data.columns.map(clean_string)
    data.columns = make_unique_columns(data.columns.tolist())
    logger.info("Column names cleaned.")

    # %% [markdown]
    # #### 17. Missing Value Imputation

    # %%
    target_columns = [
        'curb_weight', 'power', 'engine_cap', 'depreciation', 'dereg_value', 'omv'
    ]
    estimation_features = [
        'vehicle_age', 'manufactured', 'make', 'model', 'value_mean', 'power_to_weight'
    ] + [col for col in data.columns if col.startswith('vehicle_type_') or col.startswith('category_') or col.startswith('transmission_type_')]

    # Remove target_columns from estimation_features
    estimation_features = list(set(estimation_features) - set(target_columns))

    # Apply missing value imputation (serial)
    data, imputation_stats = impute_missing_values_serial(
        data=data,
        target_columns=target_columns,
        estimation_features=estimation_features,
        n_neighbors=5,
        min_features=3,
        use_rf_importance=True,
        random_state=42
    )

    # Log imputation statistics
    for col, stats in imputation_stats.items():
        logger.info(f"Imputation stats for '{col}': {stats}")

    # %% [markdown]
    # #### 18. Final Checks and Saving Data

    # %%
    # Check for any remaining missing values
    remaining_missing = data.isnull().sum()
    missing_cols = remaining_missing[remaining_missing > 0]
    if not missing_cols.empty:
        logger.warning("Remaining missing values in columns:")
        logger.warning(missing_cols)
    else:
        logger.info("No remaining missing values.")

    # Save the cleaned data
    save_filepath = os.path.abspath(output_path)
    data.to_csv(save_filepath, index=False)
    logger.info(f"Cleaned data saved to {save_filepath}")

    logger.info("Data preprocessing completed.")


if __name__ == '__main__':
    main()