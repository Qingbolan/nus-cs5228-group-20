# %% [markdown]
# ## Data Preprocessing for Car Price Prediction

# %%
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Union, Dict, Tuple, Optional
import seaborn as sns
from fuzzywuzzy import process

# %% [markdown]
# ### Load Data

# %%
# Load the dataset
data = pd.read_csv('data/for-experiment-raw/for_test.csv')

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
    best_match = process.extractOne(x, choices)
    return best_match[0] if best_match and best_match[1] >= cutoff else x

def clean_string(s):
    s = s.lower()
    s = s.replace('/', '_')
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip()

def clean_column_name(s):
    s = s.lower()
    s = s.replace('/', '_')
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip()

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
        stats['initial_missing'] = data[target_column].isnull().sum()

        is_categorical = False
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            is_categorical = True
            le = LabelEncoder()
            data[target_column] = le.fit_transform(data[target_column].astype(str))

        numeric_features = data[available_estimation_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = []

        if numeric_features:
            valid_data = data.dropna(subset=[target_column] + numeric_features)
            if not valid_data.empty:
                if use_rf_importance:
                    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
                    rf.fit(valid_data[numeric_features], valid_data[target_column])
                    importance = pd.Series(rf.feature_importances_, index=numeric_features)
                    selected_features = importance.nlargest(min_features + 1).index.tolist()
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(min_features + 1, len(numeric_features)))
                    selector.fit(valid_data[numeric_features], valid_data[target_column])
                    selected_features = [feature for feature, selected in 
                                         zip(numeric_features, selector.get_support()) if selected]

        # Ensure target_column is not in selected_features
        selected_features = [feat for feat in selected_features if feat != target_column]

        if len(selected_features) < min_features and numeric_features:
            additional_features = [f for f in numeric_features if f not in selected_features and f != target_column]
            needed = min_features - len(selected_features)
            selected_features += additional_features[:needed]

        if not selected_features:
            msg = f"Target column '{target_column}' has insufficient numeric features for imputation. Filling with median."
            logger.warning(msg)
            stats['errors'].append(msg)
            median_value = data[target_column].median()
            data[target_column].fillna(median_value, inplace=True)
            stats['final_missing'] = data[target_column].isnull().sum()
            stats['filled_values'] = stats['initial_missing'] - stats['final_missing']
            if is_categorical:
                data[target_column] = le.inverse_transform(data[target_column].astype(int))
            return target_column, data[target_column], stats

        # Remove duplicates and ensure target_column is last
        impute_features = selected_features + [target_column]
        impute_features = list(dict.fromkeys(impute_features))

        impute_data = data[impute_features].copy()

        scaler = StandardScaler()
        impute_data[selected_features] = scaler.fit_transform(impute_data[selected_features])

        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_array = imputer.fit_transform(impute_data)
        imputed_df = pd.DataFrame(imputed_array, columns=impute_features, index=data.index)

        imputed_df[selected_features] = scaler.inverse_transform(imputed_df[selected_features])

        mask = data[target_column].isnull()
        filled_values = imputed_df.loc[mask, target_column]
        data.loc[mask, target_column] = filled_values
        stats['filled_values'] = filled_values.notna().sum()

        stats['final_missing'] = data[target_column].isnull().sum()

        if is_categorical:
            data[target_column] = le.inverse_transform(data[target_column].astype(int))

        return target_column, data[target_column], stats

    except Exception as e:
        error_msg = f"Error processing column '{target_column}': {str(e)}"
        logger.error(error_msg)
        stats['errors'].append(error_msg)
        return target_column, data[target_column], stats

def impute_missing_values_optimized(
    data: pd.DataFrame,
    target_columns: Union[List[str], str],
    estimation_features: List[str],
    n_neighbors: int = 5,
    min_features: int = 3,
    n_jobs: int = -1,
    use_rf_importance: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict]:
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        data = data.copy()
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        available_target_columns = [col for col in target_columns if col in data.columns]
        if not available_target_columns:
            raise ValueError(f"Target columns not found in data: {target_columns}")

        available_estimation_features = [col for col in estimation_features if col in data.columns]
        if not available_estimation_features:
            raise ValueError(f"Estimation features not found in data: {estimation_features}")

        # Remove target_columns from estimation_features
        available_estimation_features = list(set(available_estimation_features) - set(target_columns))

        categorical_features = data[available_estimation_features].select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            logger.info(f"Handling categorical features: {categorical_features}")
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(data[categorical_features].fillna('Missing'))
            encoded_feature_names = []
            for i, feature in enumerate(categorical_features):
                categories = encoder.categories_[i]
                encoded_feature_names.extend([f"{clean_column_name(feature)}_{clean_column_name(str(cat))}" for cat in categories])
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)
            available_estimation_features = [feat for feat in available_estimation_features 
                                             if feat not in categorical_features] + encoded_feature_names

            # Ensure no duplicate columns after encoding
            data.columns = make_unique_columns(data.columns.tolist())
            logger.info("Ensured unique column names after handling categorical features.")

        # Initialize statistics
        imputation_stats = {col: {
            'initial_missing': data[col].isnull().sum(),
            'final_missing': 0,
            'filled_values': 0,
            'errors': []
        } for col in available_target_columns}

        # Process each target column
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
                    error_msg = f"Error processing column '{target_column}': {str(e)}"
                    logger.error(error_msg)
                    imputation_stats[target_column]['errors'].append(error_msg)

        # Update final statistics
        for col in available_target_columns:
            if 'final_missing' not in imputation_stats[col]:
                imputation_stats[col]['final_missing'] = data[col].isnull().sum()
                imputation_stats[col]['filled_values'] = (
                    imputation_stats[col]['initial_missing'] - 
                    imputation_stats[col]['final_missing']
                )

        logger.info("Missing value imputation completed.")
        return data, imputation_stats

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return data, {'error': str(e)}

# %% [markdown]
# ### Data Preprocessing Steps

# %%
print("\nAll column names:")
print(data.columns.tolist())

# %% [markdown]
# #### 1. Clean 'title' Column

# %%
data['cleaned_title'] = data['title'].apply(clean_title)
data = data.drop('title', axis=1)

# %% [markdown]
# #### 2. Process 'make' Column

# %%
print("Number of missing 'make' values before cleaning:", data['make'].isnull().sum())
known_makes = data['make'].dropna().unique()
data.loc[data['make'].isnull(), 'make'] = data.loc[data['make'].isnull(), 'cleaned_title'].apply(
    lambda x: find_best_match(x.split()[0], known_makes)
)
print("Number of missing 'make' values after cleaning:", data['make'].isnull().sum())

# %% [markdown]
# #### 3. Process 'model' Column

# %%
print("Number of missing 'model' values before cleaning:", data['model'].isnull().sum())
data['model'] = data['model'].fillna(data['cleaned_title'])
print("Number of missing 'model' values after cleaning:", data['model'].isnull().sum())
data = data.drop('cleaned_title', axis=1)

# %% [markdown]
# #### 4. Drop 'description' Column

# %%
data = data.drop('description', axis=1)

# %% [markdown]
# #### 5. Process 'manufactured' Column

# %%
data['manufactured'] = pd.to_numeric(data['manufactured'], errors='coerce')
mean_manufactured = data.groupby('type_of_vehicle')['manufactured'].transform('mean')
data['manufactured'] = data['manufactured'].fillna(mean_manufactured)
data['manufactured'] = data['manufactured'].fillna(data['manufactured'].mean())
data['manufactured'] = data['manufactured'].astype(int)
print(f"Number of missing 'manufactured' values after processing: {data['manufactured'].isnull().sum()}")

# %% [markdown]
# #### 6. Drop 'original_reg_date' Column

# %%
data = data.drop('original_reg_date', axis=1)

# %% [markdown]
# #### 7. Process 'reg_date' and Create 'vehicle_age'

# %%
data['reg_date'] = pd.to_datetime(data['reg_date'])
current_date = pd.Timestamp.now()
data['vehicle_age'] = (current_date - data['reg_date']).astype('<m8[Y]')
data = data.drop('reg_date', axis=1)

# %% [markdown]
# #### 8. Process 'type_of_vehicle' Column

# %%
data['type_of_vehicle'] = data['type_of_vehicle'].apply(clean_string)
data['type_of_vehicle'] = data['type_of_vehicle'].str.split()
data = data.explode('type_of_vehicle')
data['type_of_vehicle'] = data['type_of_vehicle'].apply(clean_column_name)
data_type_of_vehicle_encoded = pd.get_dummies(data['type_of_vehicle'], prefix='vehicle_type').astype(int)
data_type_of_vehicle_encoded = data_type_of_vehicle_encoded.loc[:, ~data_type_of_vehicle_encoded.columns.duplicated()]
col_idx = data.columns.get_loc('type_of_vehicle')

# Insert the one-hot encoded columns
for col in data_type_of_vehicle_encoded.columns:
    data.insert(col_idx, col, data_type_of_vehicle_encoded[col])

# Ensure unique column names
data.columns = make_unique_columns(data.columns.tolist())

# Drop the original 'type_of_vehicle' column
data.drop('type_of_vehicle', axis=1, inplace=True)

# %% [markdown]
# #### 9. Process 'category' Column

# %%
data['category'] = data['category'].replace('-', '')
categories_split = data['category'].apply(lambda x: [clean_column_name(item.strip()) for item in x.split(',')] if isinstance(x, str) and x else [])
mlb = MultiLabelBinarizer()
category_encoded = mlb.fit_transform(categories_split)
category_df = pd.DataFrame(category_encoded, columns=[f"category_{clean_column_name(str(c))}" for c in mlb.classes_])
category_df = category_df.reset_index(drop=True)
data = data.reset_index(drop=True)
data = pd.concat([data.drop('category', axis=1), category_df], axis=1)

# Ensure unique column names
data.columns = make_unique_columns(data.columns.tolist())

# %% [markdown]
# #### 10. Process 'transmission' Column

# %%
data_transmission_encoded = pd.get_dummies(data['transmission'], prefix='transmission_type').astype(int)
data_transmission_encoded = data_transmission_encoded.loc[:, ~data_transmission_encoded.columns.duplicated()]
col_idx = data.columns.get_loc('transmission')

# Insert the one-hot encoded columns
for col in data_transmission_encoded.columns:
    data.insert(col_idx, col, data_transmission_encoded[col])

# Ensure unique column names
data.columns = make_unique_columns(data.columns.tolist())

# Drop the original 'transmission' column
data.drop('transmission', axis=1, inplace=True)

# %% [markdown]
# #### 11. Drop Irrelevant Columns

# %%
columns_to_drop = [
    "fuel_type", "mileage", "opc_scheme", "lifespan", "eco_category",
    "features", "accessories", "indicative_price",
    "make", "model"
]
columns_to_drop = [clean_column_name(col) for col in columns_to_drop]
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
data = data.drop(columns=existing_columns_to_drop, axis=1)

# Ensure unique column names
data.columns = make_unique_columns(data.columns.tolist())

# %% [markdown]
# #### 12. Clean Column Names

# %%
data.columns = data.columns.map(clean_column_name)
data.columns = make_unique_columns(data.columns.tolist())

# %% [markdown]
# #### 13. Missing Value Imputation

# %%
target_columns = [
    'curb_weight', 'power', 'engine_cap', 'no_of_owners',
    'depreciation', 'road_tax', 'dereg_value', 'omv', 'arf'
]
estimation_features = [
    'engine_cap', 'power', 'road_tax', 'omv', 'arf',
    'curb_weight', 'vehicle_age'
] + list(data.columns[data.columns.str.startswith('vehicle_type_')]) + list(data.columns[data.columns.str.startswith('category_')])

# Remove target_columns from estimation_features
estimation_features = list(set(estimation_features) - set(target_columns))
estimation_features = [feat for feat in estimation_features if feat in data.columns]

# Apply missing value imputation
data, imputation_stats = impute_missing_values_optimized(
    data=data,
    target_columns=target_columns,
    estimation_features=estimation_features,
    n_neighbors=5,
    min_features=3,
    n_jobs=-1,
    use_rf_importance=True,
    random_state=42
)

# Print imputation statistics
for col, stats in imputation_stats.items():
    print(f"Column: {col}")
    print(f"Initial missing values: {stats['initial_missing']}")
    print(f"Final missing values: {stats['final_missing']}")
    print(f"Filled values: {stats['filled_values']}")
    if stats['errors']:
        print(f"Errors: {stats['errors']}")
    print("-" * 40)

# %% [markdown]
# #### 14. Final Checks and Saving Data

# %%
# Check for any remaining missing values
remaining_missing = data.isnull().sum()
print("Remaining missing values in each column:")
print(remaining_missing[remaining_missing > 0])

# Save the cleaned data
current_dir = os.getcwd()
save_filename = 'test_cleaned.csv'
save_filepath = os.path.join(current_dir, save_filename)
data.to_csv(save_filepath, index=False)
print(f"Cleaned data saved to {save_filepath}")
