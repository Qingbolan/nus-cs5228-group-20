# Import necessary libraries
import pandas as pd
import numpy as np
import os

# Define function to load data and compare missing values of specific columns
def load_and_compare_data(raw_file_path, cleaned_file_path, columns_to_compare):
    """
    Load raw and cleaned data, compare specified columns, and identify positions where missing values were filled.

    Parameters:
    - raw_file_path: Path to the raw data file
    - cleaned_file_path: Path to the cleaned data file
    - columns_to_compare: List of columns to compare for missing values

    Returns:
    - raw_data: DataFrame of raw data
    - cleaned_data: DataFrame of cleaned data
    - missing_value_positions: DataFrame indicating positions where raw data has missing and cleaned data has values
    """
    # Load raw data
    raw_data = pd.read_csv(raw_file_path)
    # Load cleaned data
    cleaned_data = pd.read_csv(cleaned_file_path)

    # Ensure the specified columns are in both dataframes
    for col in columns_to_compare:
        if col not in raw_data.columns or col not in cleaned_data.columns:
            raise ValueError(f"Column '{col}' not found in both datasets")

    # Compare missing values only on the specified columns
    missing_in_raw = raw_data[columns_to_compare].isnull()
    filled_in_cleaned = cleaned_data[columns_to_compare].notnull()
    missing_value_positions = missing_in_raw & filled_in_cleaned

    return raw_data, cleaned_data, missing_value_positions

# Define function to impute missing values using models trained on other data
def impute_missing_with_model(cleaned_data, missing_value_positions, columns_to_impute):
    """
    Re-impute the specified missing values in cleaned_data using models trained on available data.

    Parameters:
    - cleaned_data: DataFrame of cleaned data
    - missing_value_positions: DataFrame indicating positions to re-impute
    - columns_to_impute: List of columns to re-impute

    Returns:
    - optimized_data: DataFrame after re-imputing missing values
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    # Make a copy of the cleaned data to preserve original
    optimized_data = cleaned_data.copy()

    # Handle categorical variables before modeling
    categorical_cols = optimized_data.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        optimized_data[col] = optimized_data[col].astype(str)
        optimized_data[col] = le.fit_transform(optimized_data[col])
        label_encoders[col] = le

    # For each column that needs to be imputed
    for col in columns_to_impute:
        if col not in optimized_data.columns:
            print(f"Column '{col}' not found in data. Skipping.")
            continue
        if col not in missing_value_positions.columns:
            print(f"Column '{col}' not found in missing value positions. Skipping.")
            continue

        print(f"Re-imputing column: {col}")

        # Get the rows where this column needs to be imputed
        rows_to_impute = missing_value_positions[col]

        # Prepare the data for training and prediction
        # Use data where the target column is not missing to train
        data_not_missing = optimized_data[optimized_data[col].notnull()]
        data_missing = optimized_data.loc[rows_to_impute.index[rows_to_impute]]

        # Features to use for training (exclude the target column and columns to impute)
        exclude_cols = columns_to_impute + ['price']
        X_train = data_not_missing.drop(columns=exclude_cols, errors='ignore')
        y_train = data_not_missing[col]

        # Features for prediction
        X_pred = data_missing.drop(columns=exclude_cols, errors='ignore')

        # Check if there are enough samples to train
        if len(X_train) < 10:
            print(f"Not enough data to train model for column '{col}'. Skipping.")
            continue

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict the missing values
        y_pred = model.predict(X_pred)

        # Assign the predicted values
        optimized_data.loc[data_missing.index, col] = y_pred

    return optimized_data

# Main function
def main():
    # Define file paths
    raw_file_path = 'data\\for-experiment-raw\\for_train.csv'  # 替换为您的原始数据文件路径
    cleaned_file_path = 'preprocessing\\2024-10-21-silan\\train_cleaned.csv'  # 替换为您的清洗后数据文件路径

    # List of columns to compare and re-impute
    columns_to_compare = ['curb_weight', 'power', 'engine_cap', 'no_of_owners', 'depreciation',
                          'road_tax', 'dereg_value', 'omv', 'arf']

    # Load data and compare missing values of specified columns
    raw_data, cleaned_data, missing_value_positions = load_and_compare_data(
        raw_file_path, cleaned_file_path, columns_to_compare)

    # Impute missing values using models
    optimized_data = impute_missing_with_model(cleaned_data, missing_value_positions, columns_to_compare)

    # Save the optimized data
    optimized_data.to_csv('train_optimized_cleaned_data.csv', index=False)
    print("Optimized cleaned data saved to 'train_optimized_cleaned_data.csv'")
    
    
    # Define file paths
    raw_file_path = 'data\\for-experiment-raw\\for_test.csv'  # 替换为您的原始数据文件路径
    cleaned_file_path = 'preprocessing\\2024-10-21-silan\\test_cleaned.csv'  # 替换为您的清洗后数据文件路径

    # List of columns to compare and re-impute
    columns_to_compare = ['curb_weight', 'power', 'engine_cap', 'no_of_owners', 'depreciation',
                          'road_tax', 'dereg_value', 'omv', 'arf']

    # Load data and compare missing values of specified columns
    raw_data, cleaned_data, missing_value_positions = load_and_compare_data(
        raw_file_path, cleaned_file_path, columns_to_compare)

    # Impute missing values using models
    optimized_data = impute_missing_with_model(cleaned_data, missing_value_positions, columns_to_compare)

    # Save the optimized data
    optimized_data.to_csv('test_optimized_cleaned_data.csv', index=False)
    print("Optimized cleaned data saved to 'test_optimized_cleaned_data.csv'")


if __name__ == '__main__':
    main()