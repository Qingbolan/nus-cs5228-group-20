# %% [markdown]
# <img src="images/cs5228-header-title.png" />

# %% [markdown]
# ### Group_work - Data Preprocessing and Optimized Missing Value Imputation

# %% [markdown]
# ### Step 1: Data Loading and Initial Preprocessing (Recording Missing Value Positions)

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('preprocessing\\2024-10-21-silan\\train_cleaned.csv', index_col=0)

print(f"Initial data shape: {data.shape}")
data.head()

# %%
# Function to clean title and extract useful information
def clean_title(title):
    # Extract the COE information
    title = re.sub(r'\(COE till \d{2}/\d{4}\)', '', title)
    # Remove the year, capacity and mileage information
    title = re.sub(r'\b\d{4}\b|\d+(\.\d+)?(cc|l|litre|liter)|[0-9]+\s*(km|kms)', '', title, flags=re.IGNORECASE)
    # Remove common words like 'auto', 'manual', 'diesel', 'petrol'
    title = re.sub(r'\b(auto|manual|diesel|petrol)\b', '', title, flags=re.IGNORECASE)
    # Remove extra whitespaces
    return ' '.join(title.split())

# Function to find the best match for make
from fuzzywuzzy import process

def find_best_match(x, choices, cutoff=80):
    best_match = process.extractOne(x, choices)
    return best_match[0] if best_match and best_match[1] >= cutoff else x

# Clean title and extract make and model
data['cleaned_title'] = data['title'].apply(clean_title)

# Drop the original title column
# data = data.drop('title', axis=1)

# Process make
print("Number of missing 'make':", data['make'].isnull().sum())

# Gain the known makes
known_makes = data['make'].dropna().unique()

# Fill the missing values in 'make'
data.loc[data['make'].isnull(), 'make'] = data.loc[data['make'].isnull(), 'cleaned_title'].apply(
    lambda x: find_best_match(x.split()[0], known_makes)
)

print("Number of missing 'make' after filling:", data['make'].isnull().sum())

# Drop 'description' column
data = data.drop('description', axis=1)

# Convert 'manufactured' to numeric
data['manufactured'] = pd.to_numeric(data['manufactured'], errors='coerce')

# Drop 'original_reg_date' column
data = data.drop('original_reg_date', axis=1)

# Process 'reg_date' and create 'vehicle_age'
data['reg_date'] = pd.to_datetime(data['reg_date'], errors='coerce')
current_date = pd.Timestamp.now()
data['vehicle_age'] = (current_date - data['reg_date']).astype('<m8[Y]')
data = data.drop('reg_date', axis=1)

# Process 'type_of_vehicle'
def clean_string(s):
    s = s.lower()
    s = s.replace('/', '_')
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip()

data['type_of_vehicle'] = data['type_of_vehicle'].apply(clean_string)
data['type_of_vehicle'] = data['type_of_vehicle'].str.split()

# Explode 'type_of_vehicle' into multiple rows
data = data.explode('type_of_vehicle')

# One-hot encode 'type_of_vehicle'
data_type_of_vehicle_encoded = pd.get_dummies(data['type_of_vehicle'], prefix='vehicle_type').astype(int)
data = pd.concat([data.drop('type_of_vehicle', axis=1), data_type_of_vehicle_encoded], axis=1)

# Process 'category'
data['category'] = data['category'].replace('-', '')

# Split 'category' by comma
categories_split = data['category'].apply(lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) and x else [])

# One-hot encode 'category' using MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
category_encoded = mlb.fit_transform(categories_split)
category_df = pd.DataFrame(category_encoded, columns=mlb.classes_)
data = pd.concat([data.drop('category', axis=1).reset_index(drop=True), category_df.reset_index(drop=True)], axis=1)

# Process 'transmission'
data_transmission_encoded = pd.get_dummies(data['transmission'], prefix='transmission').astype(int)
data = pd.concat([data.drop('transmission', axis=1), data_transmission_encoded], axis=1)

# Remove 'cleaned_title' as it's no longer needed
data = data.drop('cleaned_title', axis=1)

# Drop columns that will not be used or have too many missing values
data = data.drop(columns=["fuel_type", "mileage", "opc_scheme", "lifespan", "eco_category", "features", "accessories", "indicative_price"])

# Clean column names
def clean_column_name(s):
    s = s.lower()
    s = s.replace('/', '_')
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip()

data.columns = data.columns.map(clean_column_name)

# Record missing value positions
missing_value_positions = data.isnull()
print("Missing value positions:")
print(missing_value_positions.sum())

print(f"Data shape after preprocessing (with missing values): {data.shape}")
data.head()

# %% [markdown]
# ### Step 2: Convert Categorical Features to Numeric

# %%
# Identify categorical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features)

# For simplicity, we can use Label Encoding for 'make' and 'model'

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = data[col].astype(str)  # Ensure all data is str type
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Now, all features should be numeric
print("Data types after label encoding:")
print(data.dtypes)

columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
columns_to_standardize = [col for col in columns_to_standardize if col in data.columns]
data[columns_to_standardize] = data[columns_to_standardize].astype(float)

# %% [markdown]
# ### Step 3: Load Trained Model and Impute Missing Values

# %%
# Load the trained model
# Assuming you have trained the model and saved it as 'trained_model.pkl'
import pickle

# Load the trained price prediction model
with open('optimized_svr_models.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# Now, we will train models to predict missing values for each feature
# List of features with missing values
features_with_missing = data.columns[data.isnull().any()].tolist()
print("Features with missing values:", features_with_missing)

# Note: Since we have converted categorical features to numeric, we can proceed to train models

from sklearn.ensemble import RandomForestRegressor

# Iterate over each feature and fill missing values
for feature in features_with_missing:
    print(f"Imputing missing values for {feature}...")
    # Create a copy of the data excluding rows where this feature is missing
    data_known = data[data[feature].notnull()]
    data_unknown = data[data[feature].isnull()]
    
    # If there are no known values for this feature, we cannot train a model
    if data_known.shape[0] == 0:
        print(f"No known values for {feature}, cannot impute.")
        continue

    # Features to use for training (excluding the target feature and any other features with missing values)
    other_features = data.columns.difference([feature] + features_with_missing).tolist()
    
    # If there are no other features to use for prediction, we cannot train a model
    if len(other_features) == 0:
        print(f"No other features to predict {feature}, cannot impute.")
        continue

    # Prepare training data
    X_train = data_known[other_features]
    y_train = data_known[feature]
    
    # Prepare data to predict
    X_pred = data_unknown[other_features]
    
    # Train a model to predict this feature
    impute_model = RandomForestRegressor(n_estimators=100, random_state=42)
    impute_model.fit(X_train, y_train)
    
    # Predict missing values
    y_pred = impute_model.predict(X_pred)
    
    # Fill the missing values
    data.loc[data[feature].isnull(), feature] = y_pred

# After imputing all missing values
print("Missing values after imputation:")
print(data.isnull().sum())

# %% [markdown]
# ### Step 4: Save the Optimized Cleaned Data

# %%
# Now, data has all missing values filled
# Save the optimized cleaned data
current_dir = os.getcwd()
save_filename = 'train_cleaned_optimized.csv'
save_filepath = os.path.join(current_dir, save_filename)
data.to_csv(save_filepath, index=False)
print(f"Optimized cleaned data saved to {save_filepath}")