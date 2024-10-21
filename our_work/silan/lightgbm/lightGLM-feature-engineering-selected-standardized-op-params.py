import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from scipy.stats import skew, uniform, randint
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_and_preprocess_data(file_path, is_training=True):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    if is_training:
        X = data.drop('price', axis=1)
        y = data['price']
    else:
        X = data
        y = None
    
    return X, y

def safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def remove_outliers(X, y, columns, factor=1.5):
    for column in columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (X[column] >= lower_bound) & (X[column] <= upper_bound)
        X = X[mask]
        y = y[mask] if y is not None else None
    return X, y

def preprocess_features(X, y=None, is_training=True, num_imputer=None, cat_imputer=None, 
                        label_encoders=None, standard_scaler=None, robust_scaler=None):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    
    # Handle numeric features
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    else:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
    
    # Standardize specific columns
    if standard_scaler is None:
        standard_scaler = StandardScaler()
        X[columns_to_standardize] = standard_scaler.fit_transform(X[columns_to_standardize])
    else:
        X[columns_to_standardize] = standard_scaler.transform(X[columns_to_standardize])
    
    # Use RobustScaler for other numeric features
    other_numeric = [col for col in numeric_features if col not in columns_to_standardize]
    if robust_scaler is None:
        robust_scaler = RobustScaler()
        X[other_numeric] = robust_scaler.fit_transform(X[other_numeric])
    else:
        X[other_numeric] = robust_scaler.transform(X[other_numeric])
    
    # Handle categorical features
    if cat_imputer is None:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
    else:
        X[categorical_features] = cat_imputer.transform(X[categorical_features])
    
    # Label encode categorical features
    if label_encoders is None:
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    else:
        for col in categorical_features:
            le = label_encoders[col]
            X[col] = X[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
    
    # Feature engineering
    X['power_to_weight'] = safe_divide(X['power'], X['curb_weight'])
    X['engine_cap_to_weight'] = safe_divide(X['engine_cap'], X['curb_weight'])
    
    # Create interaction features for vehicle types
    vehicle_types = [col for col in X.columns if col.startswith('vehicle_type_')]
    for i, type1 in enumerate(vehicle_types):
        for type2 in vehicle_types[i+1:]:
            X[f'{type1}_{type2}_interaction'] = X[type1] * X[type2]
    
    # Handle skewed numerical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    skewed_features = X[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[skewed_features > 0.5]
    X[high_skew.index] = X[high_skew.index].apply(safe_log1p)
    
    # Final check for NaN and infinity values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        print("Warning: NaN values found after preprocessing. Applying additional imputation.")
        final_imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(final_imputer.fit_transform(X), columns=X.columns)
    
    return X, num_imputer, cat_imputer, label_encoders, standard_scaler, robust_scaler

def create_base_models():
    models = {
        'lgb': lgb.LGBMRegressor(
            random_state=42,
            device='gpu',
            min_child_samples=20,
            min_data_in_leaf=20,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'xgb': xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0),
        'ridge': Ridge(random_state=42, max_iter=10000),
        'lasso': Lasso(random_state=42, max_iter=10000),
        'elastic': ElasticNet(random_state=42, max_iter=10000),
        'gbr': GradientBoostingRegressor(random_state=42)
    }
    return models

def optimize_hyperparameters(X, y, models):
    optimized_models = {}
    
    param_distributions = {
        'lgb': {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(20, 100),
            'min_data_in_leaf': randint(20, 100)
        },
        'xgb': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': uniform(1, 10)
        },
        'ridge': {
            'alpha': uniform(0.1, 10)
        },
        'lasso': {
            'alpha': uniform(0.1, 10)
        },
        'elastic': {
            'alpha': uniform(0.1, 10),
            'l1_ratio': uniform(0, 1)
        },
        'gbr': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    }
    
    for name, model in models.items():
        print(f"Optimizing {name}...")
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_distributions[name], 
            n_iter=50,
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, 
            random_state=42,
            verbose=1
        )
        random_search.fit(X, y)
        optimized_models[name] = random_search.best_estimator_
        print(f"Best parameters for {name}: {random_search.best_params_}")
        print(f"Best score for {name}: {-random_search.best_score_}")
    
    return optimized_models

def train_stacking_model(X, y, base_models):
    estimators = [(name, model) for name, model in base_models.items()]
    final_estimator = Ridge()
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    
    stacking_regressor.fit(X, y)
    
    return stacking_regressor

def main():
    np.random.seed(42)
    
    # Load and preprocess training data
    X_train, y_train = load_and_preprocess_data('preprocessing/2024-10-10-silan/train_cleaned.csv', is_training=True)
    
    # Remove outliers
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train, y_train = remove_outliers(X_train, y_train, numeric_columns)
    
    # Apply RobustScaler to all numeric features
    scaler = RobustScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    
    X_train, num_imputer, cat_imputer, label_encoders, standard_scaler, robust_scaler = preprocess_features(X_train, y_train, is_training=True)
    
    # Check for NaN values
    if np.isnan(X_train.values).any():
        print("Error: NaN values still present after preprocessing. Please check your data and preprocessing steps.")
        return
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and optimize base models
    base_models = create_base_models()
    optimized_models = optimize_hyperparameters(X_train, y_train, base_models)
    
    # Evaluate individual models
    for name, model in optimized_models.items():
        val_predictions = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        print(f"{name} - RMSE: {np.sqrt(val_mse):.4f}, R2: {val_r2:.4f}")
    
    # Train stacking model
    stacking_model = train_stacking_model(X_train, y_train, optimized_models)
    
    # Evaluate stacking model
    val_predictions = stacking_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    print(f"Stacking Model - RMSE: {np.sqrt(val_mse):.4f}, R2: {val_r2:.4f}")
    
    # Retrain on full dataset
    X_full, y_full = load_and_preprocess_data('preprocessing/2024-10-10-silan/train_cleaned.csv', is_training=True)
    X_full, y_full = remove_outliers(X_full, y_full, numeric_columns)
    X_full[numeric_columns] = scaler.fit_transform(X_full[numeric_columns])
    X_full, _, _, _, _, _ = preprocess_features(X_full, y_full, is_training=True, 
                                                num_imputer=num_imputer, cat_imputer=cat_imputer, 
                                                label_encoders=label_encoders, standard_scaler=standard_scaler, 
                                                robust_scaler=robust_scaler)
    final_stacking_model = train_stacking_model(X_full, y_full, optimized_models)
    
    # Save model and preprocessing objects
    with open('final_stacking_model.pkl', 'wb') as f:
        pickle.dump(final_stacking_model, f)
    with open('preprocessing_objects.pkl', 'wb') as f:
        pickle.dump((num_imputer, cat_imputer, label_encoders, standard_scaler, robust_scaler, scaler), f)
    print("Model and preprocessing objects saved")
    
    # Process test data
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-10-silan/test_cleaned.csv', is_training=False)
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    X_test, _, _, _, _, _ = preprocess_features(X_test, is_training=False, 
                                                num_imputer=num_imputer, cat_imputer=cat_imputer, 
                                                label_encoders=label_encoders, standard_scaler=standard_scaler, 
                                                robust_scaler=robust_scaler)
    
    # Make predictions
    test_predictions = final_stacking_model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    submission.to_csv('submission_stacking_model.csv', index=False)
    print("Prediction completed. Submission saved to 'submission_stacking_model.csv'")
    print("\nPrediction Statistics:")
    print(f"Min: {test_predictions.min()}")
    print(f"Max: {test_predictions.max()}")
    print(f"Mean: {test_predictions.mean()}")
    print(f"Median: {np.median(test_predictions)}")
    
    # Feature importance (using LightGBM as an example)
    lgb_model = optimized_models['lgb']
    importance = lgb_model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X_full.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features (based on LightGBM):")
    print(feature_importance.head(10))

if __name__ == '__main__':
    main()