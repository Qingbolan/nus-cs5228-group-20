import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import time

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    return X, y

def preprocess_features(X, num_imputer=None, cat_imputer=None, label_encoders=None, scaler=None):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Specific columns to standardize
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    
    # Handle numeric features
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    else:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
    
    # Standardize specific numeric features
    if scaler is None:
        scaler = StandardScaler()
        X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
    else:
        X[columns_to_standardize] = scaler.transform(X[columns_to_standardize])
    
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
            # Handle new categories
            le = label_encoders[col]
            new_categories = set(X[col]) - set(le.classes_)
            if new_categories:
                print(f"New categories found in {col}: {new_categories}")
                new_label = len(le.classes_)
                le.classes_ = np.append(le.classes_, list(new_categories))
            X[col] = X[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else new_label)
    
    return X, num_imputer, cat_imputer, label_encoders, scaler

def train_evaluate_lightgbm(X, y, params, num_rounds=1000, early_stopping_rounds=50):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=early_stopping_rounds,
        # verbose_eval=100
    )
    
    return model

def optimize_params(X, y):
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10, 15, 20],
        'min_child_samples': [10, 20, 30],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'bagging_freq': [3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'cat_smooth': [1, 10, 20],
        'cat_l2': [1, 10, 20]
    }
    
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42
    }
    
    model = lgb.LGBMRegressor(**base_params)
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=100,  # number of parameter settings that are sampled
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1  # use all available cores
    )
    
    random_search.fit(X, y)
    
    print("Best parameters found: ", random_search.best_params_)
    print("Best score found: ", np.sqrt(-random_search.best_score_))
    
    return random_search.best_params_

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-10-silan/train_cleaned.csv')
    X, num_imputer, cat_imputer, label_encoders, scaler = preprocess_features(X)
    
    # Optimize parameters
    print("Optimizing parameters...")
    best_params = optimize_params(X, y)
    
    # Add base parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        **best_params
    }
    
    print("Final parameters:", params)
    
    # Cross-validation
    cv_scores = cross_val_score(lgb.LGBMRegressor(**params), X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std() * 2):.4f})")
    
    start_time = time.time()
    model = train_evaluate_lightgbm(X, y, params)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time/60:.2f} minutes")
    
    # Evaluate on training set
    train_predictions = model.predict(X, num_iteration=model.best_iteration)
    train_mse = mean_squared_error(y, train_predictions)
    train_r2 = r2_score(y, train_predictions)
    print(f"Training Set - RMSE: {np.sqrt(train_mse):.4f}, R2: {train_r2:.4f}")
    
    # Save model and preprocessing objects
    model.save_model('./lightgbm_model_optimized.txt')
    with open('preprocessing_objects_optimized.pkl', 'wb') as f:
        pickle.dump((num_imputer, cat_imputer, label_encoders, scaler), f)
    print("Model and preprocessing objects saved")
    
    # Process test data
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-10-silan/test_cleaned.csv')
    
    # Apply preprocessing to test data, handling new categories and standardizing
    X_test, _, _, _, _ = preprocess_features(X_test, num_imputer, cat_imputer, label_encoders, scaler)
    
    test_predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    submission = pd.DataFrame({
        'Id': range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    
    submission.to_csv('./submission_lightgbm_optimized.csv', index=False)
    print("Prediction completed. Submission saved to 'submission_lightgbm_optimized.csv'")
    
    print("\nPrediction Statistics:")
    print(f"Min: {test_predictions.min()}")
    print(f"Max: {test_predictions.max()}")
    print(f"Mean: {test_predictions.mean()}")
    print(f"Median: {np.median(test_predictions)}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == '__main__':
    main()