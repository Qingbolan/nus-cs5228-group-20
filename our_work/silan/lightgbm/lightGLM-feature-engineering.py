import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle
import time

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    return X, y

def preprocess_features(X, num_imputer=None, cat_imputer=None, label_encoders=None):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Handle numeric features
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    else:
        X[numeric_features] = num_imputer.transform(X[numeric_features])
    
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
    
    return X, num_imputer, cat_imputer, label_encoders

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

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-10-silan/train_cleaned.csv')
    X, num_imputer, cat_imputer, label_encoders = preprocess_features(X)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'cat_smooth': 10,
        'cat_l2': 10,
    }
    
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
    model.save_model('./lightgbm_model_handle_new_categories.txt')
    with open('preprocessing_objects_handle_new_categories.pkl', 'wb') as f:
        pickle.dump((num_imputer, cat_imputer, label_encoders), f)
    print("Model and preprocessing objects saved")
    
    # Process test data
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-10-silan/test_cleaned.csv')
    
    # Apply preprocessing to test data, handling new categories
    X_test, _, _, _ = preprocess_features(X_test, num_imputer, cat_imputer, label_encoders)
    
    test_predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    submission = pd.DataFrame({
        'Id': range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    
    submission.to_csv('./submission_lightgbm_handle_new_categories.csv', index=False)
    print("Prediction completed. Submission saved to 'submission_lightgbm_handle_new_categories.csv'")
    
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