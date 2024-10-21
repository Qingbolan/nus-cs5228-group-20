import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import time
from category_encoders import TargetEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    print(f"Columns in {file_path}:", X.columns)
    
    return X, y

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                        target_encoder=None, scaler=None, 
                        target_encode_cols=['make', 'model'], 
                        encoding_smoothing=1.0):
    X = X.copy()  # Create a copy to avoid modifying the original dataframe
    X['make'] = X['make'].astype('object')
    X['model'] = X['model'].astype('object')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)
        else:
            X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)

    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        else:
            X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        
        if target_encode_features:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = pd.DataFrame(target_encoder.fit_transform(X[target_encode_features], y), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
            else:
                X[target_encode_features] = pd.DataFrame(target_encoder.transform(X[target_encode_features]), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
        
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)
    else:
        print("No categorical features found.")

    return X, num_imputer, cat_imputer, target_encoder, scaler

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=50,
        # verbose_eval=100
    )
    
    return model

def post_process_predictions(predictions, min_price=0, max_price=2000000):
    """
    Post-process predictions to ensure they are within a reasonable range.
    """
    predictions = np.clip(predictions, min_price, max_price)
    return predictions

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    if 'make' not in X.columns or 'model' not in X.columns:
        print("Error: 'make' or 'model' column not found in training data")
        return
    
    X['make'] = X['make'].astype('object')
    X['model'] = X['model'].astype('object')
    
    # Print some statistics about the target variable
    print("Target variable (price) statistics:")
    print(y.describe())
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_list = []
    models = []
    
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
    
    start_time = time.time()
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}")
        
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
        X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
        
        model = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, params)
        
        fold_predictions = model.predict(X_val_processed, num_iteration=model.best_iteration)
        oof_predictions[val_index] = post_process_predictions(fold_predictions)
        
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
        feature_importance_list.append(feature_importance)
        
        models.append((model, num_imputer, cat_imputer, target_encoder, scaler))
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    oof_mse = mean_squared_error(y, oof_predictions)
    oof_r2 = r2_score(y, oof_predictions)
    print(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
    print(f"Out-of-fold R2: {oof_r2:.4f}")
    
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    with open('lightgbm_models_and_preprocessors.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("Models and preprocessors saved.")
    
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    if 'make' not in X_test.columns or 'model' not in X_test.columns:
        print("Error: 'make' or 'model' column not found in test data")
        return
    
    X_test['make'] = X_test['make'].astype('object')
    X_test['model'] = X_test['model'].astype('object')
    
    test_predictions = np.zeros((len(X_test), len(models)))
    
    for i, (model, num_imputer, cat_imputer, target_encoder, scaler) in enumerate(models):
        X_test_processed, _, _, _, _ = preprocess_features(X_test, y=None, 
                                                           num_imputer=num_imputer, 
                                                           cat_imputer=cat_imputer, 
                                                           target_encoder=target_encoder, 
                                                           scaler=scaler)
        test_predictions[:, i] = model.predict(X_test_processed, num_iteration=model.best_iteration)
    
    final_predictions = np.mean(test_predictions, axis=1)
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./submission_lightgbm_5fold.csv', index=False)
    print("Predictions complete. Submission file saved as 'submission_lightgbm_5fold.csv'.")
    
    print("\nPrediction statistics:")
    print(f"Minimum: {final_predictions.min()}")
    print(f"Maximum: {final_predictions.max()}")
    print(f"Mean: {final_predictions.mean()}")
    print(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()