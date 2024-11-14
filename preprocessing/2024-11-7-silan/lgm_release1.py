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
import logging

from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv('train_data'))
print(os.getenv('test_data'))

# (test)RMSE Score: 21429.3845

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns}")
    
    return X, y

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                        target_encoder=None, scaler=None, 
                        target_encode_cols=['make', 'model'], 
                        encoding_smoothing=1.0):
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Handle numeric features
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    
    # Handle categorical features
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

    return X, num_imputer, cat_imputer, target_encoder, scaler

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
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

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data(os.getenv('train_data'))
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
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
        logging.info(f"\nTraining Fold {fold}")
        
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
        X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
        
        model = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, params)
        
        fold_predictions = np.expm1(model.predict(X_val_processed, num_iteration=model.best_iteration))
        oof_predictions[val_index] = fold_predictions
        
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
        feature_importance_list.append(feature_importance)
        
        models.append({
            'model': model,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        })
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse = mean_squared_error(y, oof_predictions)
    oof_r2 = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
    
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    with open('lightgbm_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models
        }, f)
    logging.info("Models and preprocessors saved.")
    
    X_test, _ = load_and_preprocess_data(os.getenv('test_data'))
    
    final_predictions = np.zeros(len(X_test))
    
    for i, model_dict in enumerate(models, 1):
        logging.info(f"Predicting with Fold {i}")
        model = model_dict['model']
        preprocessors = model_dict['preprocessors']
        
        try:
            X_test_processed, _, _, _, _ = preprocess_features(X_test, y=None, **preprocessors)
            predictions = np.expm1(model.predict(X_test_processed, num_iteration=model.best_iteration))
            final_predictions += predictions
        except Exception as e:
            logging.error(f"Error predicting with Fold {i}: {str(e)}")
            logging.error(f"Shape of X_test_processed: {X_test_processed.shape}")
            logging.error(f"Columns in X_test_processed: {X_test_processed.columns}")
            continue
    
    final_predictions /= len(models)
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv(os.getenv('output_prediction'), index=False)
    logging.info(f"Predictions complete. Submission file saved as '{os.getenv('output_prediction')}'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")
    
    # 处理训练集数据
    logging.info("\nProcessing training data...")
    train_data = pd.read_csv(os.getenv('train_data'))
    train_data['ref_price'] = oof_predictions  # 使用out-of-fold预测作为ref_price
    train_data.to_csv(os.getenv('ref_train'), index=False)
    logging.info(f"Training data with ref_price saved to {os.getenv('ref_train')}")
    
    # 处理测试集数据
    logging.info("\nProcessing test data...")
    test_data = pd.read_csv(os.getenv('test_data'))
    test_data['ref_price'] = final_predictions  # 使用模型预测作为ref_price
    test_data.to_csv(os.getenv('ref_test'), index=False)
    logging.info(f"Test data with ref_price saved to {os.getenv('ref_test')}")
    
    # 输出ref_price的统计信息
    logging.info("\nRef_price statistics for training data:")
    logging.info(train_data['ref_price'].describe())
    logging.info("\nRef_price statistics for test data:")
    logging.info(test_data['ref_price'].describe())

if __name__ == '__main__':
    main()
