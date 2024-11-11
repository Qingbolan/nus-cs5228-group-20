import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RandomForestGini(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _calculate_gini(self, y):
        if len(y) == 0:
            return 0
        sorted_y = np.sort(y)
        n = len(y)
        cumsum = np.cumsum(sorted_y)
        gini = (n + 1 - 2 * np.sum(((n - np.arange(n)) * sorted_y) / cumsum[-1])) / n
        return max(0, min(1, gini))
        
    def _gini_criterion(self, y, y_pred):
        residuals = y - y_pred
        return self._calculate_gini(residuals)

class EnhancedWeightedEnsembleL2:
    def __init__(self, alpha=0.1, beta=0.05):
        self.weights = None
        self.alpha = alpha
        self.beta = beta
        
    def objective(self, weights, predictions, y_true):
        pred = predictions @ weights
        mse = np.mean((y_true - pred) ** 2)
        l2_reg = self.alpha * np.sum(weights ** 2)
        entropy_reg = -self.beta * np.sum(weights * np.log(weights + 1e-10))
        return mse + l2_reg + entropy_reg
        
    def fit(self, predictions, y_true):
        n_models = predictions.shape[1]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        initial_weights = np.ones(n_models) / n_models
        result = minimize(
            self.objective,
            initial_weights,
            args=(predictions, y_true),
            constraints=constraints,
            method='SLSQP'
        )
        
        self.weights = result.x
        return self
        
    def predict(self, predictions):
        return predictions @ self.weights

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
                       target_encode_cols=[], 
                       encoding_smoothing=1.0):
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
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

    return X, num_imputer, cat_imputer, target_encoder, scaler

def create_price_clusters(X, y, n_clusters, features_for_clustering):
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(X[features_for_clustering], np.linspace(0, 100, n_clusters), axis=0)
    ])

    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=10, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])
    price_clusters = kmeans.fit_predict(cluster_features)
    
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    return kmeans, price_clusters, cluster_df

def train_evaluate_models(X_train, y_train, X_val, y_val, params):
    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    model_lgb = lgb.train(params['lgb'], train_data, num_boost_round=1000,
                         valid_sets=[train_data, val_data], valid_names=['train', 'valid'])
    
    # Train RandomForestGini
    model_rf = RandomForestGini(**params['rf'])
    model_rf.fit(X_train, np.log1p(y_train))
    
    # Train GradientBoosting
    model_gb = GradientBoostingRegressor(**params['gb'])
    model_gb.fit(X_train, np.log1p(y_train))
    
    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train))
    dval = xgb.DMatrix(X_val, label=np.log1p(y_val))
    model_xgb = xgb.train(params['xgb'], dtrain, num_boost_round=1000,
                         evals=[(dtrain, 'train'), (dval, 'valid')], verbose_eval=False)
    
    # Train CatBoost
    model_cb = CatBoostRegressor(**params['cb'])
    model_cb.fit(X_train, np.log1p(y_train), eval_set=(X_val, np.log1p(y_val)), verbose=False)
    
    return model_lgb, model_rf, model_gb, model_xgb, model_cb

def get_model_predictions(models, X):
    model_lgb, model_rf, model_gb, model_xgb, model_cb = models
    
    preds_lgb = np.expm1(model_lgb.predict(X))
    preds_rf = np.expm1(model_rf.predict(X))
    preds_gb = np.expm1(model_gb.predict(X))
    preds_xgb = np.expm1(model_xgb.predict(xgb.DMatrix(X)))
    preds_cb = np.expm1(model_cb.predict(X))
    
    return np.column_stack([preds_lgb, preds_rf, preds_gb, preds_xgb, preds_cb])

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('l3_train.csv')
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # Create clusters
    n_clusters = 3
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters, features_for_clustering)
    
    # Define model parameters
    model_params = {
        'lgb': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_child_samples': 20,
            'cat_smooth': 10,
            'cat_l2': 10,
        },
        'rf': {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42
        },
        'xgb': {
            'eta': 0.03,
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42
        },
        'gb': {
            'n_estimators': 3000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'subsample': 0.8,
            'loss': 'huber',
            'random_state': 42
        },
        'cb': {
            'iterations': 3000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': 42,
            'verbose': False
        }
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    models_by_cluster = []
    
    start_time = time.time()
    
    for cluster in range(n_clusters):
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster), 1):
            X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
            y_train, y_val = y_cluster.iloc[train_idx], y_cluster.iloc[val_idx]
            
            # Preprocess data
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(
                X_train, y_train
            )
            X_val_processed, _, _, _, _ = preprocess_features(
                X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler
            )
            
            # Train models
            models = train_evaluate_models(X_train_processed, y_train, X_val_processed, y_val, model_params)
            
            # Get predictions
            val_predictions = get_model_predictions(models, X_val_processed)
            
            # Fit ensemble
            ensemble = EnhancedWeightedEnsembleL2(alpha=0.1, beta=0.05)
            ensemble.fit(val_predictions, y_val)
            
            # Save ensemble predictions
            oof_predictions[val_idx] = ensemble.predict(val_predictions)
            
            # Store models and preprocessors
            cluster_models.append({
                'models': models,
                'ensemble': ensemble,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
            
            # Log metrics
            rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
            r2 = r2_score(y_val, oof_predictions[val_idx])
            logging.info(f"Cluster {cluster} - Fold {fold} RMSE: {rmse:.4f}, R2: {r2:.4f}")
            logging.info(f"Ensemble weights: {ensemble.weights}")
        
        models_by_cluster.append(cluster_models)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    # Process and evaluate out-of-fold predictions
    oof_predictions = post_process_predictions(oof_predictions)
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    oof_r2 = r2_score(y, oof_predictions)
    logging.info(f"Overall Out-of-fold RMSE: {oof_rmse:.4f}")
    logging.info(f"Overall Out-of-fold R2: {oof_r2:.4f}")
    
    # Save models
    with open('ensemble_clustered_models_enhanced_7.pkl', 'wb') as f:
        pickle.dump({
            'models': models_by_cluster,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    
    # Make predictions on test data
    X_test, _ = load_and_preprocess_data('l3_test.csv')
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = kmeans_model.predict(np.column_stack([
        dummy_y_test,
        X_test[features_for_clustering]
    ]))
    
    final_predictions = np.zeros(len(X_test))
    prediction_counts = np.zeros(len(X_test))
    
    for cluster in range(n_clusters):
        cluster_mask = test_clusters == cluster
        if not np.any(cluster_mask):
            continue
            
        X_test_cluster = X_test[cluster_mask]
        cluster_predictions = np.zeros(len(X_test_cluster))
        count = 0
        
        for model_dict in models_by_cluster[cluster]:
            try:
                X_test_processed, _, _, _, _ = preprocess_features(
                    X_test_cluster,
                    y=None,
                    **model_dict['preprocessors']
                )
                
                predictions = get_model_predictions(model_dict['models'], X_test_processed)
                cluster_predictions += model_dict['ensemble'].predict(predictions)
                count += 1
                
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}: {str(e)}")
                continue
        
        if count > 0:
            final_predictions[cluster_mask] = cluster_predictions / count
            prediction_counts[cluster_mask] = count
    
    final_predictions = post_process_predictions(final_predictions)
    
    # Save predictions
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    submission.to_csv('submission_enhanced_ensemble_7.csv', index=False)
    
    # Log prediction statistics
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()