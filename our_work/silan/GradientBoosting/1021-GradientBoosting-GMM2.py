import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from category_encoders import TargetEncoder
import pickle
import time
import logging

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
    X['make'] = X['make'].astype('object')
    X['model'] = X['model'].astype('object')
    
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

def create_advanced_clusters(X, y, max_clusters=10, random_state=42):
    clustering_features = ['depreciation', 'coe', 'dereg_value', 'engine_cap', 'curb_weight', 'power', 'vehicle_age', 'no_of_owners', 'omv', 'arf']
    
    # Ensure all clustering features are present in X
    clustering_features = [feature for feature in clustering_features if feature in X.columns]
    
    cluster_data = X[clustering_features].copy()
    
    # Log transform for price and some other features
    cluster_data['price'] = np.log1p(y)
    for feature in ['depreciation', 'coe', 'dereg_value', 'omv', 'arf']:
        if feature in cluster_data.columns:
            cluster_data[feature] = np.log1p(cluster_data[feature])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    bic_scores = []
    silhouette_scores = []
    n_components_range = range(2, max_clusters + 1)
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
        labels = gmm.fit_predict(scaled_data)
        
        bic = gmm.bic(scaled_data)
        silhouette = silhouette_score(scaled_data, labels)
        
        bic_scores.append(bic)
        silhouette_scores.append(silhouette)
        
        logging.info(f"For n_components = {n_components}, BIC = {bic:.2f}, Silhouette = {silhouette:.4f}")
    
    # Normalize scores
    bic_scores = np.array(bic_scores)
    silhouette_scores = np.array(silhouette_scores)
    normalized_bic = (bic_scores - bic_scores.min()) / (bic_scores.max() - bic_scores.min())
    normalized_silhouette = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())
    
    # Combine scores (lower BIC and higher silhouette are better)
    combined_scores = normalized_bic - normalized_silhouette
    
    optimal_clusters = combined_scores.argmin() + 2
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    
    final_gmm = GaussianMixture(n_components=optimal_clusters, random_state=random_state, n_init=10)
    cluster_labels = final_gmm.fit_predict(scaled_data)
    
    cluster_info = []
    for cluster in range(optimal_clusters):
        cluster_prices = y[cluster_labels == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Cluster Information:")
    logging.info(cluster_df)
    
    return final_gmm, cluster_labels, cluster_df, scaler, clustering_features

def predict_advanced_cluster(X, y, gmm_model, scaler, clustering_features):
    cluster_data = X[clustering_features].copy()
    cluster_data['price'] = y if y is not None else np.zeros(len(X))
    
    # Log transform for some features
    for feature in ['depreciation', 'coe', 'dereg_value', 'omv', 'arf', 'price']:
        if feature in cluster_data.columns:
            cluster_data[feature] = np.log1p(cluster_data[feature])
    
    scaled_data = scaler.transform(cluster_data)
    return gmm_model.predict(scaled_data)

from sklearn.model_selection import train_test_split

def train_evaluate_gradient_boosting(X_train, y_train, X_val, y_val):
    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.1,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=20,
        min_samples_split=15,
        loss='huber',
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_es, np.log1p(y_train_es))
    
    # Manual early stopping
    val_scores = []
    for i in range(1, model.n_estimators + 1):
        val_pred = model.predict(X_val_es, n_iter=i)
        val_score = mean_squared_error(y_val_es, np.expm1(val_pred))
        val_scores.append(val_score)
    
    best_iteration = np.argmin(val_scores) + 1
    
    train_predictions = np.expm1(model.predict(X_train, n_iter=best_iteration))
    val_predictions = np.expm1(model.predict(X_val, n_iter=best_iteration))
    
    train_mse = mean_squared_error(y_train, train_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    
    logging.info(f"Train RMSE: {np.sqrt(train_mse):.4f}")
    logging.info(f"Validation RMSE: {np.sqrt(val_mse):.4f}")
    logging.info(f"Best iteration: {best_iteration}")
    
    return model, best_iteration

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    if 'make' not in X.columns or 'model' not in X.columns:
        logging.error("Error: 'make' or 'model' column not found in training data")
        return
    
    X['make'] = X['make'].astype('object')
    X['model'] = X['model'].astype('object')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    gmm_model, price_clusters, cluster_info, cluster_scaler, clustering_features = create_advanced_clusters(X, y)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_list = []
    models = []
    
    start_time = time.time()
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\nTraining models for Cluster {cluster}")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            model = train_evaluate_gradient_boosting(X_train_processed, y_train, X_val_processed, y_val)
            
            fold_predictions = np.expm1(model.predict(X_val_processed))
            oof_predictions[price_clusters == cluster][val_index] = fold_predictions
            
            importance = model.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            cluster_models.append({
                'model': model,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
        
        models.append(cluster_models)
    
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
    
    with open('gmm_lightgbm_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'gmm_model': gmm_model,
            'cluster_info': cluster_info,
            'cluster_scaler': cluster_scaler,
            'clustering_features': clustering_features
        }, f)
    logging.info("Models and preprocessors saved.")
    
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    if 'make' not in X_test.columns or 'model' not in X_test.columns:
        logging.error("Error: 'make' or 'model' column not found in test data")
        return
    
    X_test['make'] = X_test['make'].astype('object')
    X_test['model'] = X_test['model'].astype('object')
    
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = predict_advanced_cluster(X_test, dummy_y_test, gmm_model, cluster_scaler, clustering_features)
    
    final_predictions = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_predictions = np.zeros((len(X_test_cluster), len(models[cluster])))
        
        for i, model_dict in enumerate(models[cluster]):
            model = model_dict['model']
            preprocessors = model_dict['preprocessors']
            
            try:
                X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **preprocessors)
                cluster_predictions[:, i] = np.expm1(model.predict(X_test_processed))
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}, model {i}: {str(e)}")
                logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                continue
        
        final_predictions[cluster_mask] = np.mean(cluster_predictions, axis=1)
    
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./submission_gmm_lightgbm_clustered_optimized.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'submission_gmm_lightgbm_clustered_optimized_new.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()