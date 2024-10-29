import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from category_encoders import TargetEncoder
import pickle
import time
import logging
from sklearn.metrics.pairwise import euclidean_distances

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
                       encoding_smoothing=1.0, is_train=True):
    X = X.copy()
    X['make'] = X['make'].astype('object')
    X['model'] = X['model'].astype('object')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    # Numeric Imputation
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
        logging.info("Numeric features imputed with median strategy.")
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
        logging.info("Numeric features imputed using existing imputer.")
    
    # Standardization
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)
            logging.info(f"Standardized columns: {columns_to_standardize}")
        else:
            X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)
            logging.info(f"Standardized columns using existing scaler: {columns_to_standardize}")
    
    # Categorical Imputation
    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
            logging.info("Categorical features imputed with 'unknown'.")
        else:
            X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
            logging.info("Categorical features imputed using existing imputer.")
        
        # Target Encoding
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        
        if target_encode_features:
            if target_encoder is None and is_train:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = pd.DataFrame(target_encoder.fit_transform(X[target_encode_features], y), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
                logging.info(f"Applied Target Encoding to: {target_encode_features}")
            elif target_encoder is not None and not is_train:
                X[target_encode_features] = pd.DataFrame(target_encoder.transform(X[target_encode_features]), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
                logging.info(f"Applied Target Encoding using existing encoder to: {target_encode_features}")
            elif target_encoder is None and not is_train:
                raise ValueError("Target Encoder is not fitted. Please fit it on training data.")
        
        # One-Hot Encoding for other categorical features
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            if is_train:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(X[other_categorical])
                encoded_feature_names = encoder.get_feature_names_out(other_categorical)
                encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
                X = pd.concat([X, encoded_df], axis=1)
                X = X.drop(columns=other_categorical)
                logging.info(f"Applied One-Hot Encoding to: {other_categorical}")
                encoder_fitted = encoder
                return X, num_imputer, cat_imputer, target_encoder, scaler, encoder_fitted
            else:
                if 'encoder_fitted' not in locals():
                    raise ValueError("OneHotEncoder not fitted. Please fit it on training data.")
                encoded_features = encoder_fitted.transform(X[other_categorical])
                encoded_feature_names = encoder_fitted.get_feature_names_out(other_categorical)
                encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
                X = pd.concat([X, encoded_df], axis=1)
                X = X.drop(columns=other_categorical)
                logging.info(f"Applied One-Hot Encoding using existing encoder to: {other_categorical}")
    return X, num_imputer, cat_imputer, target_encoder, scaler, None

def create_advanced_clusters(X, y, max_clusters=10, random_state=42):
    clustering_features = ['depreciation', 'coe', 'dereg_value', 'engine_cap', 
                          'curb_weight', 'power', 'vehicle_age', 
                          'no_of_owners', 'omv', 'arf']
    
    # Ensure all clustering features are in X
    clustering_features = [feature for feature in clustering_features if feature in X.columns]
    
    cluster_data = X[clustering_features].copy()
    
    # Log-transform skewed features
    for feature in ['depreciation', 'coe', 'dereg_value', 'omv', 'arf']:
        if feature in cluster_data.columns:
            cluster_data[feature] = np.log1p(cluster_data[feature])
            logging.info(f"Applied log1p transformation to {feature}.")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    bic_scores = []
    silhouette_scores = []
    n_components_range = range(2, max_clusters + 1)
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
        gmm.fit(scaled_data)
        bic = gmm.bic(scaled_data)
        bic_scores.append(bic)
        
        cluster_labels = gmm.predict(scaled_data)
        sil_score = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(sil_score)
        
        logging.info(f"For n_components = {n_components}, BIC: {bic}, Silhouette Score: {sil_score}")
    
    # Determine optimal number of clusters
    optimal_clusters_bic = n_components_range[np.argmin(bic_scores)]
    optimal_clusters_sil = n_components_range[np.argmax(silhouette_scores)]
    
    logging.info(f"Optimal number of clusters based on BIC: {optimal_clusters_bic}")
    logging.info(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters_sil}")
    
    # Decide on final number of clusters
    # Strategy: Choose the number of clusters with the lowest BIC among those with a reasonable silhouette score
    # Example threshold: silhouette score > 0.2
    reasonable_clusters = [n for n, s in zip(n_components_range, silhouette_scores) if s > 0.2]
    if reasonable_clusters:
        # Select the cluster with the minimum BIC among reasonable clusters
        bic_scores_reasonable = [bic for n, bic in zip(n_components_range, bic_scores) if n in reasonable_clusters]
        optimal_clusters = reasonable_clusters[np.argmin(bic_scores_reasonable)]
    else:
        optimal_clusters = optimal_clusters_bic
        logging.warning("No clusters with silhouette score > 0.2 found. Using BIC optimal number of clusters.")
    
    logging.info(f"Final number of clusters selected: {optimal_clusters}")
    
    # Final GMM with selected number of clusters
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

def evaluate_cluster_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        # early_stopping_rounds=50,
        # verbose_eval=100
    )
    
    train_predictions = model.predict(X_train, num_iteration=model.best_iteration)
    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    logging.info(f"Train RMSE (Log Space): {train_rmse:.4f}")
    logging.info(f"Validation RMSE (Log Space): {val_rmse:.4f}")
    
    return model

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def tune_lightgbm(X_train, y_train):
    param_grid = {
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'min_data_in_leaf': [20, 50, 100],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'bagging_freq': [0, 5, 10],
        'lambda_l1': [0, 0.1, 1],
        'lambda_l2': [0, 0.1, 1]
    }

    lgbm = lgb.LGBMRegressor(objective='regression', random_state=42)

    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best RMSE: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def merge_clusters(cluster_labels, gmm_model, scaler, clustering_features, high_rmse_clusters, X, y):
    """
    Merge clusters with high RMSE into the nearest cluster based on feature similarity.
    """
    # Extract cluster centers
    cluster_centers = gmm_model.means_
    
    # Compute pairwise distances between clusters
    distances = euclidean_distances(cluster_centers, cluster_centers)
    
    # Identify which clusters have already been marked for merging to prevent repeated merging
    already_merged = set()
    
    # Initialize a merge map to keep track of which clusters to merge
    merge_map = {}
    
    for cluster in high_rmse_clusters:
        if cluster in already_merged:
            continue  # Skip if already merged
        
        # Set the distance to itself as infinity to avoid self-merging
        distances[cluster, cluster] = np.inf
        nearest_cluster = np.argmin(distances[cluster])
        
        # If nearest_cluster is also in high_rmse_clusters, skip to avoid mutual merging
        if nearest_cluster in high_rmse_clusters:
            logging.warning(f"Nearest cluster {nearest_cluster} for Cluster {cluster} is also a high RMSE cluster. Skipping merge to avoid mutual merging.")
            continue
        
        merge_map[cluster] = nearest_cluster
        already_merged.add(cluster)
        logging.info(f"Merging Cluster {cluster} into Cluster {nearest_cluster}")
    
    # Update cluster labels based on the merge map
    new_cluster_labels = cluster_labels.copy()
    for cluster, merge_with in merge_map.items():
        new_cluster_labels[cluster_labels == cluster] = merge_with
    
    return new_cluster_labels

def main():
    np.random.seed(42)
    
    # Load training data
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    if 'make' not in X.columns or 'model' not in X.columns:
        logging.error("Error: 'make' or 'model' column not found in training data")
        return
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    # Log-transform the target variable
    y = np.log1p(y)
    
    # Initial Clustering
    gmm_model, cluster_labels, cluster_info, cluster_scaler, clustering_features = create_advanced_clusters(X, y)
    
    # Log cluster information
    for cluster in range(len(cluster_info)):
        cluster_samples = np.sum(cluster_labels == cluster)
        logging.info(f"Cluster {cluster} has {cluster_samples} samples")
    
    # Initialize variables for potential mergers
    merge_attempts = 0
    max_merge_attempts = 3
    rmse_threshold = 1.0  # RMSE threshold in log space
    improved = False
    
    while merge_attempts < max_merge_attempts:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        oof_predictions = np.full(len(X), np.nan)
        feature_importance_list = []
        models = []
        
        start_time = time.time()
        
        # Train models per cluster
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            X_cluster = X[cluster_labels == cluster]
            y_cluster = y[cluster_labels == cluster]
            
            if len(X_cluster) == 0:
                logging.warning(f"Cluster {cluster} has no samples. Skipping.")
                continue
            
            cluster_models = []
            
            for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
                logging.info(f"Fold {fold}")
                
                X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
                y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
                
                logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
                logging.info(f"Number of NaN in training features: {X_train.isna().sum().sum()}")
                logging.info(f"Number of NaN in training target: {y_train.isna().sum()}")
                
                # Preprocess training data
                X_train_processed, num_imputer, cat_imputer, target_encoder, scaler, encoder_fitted = preprocess_features(
                    X_train, y_train, is_train=True
                )
                # Preprocess validation data using fitted preprocessors
                X_val_processed, _, _, _, _, _ = preprocess_features(
                    X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler, is_train=False
                )
                
                model = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val)
                
                fold_predictions = model.predict(X_val_processed, num_iteration=model.best_iteration)
                logging.info(f"Number of NaN in fold predictions: {np.isnan(fold_predictions).sum()}")
                
                # Predictions are in log1p space
                oof_predictions[cluster_labels == cluster][val_index] = fold_predictions
                
                valid_fold_predictions = ~np.isnan(fold_predictions)
                logging.info(f"Fold {fold} has {np.sum(valid_fold_predictions)} valid predictions out of {len(fold_predictions)}")
                
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
        
        # Post-process predictions
        # Ensure that all OOF_predictions are filled
        if np.any(np.isnan(oof_predictions)):
            logging.warning("Some OOF predictions are missing. These will be excluded from metrics.")
        
        total_valid_predictions = np.sum(~np.isnan(oof_predictions))
        logging.info(f"Total valid OOF predictions: {total_valid_predictions} out of {len(oof_predictions)}")
        logging.info(f"Total number of NaN in OOF predictions: {np.isnan(oof_predictions).sum()}")
        logging.info(f"Percentage of NaN in OOF predictions: {100 * np.isnan(oof_predictions).sum() / len(oof_predictions):.2f}%")
        
        valid_mask = ~np.isnan(oof_predictions)
        y_true = y[valid_mask]
        y_pred = oof_predictions[valid_mask]
        
        if len(y_true) > 0:
            oof_rmse = evaluate_cluster_rmse(y_true, y_pred)
            oof_r2 = r2_score(y_true, y_pred)
            logging.info(f"Out-of-fold RMSE (Log Space): {oof_rmse:.4f}")
            logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
        else:
            logging.warning("No valid OOF predictions. Unable to calculate metrics.")
        
        # Feature importance
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
        logging.info("\nTop 10 important features:")
        logging.info(feature_importance.head(10))
        
        # Identify clusters with high RMSE
        high_rmse_clusters = []
        for cluster in range(len(cluster_info)):
            cluster_indices = (cluster_labels == cluster)
            cluster_pred = oof_predictions[cluster_indices]
            cluster_true = y[cluster_indices]
            
            # Remove NaN values
            valid_indices = ~np.isnan(cluster_pred)
            cluster_pred = cluster_pred[valid_indices]
            cluster_true = cluster_true[valid_indices]
            
            if len(cluster_true) == 0:
                logging.warning(f"Cluster {cluster} has no valid predictions for RMSE calculation.")
                continue
            
            try:
                cluster_rmse = evaluate_cluster_rmse(cluster_true, cluster_pred)
                logging.info(f"Cluster {cluster} RMSE (Log Space): {cluster_rmse:.4f}")
                if cluster_rmse > rmse_threshold:
                    high_rmse_clusters.append(cluster)
            except Exception as e:
                logging.error(f"Error calculating RMSE for cluster {cluster}: {str(e)}")
                logging.error(f"Number of valid predictions: {len(cluster_pred)}")
                logging.error(f"Number of true values: {len(cluster_true)}")
                continue
        
        if not high_rmse_clusters:
            logging.info("All clusters have RMSE below the threshold.")
            break  # Exit the loop as no clusters need merging
        
        if merge_attempts >= max_merge_attempts - 1:
            logging.warning("Maximum merge attempts reached. Proceeding without further merging.")
            break
        
        # Merge high RMSE clusters
        cluster_labels = merge_clusters(
            cluster_labels, gmm_model, cluster_scaler, clustering_features, high_rmse_clusters, X, y
        )
        
        # Update cluster_info after merging
        cluster_info = []
        for cluster in range(gmm_model.n_components):
            cluster_prices = y[cluster_labels == cluster]
            cluster_info.append({
                'cluster': cluster,
                'min': cluster_prices.min(),
                'max': cluster_prices.max(),
                'median': cluster_prices.median(),
                'count': len(cluster_prices)
            })
        cluster_df = pd.DataFrame(cluster_info)
        logging.info("Updated Cluster Information after merging:")
        logging.info(cluster_df)
        
        # Increment merge attempts
        merge_attempts += 1
        logging.info(f"Merge attempts: {merge_attempts}/{max_merge_attempts}")
    
    # Check for any potential issues in the input data
    logging.info(f"Number of NaN values in X: {X.isna().sum().sum()}")
    logging.info(f"Number of NaN values in y: {y.isna().sum()}")
    
    # Log basic statistics of the target variable
    logging.info(f"Target variable statistics:")
    logging.info(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}, Median: {y.median()}")
    
    # Save models and preprocessors
    with open('gmm_lightgbm_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'gmm_model': gmm_model,
            'cluster_info': cluster_info,
            'cluster_scaler': cluster_scaler,
            'clustering_features': clustering_features
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # ==============================
    # Prediction Phase
    # ==============================
    
    # Load test data
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    if 'make' not in X_test.columns or 'model' not in X_test.columns:
        logging.error("Error: 'make' or 'model' column not found in test data")
        return
    
    # Assign clusters to test data
    cluster_data_test = X_test[clustering_features].copy()
    
    # Apply log transformation to the same features as training
    for feature in ['depreciation', 'coe', 'dereg_value', 'omv', 'arf']:
        if feature in cluster_data_test.columns:
            cluster_data_test[feature] = np.log1p(cluster_data_test[feature])
            logging.info(f"Applied log1p transformation to {feature} in test data.")
    
    # Scale the test clustering features
    scaled_test_data = cluster_scaler.transform(cluster_data_test)
    test_clusters = gmm_model.predict(scaled_test_data)
    
    final_predictions_log = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\nPredicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_predictions = []
        
        for i, model_dict in enumerate(models[cluster]):
            model = model_dict['model']
            preprocessors = model_dict['preprocessors']
            
            try:
                # Apply preprocessing using stored preprocessors
                X_test_processed_cluster, _, _, _, scaler_cluster, _ = preprocess_features(
                    X_test_cluster, 
                    y=None, 
                    num_imputer=preprocessors['num_imputer'], 
                    cat_imputer=preprocessors['cat_imputer'], 
                    target_encoder=preprocessors['target_encoder'], 
                    scaler=preprocessors['scaler'],
                    is_train=False
                )
                preds = model.predict(X_test_processed_cluster, num_iteration=model.best_iteration)
                # Predictions are in log1p space
                cluster_predictions.append(preds)
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster}, model {i}: {str(e)}")
                logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                continue
        
        if cluster_predictions:
            cluster_predictions = np.array(cluster_predictions)
            final_predictions_log[cluster_mask] = cluster_predictions.mean(axis=0)
    
    # Post-process final predictions in log space
    final_predictions_log = post_process_predictions(final_predictions_log, 
                                                      min_price=np.log1p(700), 
                                                      max_price=np.log1p(2900000))
    
    # Transform predictions back to original space for submission
    final_predictions = np.expm1(final_predictions_log)
    
    # Post-process predictions in original space
    final_predictions = post_process_predictions(final_predictions)
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./submission_gmm_lightgbm_clustered_optimized_new.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'submission_gmm_lightgbm_clustered_optimized_new.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()