# Car Price Prediction with Clustering

## Project Overview

This project aims to predict car prices using a combination of clustering and gradient boosting techniques. The approach involves segmenting the car market into distinct clusters based on key features and then training separate LightGBM models for each cluster. This method allows for more nuanced predictions that can capture the varying pricing dynamics across different segments of the car market.

## Key Features

1. **Dynamic Clustering**: Utilizes KMeans clustering with an optimized number of clusters determined by silhouette score analysis.
2. **Feature Engineering**: Implements various preprocessing techniques including target encoding, standard scaling, and one-hot encoding.
3. **Log-Transformed Target**: Uses log-transformation on the target variable (price) to handle skewed distributions.
4. **Cross-Validation**: Employs 5-fold cross-validation to ensure model stability and generalization.
5. **Cluster-Specific Models**: Trains separate LightGBM models for each identified market segment (cluster).
6. **Robust Prediction Pipeline**: Includes a comprehensive prediction pipeline that can handle new, unseen data.
7. **Post-Processing**: Implements post-processing steps to ensure predictions fall within reasonable bounds.

## Technical Details

### Data Preprocessing

- Handling of missing values using median imputation for numeric features and constant imputation for categorical features.
- Target encoding for high-cardinality categorical variables (e.g., 'make', 'model').
- Standard scaling for specific numeric features.
- One-hot encoding for remaining categorical features.

### Clustering

- Uses KMeans algorithm with KMeans++ initialization.
- Determines optimal number of clusters (2-10) using silhouette score.
- Clustering features: 'depreciation', 'coe', 'dereg_value', and log-transformed price.

### Model Training

- LightGBM regression models trained for each cluster.
- Log-transformation of the target variable (price) during training and inverse transformation during prediction.
- 5-fold cross-validation within each cluster.

### Prediction

- New data is assigned to clusters using the trained KMeans model.
- Predictions are made using the corresponding cluster-specific LightGBM models.
- Final predictions are post-processed to ensure they fall within predefined price bounds.

## Usage

1. Ensure all required libraries are installed:

   ```
   pip install pandas numpy scikit-learn lightgbm category_encoders
   ```
2. Prepare your data:

   - Training data should be in 'preprocessing/2024-10-21-silan/train_cleaned.csv'
   - Test data should be in 'preprocessing/2024-10-21-silan/test_cleaned.csv'
3. Run the script:

   ```
   python car_price_prediction.py
   ```
4. The script will output:

   - Logging information about the clustering and training process.
   - A pickle file 'lightgbm_clustered_models.pkl' containing the trained models and preprocessors.
   - A CSV file 'submission_lightgbm_clustered_optimized.csv' with the predictions for the test set.

## Future Improvements

1. Feature Importance Analysis: Conduct a more in-depth analysis of feature importance across clusters to gain insights into different market segments.
2. Hyperparameter Tuning: Implement automated hyperparameter tuning for LightGBM models, potentially customized for each cluster.
3. Alternative Clustering Methods: Explore other clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models) to compare performance.
4. Ensemble Methods: Investigate the potential of ensemble methods combining predictions from different clustering approaches.
5. Time Series Analysis: If temporal data is available, incorporate time series analysis to capture trends and seasonality in car prices.
6. External Data Integration: Explore the possibility of integrating external economic indicators or market data to improve predictions.

## Conclusion

This project demonstrates an advanced approach to car price prediction by combining market segmentation through clustering with gradient boosting regression. The method's strength lies in its ability to capture varied pricing dynamics across different segments of the car market, potentially leading to more accurate and nuanced predictions.
