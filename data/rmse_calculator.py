# support function to calculate RMSE

# by terminal, run:
# python rmse_calculator.py path/to/your/submission.csv

# by function, run:
# from data.rmse_calculator import calculate_rmse
# calculate_rmse('path/to/your/submission.csv')

import pandas as pd
import numpy as np
import argparse

def calculate_rmse(submission_path, true_prices_path='data/for-experiment-raw/for_test_price.csv'):
    """
    Calculate the Root Mean Square Error (RMSE) between predicted prices and true prices.
    
    Parameters:
    submission_path (str): Path to the submission CSV file with predicted prices.
    true_prices_path (str): Path to the CSV file with true prices (for_test_price.csv).
    
    Returns:
    float: The calculated RMSE value.
    """
    try:
        # Read the submission file and true prices file
        submission = pd.read_csv(submission_path)
        true_prices = pd.read_csv(true_prices_path)
        
        # Check for required columns
        if 'Id' not in submission.columns or 'Predicted' not in submission.columns:
            raise ValueError("Submission file must contain 'Id' and 'Predicted' columns.")
        if 'price' not in true_prices.columns:
            raise ValueError("True prices file must contain 'price' column.")
        
        # Ensure both DataFrames have the same number of rows
        if len(submission) != len(true_prices):
            raise ValueError("Submission file and true prices file must have the same number of rows.")
        
        # Calculate the RMSE directly
        rmse = np.sqrt(((true_prices['price'] - submission['Predicted']) ** 2).mean())
        
        return rmse
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSE for price predictions.')
    parser.add_argument('submission_path', type=str, help='Path to the submission CSV file')
    parser.add_argument('--true_prices_path', type=str, 
                        default='data/for-experiment-raw/for_test_price.csv',
                        help='Path to the true prices CSV file (default: data/for-experiment-raw/for_test_price.csv)')
    
    args = parser.parse_args()
    
    rmse_score = calculate_rmse(args.submission_path, args.true_prices_path)
    if rmse_score is not None:
        print(f"RMSE Score: {rmse_score:.4f}")

if __name__ == "__main__":
    main()