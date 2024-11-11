import pandas as pd

def add_ref_prices(data, istrain=True):
    """
    Add reference price columns for type_of_vehicle and model.
    Model is identified by combination of make and model.
    Order of operations ensures type_of_vehicle_price is added before model_price.
    
    Args:
        data (pd.DataFrame): Input dataframe
        istrain (bool): Whether this is training data
        
    Returns:
        pd.DataFrame: DataFrame with added reference columns
    """
    if istrain:
        # Calculate average prices for vehicle types
        type_price_ref = data.groupby('type_of_vehicle')['price'].mean().to_dict()
        
        # Calculate average prices for make-model combinations
        # Create a unique identifier for each make-model combination
        data['make_model'] = data['make'] + '_' + data['model']
        model_price_ref = data.groupby('make_model')['price'].mean().to_dict()
        
        # Save references for test data
        pd.Series(type_price_ref).to_csv('type_price_ref.csv')
        pd.Series(model_price_ref).to_csv('model_price_ref.csv')
        
        # Add reference columns in specific order
        data['ref_type_of_vehicle_price'] = data['type_of_vehicle'].map(type_price_ref)
        data['ref_model_price'] = data['make_model'].map(model_price_ref)
        
        # Drop the temporary column
        data = data.drop('make_model', axis=1)
        
    else:
        # Load pre-calculated references from training data
        type_price_ref = pd.read_csv('type_price_ref.csv', index_col=0).squeeze().to_dict()
        model_price_ref = pd.read_csv('model_price_ref.csv', index_col=0).squeeze().to_dict()
        
        # First add type_of_vehicle reference price
        data['ref_type_of_vehicle_price'] = data['type_of_vehicle'].map(type_price_ref)
        
        # Create make_model combination and map to reference prices
        data['make_model'] = data['make'] + '_' + data['model']
        data['ref_model_price'] = data['make_model'].map(model_price_ref)
        
        # For make-model combinations not in training set, use type_of_vehicle reference price
        mask = data['ref_model_price'].isna()
        data.loc[mask, 'ref_model_price'] = data.loc[mask, 'ref_type_of_vehicle_price']
        
        # Drop the temporary column
        data = data.drop('make_model', axis=1)
    
    return data