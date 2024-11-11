import pandas as pd
# this scipt is aming to make two ref column for data
# one for ref model prize,
# one for make prize

# all the prize value is gain from train dataset, store in a csv query file
def add_ref_make_prize(data, istrain=True):
    if istrain:
        # Calculate average prize for each make
        make_prize_ref = data.groupby('make')['price'].mean().to_dict()
        
        # Calculate average prize for each model
        model_prize_ref = data.groupby('model')['price'].mean().to_dict()
        
        # Calculate average prize for each type_of_vehicle
        type_prize_ref = data.groupby('type_of_vehicle')['price'].mean().to_dict()
        
        # Save references for test data
        pd.Series(make_prize_ref).to_csv('make_price_ref.csv')
        pd.Series(model_prize_ref).to_csv('model_price_ref.csv')
        pd.Series(type_prize_ref).to_csv('type_price_ref.csv')
        
        # Add reference columns
        data['make_price_ref'] = data['make'].map(make_prize_ref)
        data['model_price_ref'] = data['model'].map(model_prize_ref)
        
    else:
        # Load pre-calculated references from training data
        make_prize_ref = pd.read_csv('make_price_ref.csv', index_col=0).squeeze().to_dict()
        model_prize_ref = pd.read_csv('model_price_ref.csv', index_col=0).squeeze().to_dict()
        type_prize_ref = pd.read_csv('type_price_ref.csv', index_col=0).squeeze().to_dict()
        
        # First, try to get model reference price
        data['model_price_ref'] = data['model'].map(model_prize_ref)
        
        # For models not in training set, use make reference price
        data['make_price_ref'] = data['make'].map(make_prize_ref)
        mask = data['model_price_ref'].isna()
        data.loc[mask, 'model_price_ref'] = data.loc[mask, 'make_price_ref']
        
        # For makes not in training set, use type_of_vehicle reference price
        type_means = data['type_of_vehicle'].map(type_prize_ref)
        mask = data['make_price_ref'].isna()
        data.loc[mask, 'make_price_ref'] = type_means[mask]
        data.loc[mask, 'model_price_ref'] = type_means[mask]
    
    return data