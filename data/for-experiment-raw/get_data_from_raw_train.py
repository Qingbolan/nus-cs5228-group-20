# this is a script to get data from raw train data under data/raw/train.csv
# and save it to two files: for_train.csv and for_test.csv
# for_train.csv is used to train the model
# for_test.csv is used to test the model is get 5000 random data from train.csv and save it to for_test.csv
# but the data in for_test.csv is not in for_train.csv
# the data in for_train.csv is not in for_test.csv
# the for_test.csv does not have the price column, the for_test_price.csv has two columns: id and price
# the for_test.csv has same verhicle type distribution with train.csv
import pandas as pd
import numpy as np

# Read data from raw train data
data = pd.read_csv('data/raw/train.csv')

# Calculate the distribution of vehicle types in the original dataset
vehicle_type_dist = data['type_of_vehicle'].value_counts(normalize=True)

# Calculate the number of samples for the test set (5000)
test_size = 5000

# Calculate the number of samples for each vehicle type in the test set
test_samples_per_type = (vehicle_type_dist * test_size).round().astype(int)

# Initialize empty DataFrames for test and train sets
test_data = pd.DataFrame()
train_data = data.copy()

# Sample data for the test set while maintaining the vehicle type distribution
for vehicle_type, sample_size in test_samples_per_type.items():
    type_data = data[data['type_of_vehicle'] == vehicle_type]
    sampled_data = type_data.sample(n=min(sample_size, len(type_data)), random_state=42)
    test_data = pd.concat([test_data, sampled_data])
    train_data = train_data.drop(sampled_data.index)

# Reset index for both datasets
test_data = test_data.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

# Create for_test.csv (without price column)
test_data_no_price = test_data.drop('price', axis=1)
test_data_no_price.to_csv('data/for-experiment-raw/for_test.csv', index=False)

# Create for_test_price.csv with id starting from 1
test_price_data = pd.DataFrame({
    'id': range(1, len(test_data) + 1),
    'price': test_data['price']
})
test_price_data.to_csv('data/for-experiment-raw/for_test_price.csv', index=False)

# Save train data to for_train.csv
train_data.to_csv('data/for-experiment-raw/for_train.csv', index=False)

print(f"Original data shape: {data.shape}")
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print("Files created: for_train.csv, for_test.csv, for_test_price.csv")

# Verify vehicle type distribution
original_dist = data['type_of_vehicle'].value_counts(normalize=True)
test_dist = test_data['type_of_vehicle'].value_counts(normalize=True)

print("\nVehicle Type Distribution:")
print("Original | Test")
for vehicle_type in original_dist.index:
    print(f"{vehicle_type}: {original_dist[vehicle_type]:.4f} | {test_dist.get(vehicle_type, 0):.4f}")

# Verify ID range in for_test_price.csv
print(f"\nID range in for_test_price.csv: 1 to {len(test_data)}")
print(f"Number of rows in for_test.csv: {len(test_data_no_price)}")
print(f"Number of rows in for_test_price.csv: {len(test_price_data)}")