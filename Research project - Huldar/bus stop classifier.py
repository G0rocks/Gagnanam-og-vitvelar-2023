# Author:           Huldar
# Date:             2023-10-07
# Project:          Research project
# Acknowledgements: Strætó for hooking us up with the data from https://data02.straeto.is/data/raw/
'''
This program gets labeled data about bus stops, trains a radial basis function neural network (RBFNN) on part of the data and tests it
on the rest. Returns results.
'''

# Import necessary modules
import straeto_generate_labeled_data    # To generate labeled bus stop data
import custom_straeto_tools             # To use custom straeto tools
import os                               # To check and get data from other files


# Get labeled data
# Check if data exists, if not generate data
#if os.path.exists("busstop./file.txt")

# Generate data
bus_stops_static_data = straeto_generate_labeled_data.generate_bus_stop_static_data()   # Features
bus_stops_dynamic_data = straeto_generate_labeled_data.generate_bus_stop_dynamic_data(bus_stops_static_data)    # Only used for rating!!!

# Get targets for train_set and test_set
bus_stop_ratings = custom_straeto_tools.rate_bus_stops(bus_stops_dynamic_data)  # Stop ratings

# Split data into training set and testing set depending on number of bus stops. Use features and targets
train_features, test_features, train_targets, test_targets = custom_straeto_tools.split_data(bus_stops_static_data, bus_stop_ratings)

print("Train features keys:")
print(list(train_features.keys()))
print("test features keys:")
print(list(test_features.keys()))
print("Train targets keys:")
print(list(train_targets.keys()))
print("test targets keys:")
print(list(test_targets.keys()))

#train_targets = custom_straeto_tools.rate_leidir(train_set)

# Loop through a couple of different RDBFNN with different number of hidden layer nodes.
# Create radial basis function neural network


# Train RBFNN on training set


# Test RBFNN on test set, measure accuracy and log accuracy



# Plot the accuracy as a function of number of nodes






