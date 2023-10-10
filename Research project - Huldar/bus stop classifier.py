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
bus_stops_static_data = straeto_generate_labeled_data.generate_bus_stop_static_data()
bus_stops_dynamic_data = straeto_generate_labeled_data.generate_bus_stop_dynamic_data()
#print(bus_stops_data.get(1))

# Split data into training set and testing set depending on number of bus stops
train_set, test_set = custom_straeto_tools.split_data(bus_stops_static_data, bus_stops_dynamic_data)

# Get targets for train_set and test_set

# Loop through a couple of different RDBFNN with different number of hidden layer nodes.
# Create radial basis function neural network


# Train RBFNN on training set


# Test RBFNN on test set, measure accuracy and log accuracy



# Plot the accuracy as a function of number of nodes






