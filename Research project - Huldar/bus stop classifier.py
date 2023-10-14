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
import RBFNN_HI                         # To create and work with radial basis function neural networks
import matplotlib.pyplot                # To generate plots
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


# Loop through a couple of different RDBFNN with different number of hidden layer neurons.
n_neurons_to_check = [10, 30]
# alphas_to_check = [1]
rbfnn_list = []
print("Initializing RBFNNs (radial basis function neural network) with neuron numbers " + str(n_neurons_to_check))
for i in range(len(n_neurons_to_check)):
    # Create radial basis function neural network
    rbfnn_list.append(RBFNN_HI.RBFNN(n_neurons=n_neurons_to_check[i], cluster_scale_factor_alpha=1.1))

# Train RBFNN on training set
print("Training RBFNNs")
for rbfnn in rbfnn_list:
    rbfnn.train(train_features, train_targets)

# Test random classification on test set, measure and log accuracy, time to train
print("Testing random classification")
rand_accuracies = []
for rbfnn in rbfnn_list:
    rbfnn.test_random(test_features, test_targets)
    # Collect accuracies
    rand_accuracies.append(rbfnn.accuracy)
    print("Random test accuracy: " + str(rbfnn.accuracy))


# Test RBFNN on test set, measure and log accuracy, time to train
print("Testing RBFNNs")
accuracies = []
for rbfnn in rbfnn_list:
    rbfnn.test(test_features, test_targets)
    # Collect accuracies
    accuracies.append(rbfnn.accuracy)
    print("RBFNN test accuracy: " + str(rbfnn.accuracy))


# Plot the accuracy as a function of number of neurons
print("Plotting accuracies as a function of number of neurons in the RBFNNs")
matplotlib.pyplot.plot(n_neurons_to_check, accuracies)
matplotlib.pyplot.show()




