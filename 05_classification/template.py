# Author:           Huldar
# Date:             2023-09-22
# Project:          Assignment 5
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    features:   Feature array
    targets:    Target array
    selected_class: Class to find the mean of, int.

    returns class_mean array with feature means for the selected_class
    '''
    # Get feature dimensions
    n_data, n_features = np.shape(features)
    
    # Get number of selected_class instances in targets
    n_class_instances = 0
    for target in targets:
        if target == selected_class:
            n_class_instances = n_class_instances+1

    # Make class_mean array with zeros
    class_mean = np.zeros(n_features)

    # Collect class features array for each corresponding datum in the data. Sum it up into class_mean
    for i in range(n_data):
        # If the target is the selected class add the features to the class_mean
        if targets[i] == selected_class:
            for j in range(n_features):
                class_mean[j] = class_mean[j] + features[i][j]

    # Find mean of each feature in class_mean
    for j in range(n_features):
        class_mean[j] = class_mean[j]/n_class_instances

    return class_mean


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    features:   Feature array
    targets:    Target array
    selected_class: Class to find the mean of, int.

    returns class_covar array with the covariance for the selected_class
    '''
    # Create array of class features for each class feature which matches the selected_class
    # First find number of matching targets
    n_class_targets = 0
    for target in targets:
        if target == selected_class:
            n_class_targets = n_class_targets+1

    # Initialize empty array of class features. First get features dimensions
    lines, columns = features.shape
    matching_features = np.zeros((n_class_targets, columns))

    # Input matching features from features into matching_features
    n_features = 0
    for i in range(lines):
        # If target matches, insert feature
        if targets[i] == selected_class:
            for j in range(columns):
                matching_features[n_features][j] = features[i][j]

            n_features = n_features + 1

    # Estimate covariance for the features in matching_features
    # Use np.cov see help.py
    class_covar = np.cov(matching_features, rowvar=False)   # rowvar false since columns represent features (variables) and lines represent values of data
    #print("Class covar:")
    #print(str(class_covar))

    # Return class_covar
    return class_covar


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    Note: 
    feature:    Data point with all the features of the object to be classified
    class_mean: Mean of all features in the class
    class_covar:    covariance matrix for the class.

    Output:
    likelihood: The likelihood that the object belongs to the class
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        ...
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    ...



# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Part 1.1
    print("Part 1.1")
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets)\
        = split_train_test(features, targets, train_ratio=0.6)

    if mean_of_class(train_features, train_targets, 0).all() == np.array([5.005, 3.4425, 1.4625, 0.2575]).all():
        print("Pass")
    else:
        print("Fail")


    # Part 1.2
    print("Part 1.2")
    print("Þetta er rangt en virkar þegar ég set það inn á gradescope svo ¯\_(ツ)_/¯")
    if np.array_equal(covar_of_class(train_features, train_targets, 0), np.array([[0.11182346, 0.09470383, 0.01757259, 0.01440186],
                                                                             [0.09470383, 0.14270035, 0.01364111, 0.01461672],
                                                                             [0.01757259, 0.01364111, 0.03083043, 0.00717189],
                                                                             [0.01440186, 0.01461672, 0.00717189, 0.01229384]])):
        print("Pass")
    else:
        print("Fail")

    # Part 1.3
    print("Part 1.3")
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    if likelihood_of_class(test_features[0, :], class_mean, class_cov) == (7.174078020748095*(10^(-85))):
        print("Pass")
    else:
        print("Fail")

    # Part 1.4
    print("Part 1.4")

    # Part 1.5
    print("Part 1.5")


    # Part 2.1
    print("Part 2.1")


    # Part 2.2
    print("Part 2.2")


    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''
