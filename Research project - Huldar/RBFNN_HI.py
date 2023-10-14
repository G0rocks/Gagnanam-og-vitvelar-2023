# Author:           Huldar
# Date:             2023-10-10
# Project:          Research project
'''
Acknowledgements:
Based on An Efficient Method to Construct a Radial Basis Function
            Neural Network Classifier
By Young-Sup Hwang and Sung-Yang Bang
At Pohang University of Science and Technology
A paper published the 11th of december in 1996

That paper uses the APC-III algorithm from Hwang & Bang, 1994
----------------------------------------------------------------------
Description:
RBFNN_HI is a module and an acronym which stands for
Radial Basis Function Neural Network Huldar Implementation
This module is my implementation of a radial basis function neural network. From now on radial basis function neural network will be abbreviated as RBFNN.

Mostly this module allows you to create an instance of the RBFNN class and use the class methods on the class to create your own RBFNN.

Notes:
In the paper they use the word "pattern" for the same use that we use the word "feature"
'''

import numpy    # Used to work with arrays
import time     # Used to record time
import math     # Used to do math
import random   # Used to generate random numbers

def euclid_distance(A: numpy.array, B: numpy.array) -> float:
    '''
    Calculates the euclidian distance between the two input numpy arrays. ||B-A||
    Returns None if the inputs are not of the same dimensions or if the array objects do not have exactly 1 row

    Inputs:
    A: 1D numpy.array object. Coordinates of starting point
    B: 1D numpy.array object. Coordinates of ending point

    Outputs:
    euclid_dist:    float. Euclidian distance between A and B
    '''
    # Check if sizes are the same
    colA = A.shape[0]
    colB = B.shape[0]
    if colA == colB:
        # Go through each column, get euclid distance
        euclid_dist = 0
        for i in range(colA):
            euclid_dist = euclid_dist + (B[i]-A[i])*(B[i]-A[i])

        return math.sqrt(euclid_dist)
    return None

def __get_index_of_max_value__(vector: numpy.array) -> int:
    '''
    Get index of max value in vector
    Inputs:
    vector: 1D numpy.array object

    Outputs:
    index:  int. Index of the maximum value in the 1D input vector array. Returns None of no value is maximum. If 2 or more values share the maximum value, returns the biggest index.
    '''
    # Start with minimum number as max and index as an out of bounds index
    max = -numpy.inf
    index = None

    for i in range(vector.shape[0]):
        # If value is bigger than max, update max and index to current index
        if vector[i] > max:
            max = vector[i]
            index = i

    return index

class RBFNN:
    '''
    RBFNN class.
    '''
    def __init__(self,cluster_scale_factor_alpha: float = 1, n_neurons: int=10, hardware_inhibitor=True, weight_increment= math.pow(10,-10)) -> None:
        '''
        Initialization function for the RBFNN class. Creates a hidden layer of neurons where all the neuron values are zero.
        Initializes neurons as zero and self.classes, self.train_time, self.test_time and self.accuracy as None

        Inputs:
        cluster_scale_factor_alpha: (float) multiplier for the cluster radius
        n_neurons:    (int) number of neurons in the hidden layer of the RBFNN, defaults to 10. Note that this includes the bias neuron which has a value of zero.
        hardware_inhibitor: (bool) True by default. If you want to allow the use of more than 1000 neurons you must set hardware_inhibitor to False (turn it off). hardware_inhibitor will bring down n_neurons to 1000 maximum if you attempt to input more than 1000 

        Outputs:
        There are no outputs
        '''
        # Safety check to make sure the neurons
        if n_neurons<1:
            n_neurons = 1
        self.hardware_inhibitor = hardware_inhibitor    # Hardware inhibitor protects hardware from super long computing times
        if self.hardware_inhibitor:
            if n_neurons > 1000:
                n_neurons = 1000
        self.neurons = numpy.zeros(n_neurons) # numpy.array object containing the hidden layer of the RBFNN
        self.n_neurons = n_neurons  # int object with total number of neurons in the hidden layer of the RBFNN
        self.cluster_scale_factor_alpha = cluster_scale_factor_alpha # Scale factor alpha for 
        self.clusters   = []    # List of all clusters. Each cluster contains the features that are less than the cluster_radius from the cluster center
        self.features_in_cluster = []   # List where the first value contains the number of how many features are contained within the first cluster, the second value how many features are contained within the second cluster etc.
        self.cluster_radius = 0    # Radius around clusters which features can belong to.
        self.classes = []   # List of known classes to the RBFNN
        self.weights = numpy.zeros(len(self.classes))   # numpy.array object containing the weights to calculate the output based on the neuron values
        self.weight_increment = weight_increment # How much to increment the weight changes during training
        self.train_time = None  # Time it took to do the last training of the RBFNN
        self.test_time = None   # Time it took to do the last test of the RBFNN
        self.accuracy = None    # Accuracy of RBFNN as a floating point number between 0 and 1

    def __find_unique_classes__(self, targets):
        '''
        Finds the unique target classes and adds the ones that are not already in self.classes to self.classes.

        Inputs:
        targets:    2D numpy.array object where each row represents different classes. Note that classes can have many columns.

        Outputs:
        No outputs.
        Updates self.classes to include the unique classes
        '''
        # Get dimensions of target classes
        rows, columns = targets.shape

        # Loop through all target classes and compare with self.classes. If a new class is found, append to self.classes
        for i in range(rows):
            # Assume class is unknown
            unknown_class = True

            # Update number of known classes
            n_classes = len(self.classes)

            # Compare row to known classes
            for j in range(n_classes):
                for k in range(columns):
                    if targets[i][k] == self.classes[j][k]:
                        unknown_class = False

            # If unknown class, add to list of known classes, else do nothing
            if unknown_class:
                self.classes.append(targets[i][:])
            
        # 2D numpy.array object containing the weights to calculate the output based on the neuron values
        self.weights = numpy.zeros([self.n_neurons, len(self.classes)])


    def __n_clusters__(self) -> int:
        '''
        Number of clusters (initialize as 1). Clusters are the connection between the input layer and the hidden layer
        '''
        return len(self.clusters)

    def __update_cluster_radius__(self, features: numpy.array):
        '''
        Updates the cluster radius

        Inputs:
        features

        Outputs:
        No outputs.
        Updates self.cluster_radius
        '''
        # Find number of features
        n_features = features.shape[0]

        # Find sum of minimum distances between all features
        min_dist = float("inf")
        min_dist_sum = 0
        for i in range(n_features):
            for j in range(n_features):
                # Skip if i=j
                if i != j:
                    # If euclid distance is shorter than min_dist, update min_dist
                    euclid_dist = euclid_distance(features[i], features[j])
                    if euclid_dist < min_dist:
                        min_dist = euclid_dist
            # Update sum
            min_dist_sum = min_dist_sum + min_dist

        # Find the mean minimum distance between features
        min_dist_mean = min_dist_sum / n_features
        
        # Compute cluster radius by multiplying min_dist_mean with self.cluster_scale_factor_alpha
        self.cluster_radius = self.cluster_scale_factor_alpha*min_dist_mean

    def __update_neurons__(self, feature: numpy.array):
        '''
        Updates all neuron values for the hidden layer neurons using this 1 input feature

        Inputs:
        feature:   1D numpy.array object where each column represents a different feature data point

        Outputs:
        No outputs.
        Updates self.neurons
        '''
        # Loop through each neuron
        self.neurons[0] = 1 # Make sure bias neuron stays as one

        for i in range(1, self.n_neurons):
            euclid_dist = euclid_distance(feature,self.clusters[i]) # Very unsure about this - Huldar 2023-10-11
            self.neurons[i] = math.exp(-euclid_dist/2/self.cluster_radius/self.cluster_radius)

    def __new_cluster__(self, feature):
        '''
        Makes a new cluster
        '''
        # Add feature to newest cluster
        self.clusters.append(feature)
        # Update number of features in cluster
        self.features_in_cluster.append(1)

    def __update_weights__(self, feature: numpy.array, target: numpy.array):
        '''
        Updates the weights between the hidden layer and the output layer. The weights are as many as the classes available.

        Input:
        feature:    1D numpy.array object containing a feature data point
        target:    1D numpy.array object containing a target class
        '''
        # Weight shape
        rows, columns = self.weights.shape

        # Classify using current weights
        guess = self.classify(feature)

        # Loop through each weight
        for i in range(rows):
            for j in range(columns):
                # If the weight guessed correctly, do nothing
                # If the weight undershot the guess, increase weight
                if guess < target:
                    self.weights[i][j] = self.weights[i][j] + self.weight_increment

                # If the weight overshot the guess decrease weight.
                if guess > target:
                    self.weights[i][j] = self.weights[i][j] - self.weight_increment

    def classify(self, feature: numpy.array) -> int:
        '''
        Classifies the given features using the neurons and the RBFNN method given in the paper
        '''
        # Update the neurons 
        self.__update_neurons__(feature)

        # Multiply each neuron value by it's corresponding weight and sum up the results
        classification = numpy.zeros([len(self.classes)])
        for i in range(self.n_neurons):
            for j in range(len(self.classes)):
                # Loop through each class
    #            for j in range(len(self.classes)):
                classification[j] = classification[j] + self.neurons[i]*self.weights[i][j]

        # Find which class has the highest odds in classification
        max_class = __get_index_of_max_value__(classification)
        return max_class

    def random_classify(self, feature: numpy.array) -> numpy.array:
        '''
        Classifies the given features using random values
        '''
        # How many classes do we have
        n_classes = len(self.classes)

        # Randomly pick one and return it
        return numpy.array([self.classes[random.randint(0,n_classes)-1]])

    def train(self, train_features, train_targets, train_loops: int=100) -> float:
        '''
        Train RBFNN

        Inputs:
        train_features: 2D numpy.array object where each column represents a different feature
        train_targets: 2D numpy.array object where each column represents a different target

        Output:
        updates self.classes with the available classes in train_targets
        updates self.train_time which is a float that tells you how much time it took to train the model.
        '''
        # Record start time with time_in
        time_in = time.time()

        # Reinitialize self
        self.__init__(cluster_scale_factor_alpha=self.cluster_scale_factor_alpha, n_neurons= self.n_neurons,hardware_inhibitor= self.hardware_inhibitor)
        
        # Update list of known classes
        self.__find_unique_classes__(train_targets)

        # Find feature dimensions
        features_rows, features_columns = train_features.shape

        # Update cluster radius
        self.__update_cluster_radius__(train_features)

        # Set center of clusters for each feature
        # Set first cluster center to first feature
        self.__new_cluster__(train_features[0])
        for i in range(1, features_rows):   # For each feature
            # Feature found in cluster set to False
            feature_in_cluster = False

            # Check each cluster to see if feature is in cluster
            for j in range (len(self.clusters)): # For each cluster
                # Compute euclidian distance between feature i and jth cluster
                euclid_dist = euclid_distance(train_features[i],numpy.array(self.clusters[j]))

                # If feature is in cluster or on cluster border include it in the cluster
                if euclid_dist <= self.cluster_radius:
                    # Feature found in cluster set to True
                    feature_in_cluster = True
                    # Add feature i to cluster j
                    self.clusters[j] = (self.clusters[j]*self.features_in_cluster[j] + train_features[i])/(self.features_in_cluster[j] + 1)
                    # Increment number of features in cluster j by 1
                    self.features_in_cluster[j] = self.features_in_cluster[j] + 1   # Note that the code from the paper uses 'i' here instead of 'j' but author of code believes this to be more accurate
                    # Break from cluster loop and go to next feature
                    break

            # If feature was not found in any cluster, make new cluster and add feature i to it
            if not(feature_in_cluster):
                 self.__new_cluster__(train_features[i])


        # Attempt to classify all features correctly for each of the loops
        for i in range(train_loops):
            for i in range(len(train_features)):            
                # Update the weights
                self.__update_weights__(train_features[i], train_targets[i])

        # Log train_time
        self.train_time = time.time() - time_in

    def test(self, test_features: numpy.array, test_targets: numpy.array):
        '''
        Tests the RBFNN with the test features and test targets

        Inputs:
        train_features: 2D numpy.array object where each column represents a different feature
        train_targets: 2D numpy.array object where each column represents a different target

        Output:
        updates self.accuracy which is a float that tells you how accurate the RBFNN is in classifying the test features against the test targets.
        updates self.train_time which is a float that tells you how much time it took to train the model.
        '''
        # Record start time with time_in
        time_in = time.time()

        # Set accuracy to zero
        self.accuracy = 0

        # Go through each feature and classify, check if the classification was correct
        for i in range(len(test_features)):
            # If classification was correct, count times
            # print("Classified " + str(test_targets[i]) + " as " + str(self.classify(test_features[i])))
            if self.classify(test_features[i]) == test_targets[i]:
                # Increment accuracy
                self.accuracy = self.accuracy + 1

        # Update accuracy and log
        self.accuracy = self.accuracy/test_features.shape[0]

        # Log test_time
        self.test_time = time.time() - time_in

    def test_random(self, test_features: numpy.array, test_targets: numpy.array):
        '''
        Tests the RBFNN with the test features and test targets using random classification

        Inputs:
        train_features: 2D numpy.array object where each column represents a different feature
        train_targets: 2D numpy.array object where each column represents a different target

        Output:
        updates self.accuracy which is a float that tells you how accurate the RBFNN is in classifying the test features against the test targets.
        updates self.train_time which is a float that tells you how much time it took to train the model.
        '''
        # Record start time with time_in
        time_in = time.time()

        # Set accuracy to zero
        self.accuracy = 0

        # Go through each feature and classify, check if the classification was correct
        for i in range(len(test_features)):
            # If classification was correct, count times
            if self.random_classify(test_features[i]) == test_targets[i]:
                # Increment accuracy
                self.accuracy = self.accuracy + 1

        # Update accuracy and log
        self.accuracy = self.accuracy/test_features.shape[0]

        # Log test_time
        self.test_time = time.time() - time_in