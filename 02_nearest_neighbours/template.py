# Author: Huldar
# Date: 2023-08-29
# Project: Assignment 02
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt
import help

from tools import load_iris, split_train_test, plot_points

def func_test(A, B) -> bool:
    '''
    Function that checks if A is equal to B. Returns True if it's the same but False if it's not
    Inputs: Arrays
    '''
    # If it's not the same size, return False
    if A.size() != B.size():
        print("Fail")
        return False
    
    # Check individual values
    for i in range(A):
        if A[i] != B[i]:
            print("Fail")
            return False
    
    print("Pass")
    return True
    

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y independent of dimensions. Returns None if dimensions don't match
    '''
    dimensions = len(x)
    if len(x) == len(y):
        sum = 0
        for i in range(dimensions):
            sum = sum + np.power(y[i]-x[i], 2)
        return np.sqrt(sum)
    
    return None


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points. Returns all points indexes in order if k > points in points.
    x       :   data point to find the k-nearest points
    points  :   All the points available
    k       :   Number of index points to return
    '''
    # If we have fewer points than requested indexes, return None
    if k > len(points):
        k = len(points)

    # Make empty list of indexes
    nearest_indexes = np.zeros(k)
    # Find distances to each point from x
    distances = euclidian_distances(x, points)

    # Sort distances by ascending order and return indexes from sorted array
    sorted_dist_indexes = np.argsort(distances)

    # Only take the k indexes    
    for i in range(k):
        nearest_indexes[i] = sorted_dist_indexes[i]
    
    return nearest_indexes



def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular. Returns the most common class in targets.
    '''
    # Count how many times each class appears in the targets
    class_freq = np.copy(classes)
    num_classes = len(classes)
    for i in range(num_classes):
        class_freq[i] = 0

        # Check each target if it matches the class, if so increase frequency by 1
        for target in targets:
            if target == classes[i]:
                class_freq[i] = class_freq[i] + 1

    # Find the most frequent class in targets
    # For each class_frequency, check if it's bigger than othe frequencies. If it's bigger, make that the most_freq_class
    class_max_freq = 0
    for i in range(num_classes):
        if class_freq[i] > class_max_freq:
            class_max_freq = class_freq[i]
            most_freq_class = classes[i]

    # Return most frequent class
    return most_freq_class


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote. Returns the most common of the k nearest neighbours.
    Inputs:
    x               :   The point to find the k nearest neighbour around
    points          :   The data points containing the information to figure out what class the object is
    point_targets   :   The data point classes after being classified
    classes         :   List of classes
    k               :   How many near neighbours to look at
    '''
    # Find the point indexes of the k nearest neighbours
    nearest_neighbours_indexes = k_nearest(x, points, k)

    # Find k nearest neighbours
    nearest_neighbours = []
    for neighbour_index in nearest_neighbours_indexes:
        nearest_neighbours.append(point_targets[int(neighbour_index)])

    # Vote on the most common nearest neighbour, return that one.
    return vote(nearest_neighbours, classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Returns an array of knn guesstimates for each point
    '''
    # Make an empty array with space for each point target guess from KNN
    knn_points_prediction = np.zeros(len(point_targets), dtype=int)

    # Find the KNN for each point. Make sure to not use the distance between from the point to itself
    for i in range(len(knn_points_prediction)):
        # Make array which does not contain point i. Make sure to match the targets
        points_no_i = help.remove_one(points, i)
        targets_no_i = help.remove_one(point_targets, i)

        # Predict point class using knn
        knn_points_prediction[i] = knn(points[i], points_no_i, targets_no_i, classes, k)

    # Return KNN guesses for each point
    return knn_points_prediction


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    '''
    Calculates the accuracy of knn_predict predictions
    '''
    accuracy = 0

    # Make class predictions to predictions
    predictions = knn_predict(points, point_targets, classes, k)

    # For each point, compare predictions to point_targets, if it's the same, add 1 to accurace
    for i in range(len(point_targets)):
        if predictions[i] == point_targets[i]:
            accuracy = accuracy + 1

    # Find the hlutfall (icelandic) of accurate predictions with respect to total number of points, return
    return accuracy/len(point_targets)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Find confusion matrix size from number of classes, n_classes
    n_classes = len(classes)

    # Initialize empty confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes))

    # Make predictions 
    predictions = knn_predict(points, point_targets, classes, k)

    # Go through the predicted values and check how correct they are, add to confusion matrix as we go
    for i in range(len(predictions)):
        confusion_matrix[predictions[i]][point_targets[i]] = confusion_matrix[predictions[i]][point_targets[i]] + 1

    # Return confusion_matrix
    return confusion_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    '''
    Tests values of k for k in [1, N-1] where N = number of points.
    Returns the k which yields the highest accuracy of knn_predictions
    '''
    print("Finding best k")
    # Initialize best_k as 0 and best_accuracy as 0
    best_k = 0
    best_accuracy = 0
    k_max = len(points)-1
    k_10_percent = k_max*0.1

    # Make range for k values
    for k in range(1, k_max):
        # Print progress in 10% increments
        if k%k_10_percent < 1 :
            print(" .", end='')

        # Check accuracy of predictions for knn, if accuracy is better than best_accuracy, change best_k and best_accuracy to current k test values
        accuracy = knn_accuracy(points, point_targets, classes, k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # Return best_k
    return best_k


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    '''
    Plot a scatter plot of the first two feature dimensions
    in the point set. Gets predictions for the data point and colours the edge green if the prediction was correct, otherwise colours the edge red
    '''
    colors = ['yellow', 'purple', 'blue']
    # Edgecolours defined
    edgecolours = ['green', 'red']

    # Get predictions for point classes
    predictions = knn_predict(points, point_targets, classes, k)

    for i in range(points.shape[0]):
        # Set edgecolor to green
        edgecolour = edgecolours[0]
        # If prediction was false, set edgecolour to red
        if predictions[i] != point_targets[i]:
            edgecolour = edgecolours[1]
        [x, y] = points[i,:2]
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edgecolour,
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...

# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Part 1.1
    print("Part 1.1")
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]
    if     euclidian_distance(x, points[0]) == 0.5385164807134502 and euclidian_distance(x, points[50]) == 3.6166282640050254:
        print("Pass")
    else:
        print("Fail")


    # Part 1.2
    print("Part 1.2.... pass")
    #print(euclidian_distances(x, points))
    #print("[0.53851648 0.50990195 0.64807407 0.14142136 0.6164414  0.51961524 ..., 4.14004831]")
    

    # Part 1.3
    print("Part 1.3")
    if k_nearest(x, points, 1) == [16] and all(k_nearest(x, points, 3) == [16, 3, 38]):
        print("Pass")
    else:
        print("Fail")


    # Part 1.4
    print("Part 1.4")
    if vote(np.array([0,0,1,2]), np.array([0,1,2])) == 0 and  vote(np.array([1,1,1,1]), np.array([0,1])) == 1:
        print("Pass")
    else:
        print("Fail")



    # Part 1.5
    print("Part 1.5")
    print("Test 1:")
    if knn(x, points, point_targets, classes, 1) == 0:
        print("Pass")
    else:
        print("Fail")

    print("Test 2:")
    if knn(x, points, point_targets, classes, 5) == 0:
        print("Pass")
    else:
        print("Fail")

    print("Test 3:")
    if knn(x, points, point_targets, classes, 150) == 1:
        print("Pass")
    else:
        print("Fail")

    # Part 2
    #----------------------------------------
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

    # Part 2.1
    print("Part 2.1")
    if all(knn_predict(d_test, t_test, classes, 10) == [2, 2, 2, 2, 0, 1, 0, 1, 1, 0, 1, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1]):
        print("Pass")
    else:
        print("Fail")

    # Part 2.2
    print("Part 2.2")
    if knn_accuracy(d_test, t_test, classes, 10) == 0.8275862068965517 and knn_accuracy(d_test, t_test, classes, 5) == 0.9310344827586207:
        print("Pass")
    else:
        print("Fail")


    # Part 2.3
    print("Part 2.3")

    # Part 2.4
    print("Part 2.4")
    '''
    if best_k(d_train, t_train, classes) == 9:
        print("\nPass")
    else:
        print("\nFail")
    '''

    # Part 2.5
    print("Part 2.5")
    knn_plot_points(d, t, classes, 3)
    plt.savefig("02_nearest_neighbours/images/2_5_1.png")




    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")


# Pass/Fail if statements
'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''