# Author: Huldar
# Date: 2023-08-26
# Project: 01_Decision trees
# Acknowledgements: 
#

'''
Important iris info.
Iris classes:
0 : Iris Setosa
1 : Iris Versicolour
2 : Iris Virginica

Iris features:
x_i1 is sepal length in centimeters
x_i2 is sepal width in centimeters
x_i3 is petal length in centimeters
x_i4 is petal width in centimeters
'''


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    # Count how many times each class appears in the targets
    classes_frequency = []
    n_targets = len(targets)
    if n_targets > 0:
        # For each class, count number of occurences
        for i in range(len(classes)):
            frequency = 0
            # Count occurences
            for j in range(len(targets)):
                if targets[j] == classes[i]:
                    frequency = frequency+1
            # Add number of occurences to classes_frequency
            classes_frequency.append(frequency/n_targets)

    return classes_frequency

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    # Create completely empty lists to add to later
    features_1 = []
    targets_1 = []
    features_2 = []
    targets_2 = []
    
    # If feature < theta, add to features_1 else add to features_2
    for i in range(len(targets)):
        if features[i][split_feature_index] < theta:
            features_1.append(features[i])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i])
            targets_2.append(targets[i])

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    # Sum odds of classes for each class
    return 0.5*(1 - sum(np.power(prior(targets, classes), 2)))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = len(t1) + len(t2)
    
    return (len(t1) * g1 + len(t2) * g2) / n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)

    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None

    # iterate feature dimensions
    for split_feature_index in range(features.shape[1]):
        # create the thresholds
        min_value = features[split_feature_index].min()
        max_value = features[split_feature_index].max()
        thetas = np.linspace(min_value, max_value, num_tries + 2)[1:-1]
        #print("Thetas array:")
        #print(thetas)

        # iterate thresholds
        for theta in thetas:
            try:
                gini_impurity = total_gini_impurity(features, targets, classes, split_feature_index, theta)
                if gini_impurity < best_gini:
                    best_gini = gini_impurity
                    best_dim = split_feature_index
                    best_theta = theta
            except Exception as e:
                print(e)
                print("Feature index: " + str(split_feature_index) + " - Theta: " + str(theta))

    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        '''
        Train the data tree model, using the features, targets and classes
        '''
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        '''
        Checks the accuracy of the decision tree using the test data.
        Returns None if self.test_targets contains no data.
        '''
        # for test_feature in self.test_features:
        predicted_targets = self.tree.predict(self.test_features)
        num_test_targets = len(self.test_targets)
        if num_test_targets == 0:
            return None
        accuracy = 0
        for i in range(num_test_targets):
            if predicted_targets[i] == self.test_targets[i]:
                accuracy = accuracy + 1

        return accuracy/num_test_targets


    def plot(self):
        '''
        Plot decision tree
        '''
        plt.clf()
        tree.plot_tree(self.tree)
        

    def plot_progress(self):
        '''
        Plots the accuracy on the test set as a function of training samples.
        Starts by training on only one sample and end on training on all the training samples
        '''
        # Independent section
        # Remove this method if you don't go for independent section.
        print("Plotting accuracy progress")

        # Find number of training samples
        max_training_samples = len(self.train_features)

        n_train_samples = list(range(max_training_samples))

        accuracy = np.zeros(max_training_samples)

        # Measure progress
        for i in range(max_training_samples):
            # Make subset of training samples - Possible improvement, make subset a random subset from training samples
            subset_train_samples = self.train_features[0:i+1]
            subset_train_targets = self.train_targets[0:i+1]

            # Train tree with samples
            self.tree.fit(subset_train_samples, subset_train_targets)

            # Check accuracy for subset of training samples
            accuracy[i] = self.accuracy()
                
        # Plot accuracy as a function of sample number
        plt.clf()
        plt.plot(n_train_samples, accuracy)



        

    def guess(self):
        '''
        Returns the predictions on the test data
        '''
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        '''
        Returns the confusion matrix for the test data
        '''
        n_classes = len(self.classes)

        # Initialize empty confusion matrix
        confusion_matrix = np.zeros((n_classes, n_classes))

        guesses = self.guess()

        # Go through the predicted values and check how correct they are, add to confusion matrix as we go
        for i in range(len(guesses)):
            confusion_matrix[guesses[i]][self.test_targets[i]] = confusion_matrix[guesses[i]][self.test_targets[i]] + 1

        return confusion_matrix






# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Part 1.1
    print("Part 1.1")
    print(prior([0, 0, 1], [0, 1]))
    print(prior([0, 2, 3, 3], [0, 1, 2, 3]))

    print("Part 1.2")
    print("Loading iris dataset...")
    features, targets, classes = load_iris()

    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    print("f_1 number of samples: " + str(len(f_1)))
    print("f_2 number of samples: " + str(len(f_2)))

    print("Part 1.3")
    print("Gini impurity test 1:")
    if gini_impurity(t_1, classes) == 0.2517283950617284:
        print("Pass")
    else:
        print("Fail")
    print("Gini impurity test 2:")
    if gini_impurity(t_2, classes) == 0.1497222222222222:
        print("Pass")
    else:
        print("Fail")

    # Part 1.4
    print("Part 1.4")
    print("Weighted impurity: " + str(weighted_impurity(t_1, t_2, classes)))
    print("Weighted impurity test:")
    if weighted_impurity(t_1, t_2, classes) == 0.2109259259259259:
        print("Pass")
    else:
        print("Fail")
        
    # Part 1.5
    print("Part 1.5")
    if total_gini_impurity(features, targets, classes, 2, 4.65) == 0.2109259259259259:
        print("Pass")
    else:
        print("Fail")

    # Part 1.6
    print("Part 1.6")
    print("Brute best split")
    brute_split_result = brute_best_split(features, targets, classes, 30)
    print("Brute split results:")
    print(brute_split_result)
    if brute_split_result == (0.16666666666666666, 2, 1.9516129032258065):
        print("Pass")
    else:
        print("Fail")

    # Part 2 test:
    dt = IrisTreeTrainer(features, targets, classes=classes)
    dt.train()
    print(f'The accuracy is: {dt.accuracy()}')

    # Part 2.3
    dt.plot()
    plt.savefig("./01_decision_trees/images/2_3_1.png")

    # Part 2.4
    print("Part 2.4")
    print(f'I guessed:\n {dt.guess()}')
    print(f'The true targets are:\n {dt.test_targets}')

    # Part 2.5
    print("Part 2.5")
    print(dt.confusion_matrix())


    # Independent part
    print("Independent part")
    dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    dt.plot_progress()
    plt.savefig("01_decision_trees/images/indep_1.png")


    print("Run succesful")