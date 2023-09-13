# Author:   Huldar
# Date:     2023-09-12
# Project:  Assignment 3
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np

# Use same random seed as professor
np.random.seed(1234)

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution.
    n: Number of vectors
    k: Dimension of each vector
    mean:   Mean for K-variate normal distribution
    var:    variance for K-variate normal distribution

    Returns nxK array X of vectors x_i
    '''
    # Generate index matrix
    variance_index_matrix = [[var, 0], [0, var]]

    X_array = np.random.multivariate_normal(mean, variance_index_matrix, size=(n,k))
    # Return X_array
    return X_array


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    pass


def _plot_sequence_estimate():
    data = None # Set this as the data
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        """
            your code here
        """
    plt.plot([e[0] for e in estimates], label='First dimension')
    """
        your code here
    """
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    pass


def _plot_mean_square_error():
    pass


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass



# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Part 1.1
    print("Part 1.1")
    if gen_data(2, 3, np.array([0, 1, -1]), 1.3) == np.array([[ 0.61286571, -0.5482684 ,  0.86251906],
 [-0.40644746,  0.06323465,  0.15331182]]):
        print("Pass")
    else:
        print("Fail")



    # Part 1.2
    print("Part 1.2")
    

    # Part 1.3
    print("Part 1.3")
    
    # Part 1.4
    print("Part 1.4")

    # Part 1.5
    print("Part 1.5")

    # Part 1.6
    print("Part 1.6")


    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''
