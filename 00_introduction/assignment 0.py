import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    p = np.exp(-(np.power((x - mu),2))/(2*np.power(sigma,2)) )/np.sqrt(2*np.pi*np.power(sigma,2))
    return p

#def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2

#def _plot_three_normals():
    # Part 1.2

#def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1

#def _compare_components_and_mixture():
    # Part 2.2

#def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

#def _plot_mixture_and_samples():
    # Part 3.2

# Tests
if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    print("Test 1:")
    print(normal(0, 1, 0) == 0.3989422804014327)
    print("Test 2:")
    print(normal(3, 1, 5) == 0.05399096651318806)
    print("Test 3:")
    print(normal(np.array([-1,0,1]), 1, 0))
    #print(np.array[0.24197072, 0.39894228, 0.24197072])
    print(normal(np.array([-1,0,1]), 1, 0) == [0.24197072, 0.39894228, 0.24197072])
