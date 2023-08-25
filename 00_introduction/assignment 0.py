import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)   # To have the same as gradescope

def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    p = np.exp(-(np.power((x - mu),2))/(2*np.power(sigma,2)) )/np.sqrt(2*np.pi*np.power(sigma,2))
    return p

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x_range = np.linspace(x_start, x_end, 500)
    plt.plot(x_range, normal(x_range,sigma, mu))

def _plot_three_normals():
    # Part 1.2
    plt.clf()
    mu = 0
    sigma = 0.5
    plot_normal(sigma,mu,-5,5)
    mu = 1
    sigma = 0.25
    plot_normal(sigma,mu,-5,5)
    mu = 1.5
    sigma = 1
    plot_normal(sigma,mu,-5,5)

    plt.savefig("./1_2_1.png")


def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    sum = 0
    for i in range(len(weights)):
        sum = sum + weights[i]*np.exp(-np.power(x-mus[i],2)/(2*np.power(sigmas[i],2)))/(np.sqrt(2*np.pi*np.power(sigmas[i],2)))

    return sum

def _compare_components_and_mixture():
    # Part 2.2
    plt.clf()
    mus = [0, -0.5, 1.5]
    sigmas = [0.5, 1.5, 0.25]
    phis = [1/3, 1/3, 1/3]

    for i in range(len(mus)):
        plot_normal(sigmas[i],mus[i],-5,5)

    x_range = np.linspace(-5, 5, 500)
    plt.plot(x_range, normal_mixture(x_range,sigmas, mus, phis))

    plt.savefig("./2_2_1.png")
    plt.clf()

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    print("Part 3.1")
    # Step 1
    samples = []

    normal_components = np.random.multinomial(n_samples, weights)

    for i in range(len(sigmas)):
        samples.append(np.random.normal(mus[i], sigmas[i], normal_components))
        

    return samples



#def _plot_mixture_and_samples():
    # Part 3.2

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    # Part 1.1
    #print("Test 1:")
    #print(normal(0, 1, 0) == 0.3989422804014327)
    #print("Test 2:")
    #print(normal(3, 1, 5) == 0.05399096651318806)
    #print("Test 3:")
    #print(normal(np.array([-1,0,1]), 1, 0))
    #print(np.array[0.24197072, 0.39894228, 0.24197072])
    #print(normal(np.array([-1,0,1]), 1, 0) == [0.24197072, 0.39894228, 0.24197072])

    # Part 1.2
    #_plot_three_normals()

    # Part 2.1
    #print("Part 2.1")
    #print("example 1")
    #print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))
    #print(normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1]))

    #_compare_components_and_mixture()

    # Part 3.1
    print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3))


    print("#########################################")
    print("Run succesfull :)")