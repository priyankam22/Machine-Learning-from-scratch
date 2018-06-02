import numpy as np
import split_standardize

def load_simulated_data(n=1000, a=10, b=5):
    '''
    Generates simulated data for testing SVM
    '''
    X = np.vstack((np.random.uniform(size=n), np.random.uniform(size=n))).T
    t = X[:,1] - np.sin(a*X[:,0])
    prob = 1/(1+np.exp(-b*t))
    Y = np.random.binomial(n=1, p=prob, size=n)
    
    # Visualize
    # plt.scatter(X[:,0], X[:,1], c=['red' if y == 1 else 'blue' for y in Y])
    
    return split_standardize.split_standardize(X, Y)