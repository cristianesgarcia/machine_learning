# from utils import messages
import numpy as np

class GradientDescent:
    """Class to gradient descent"""

    def __init__(self):
        self.a = 1
        # self.msg = messages.Messages()
    
    def gradient_descent(self, X, y, gamma=0.1, initial_guess='random', number_iterations=1000):
        N = X.shape[0]
        number_params = X.shape[1]
        np.random.seed(42)
        theta_evolution = np.ones((number_params,number_iterations))

        if initial_guess == 'random':
            theta = np.random.randn(number_params,1)
        else:
            theta = initial_guess
        
        for iteration in range(number_iterations):
            # evolution of the parameter vector
            theta_evolution.itemset((0,iteration), theta[0])
            theta_evolution.itemset((1,iteration), theta[1])
            # calculate the gradient at each iteration
            gradient = 2/N * X.T.dot(X.dot(theta) - y)
            # update theta
            theta = theta - gamma*gradient

        return theta, theta_evolution
        


