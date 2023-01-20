import pytest
import numpy as np
from random import seed
from random import gauss

from main.supervised_learning.polynomial_regression import polynomial_regression
from utils import messages

class TestPolynomialRegression:
    def setup(self):
        self.polynomial_regression = polynomial_regression.PolynomialRegression()
        self.msg = messages.Messages()
        self.number_of_samples = 100
        self.desired_parameters = np.array([0.5, 1, 2])
        self.degree = self.desired_parameters.size - 1
        self.threshold = 0.2
        # sets the seed
        np.random.seed(1234)
        # input data
        self.x = 6*np.random.rand(self.number_of_samples, 1) - 3
        # white gaussian noise with zero mean and variance equals one
        seed(1234)
        self.noise = [gauss(0.0, 1.0) for i in range(self.number_of_samples)]
        self.noise = np.array(self.noise)
        self.noise = self.noise.reshape((self.number_of_samples,1))
        # noiseless output
        self.y_0 = np.zeros(self.number_of_samples)
        self.y_0 = self.y_0.reshape((self.number_of_samples,1))
        for index, item in enumerate(self.desired_parameters):
            self.y_0 = self.y_0 + \
                self.desired_parameters[index]*self.x**(self.degree-index)
        # output affected by noise
        self.y = self.y_0 + self.noise
        print(self.noise)

    def test_empty_input(self):
        x = np.array([])
        with pytest.raises(Exception, match=self.msg.EMPTY_ARRAY):
            self.polynomial_regression.polynomial_regression(
                x, self.y, self.degree)

    def test_empty_output(self):
        y = np.array([])
        with pytest.raises(Exception, match=self.msg.EMPTY_ARRAY):
            self.polynomial_regression.polynomial_regression(
                self.x, y, self.degree)

    def test_different_size_arrays(self):
        x = np.array([1,2,3,4,5,6])
        y = np.array([1,2])
        with pytest.raises(Exception, match=self.msg.DIFFERENT_SIZE):
            self.polynomial_regression.polynomial_regression(
                x, y, self.degree)
    
    def test_noiseless_output(self):
        parameters_estimated = self.polynomial_regression.polynomial_regression(
            self.x, self.y_0, self.degree)
        assert self.desired_parameters.tolist() == pytest.approx(
            parameters_estimated,
            self.threshold)
    
    def test_polynomial_regression(self):
        parameters_estimated = self.polynomial_regression.polynomial_regression(
            self.x, self.y, self.degree)
        assert self.desired_parameters.tolist() == pytest.approx(
            parameters_estimated, self.threshold)

