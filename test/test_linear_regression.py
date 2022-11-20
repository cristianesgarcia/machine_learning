import pytest
import numpy as np

from main.supervised_learning.linear_regression import linear_regression
from utils import messages

class TestLinearRegression:
    def setup(self):
        self.linear_regression = linear_regression.LinearRegression()
        self.msg = messages.Messages()
        self.number_of_samples = 100
        self.parameters = np.array([[4.0],[3.0]])
        self.threshold = 5e-1
        # input data
        self.x = 2*np.random.rand(self.number_of_samples,1)
        # noiseless output data
        self.y_0 = self.parameters[1]*self.x + self.parameters[0]
        # noise signal
        self.eta = np.random.rand(self.number_of_samples,1)
        # output signal affected by noise
        self.y = self.y_0 + self.eta

    # Tests for the linear regression method using the normal equation
    # approach to estimate the parameters
    def test_linear_regression_normal_equation_empty_input_output(self):
        x = np.array([])
        y = np.array([1,2,3,4,5,6])
        with pytest.raises(Exception, match=self.msg.EMPTY_ARRAY):
            self.linear_regression.normal_equation(x, y)

    def test_linear_regression_normal_equation_different_size_arrays(self):
        x = np.array([1,2,3,4,5,6])
        y = np.array([1,2,3])
        with pytest.raises(Exception, match=self.msg.DIFFERENT_SIZE):
            self.linear_regression.normal_equation(x, y)

    def test_linear_regression_normal_equation_no_noise(self):
        parameters_estimated = self.linear_regression.normal_equation(
            self.x, self.y_0)
        assert self.parameters == pytest.approx(parameters_estimated,
            rel=self.threshold)
    
    def test_linear_regression_normal_equation(self):        
        parameters_estimated = self.linear_regression.normal_equation(
            self.x, self.y)
        assert self.parameters == pytest.approx(parameters_estimated,
            rel=self.threshold)
    
    # Tests for the linear regression method using the sckikit-learn 
    # library
    def test_linear_regression_empty_input_output(self):
        x = np.array([])
        y = np.array([])
        with pytest.raises(Exception, match=self.msg.EMPTY_ARRAY):
            self.linear_regression.linear_regression(x, y)
    
    def test_linear_regression_different_size_arrays(self):
        x = np.array([1,2,3,4,5,6])
        y = np.array([1,2,3])
        with pytest.raises(Exception, match=self.msg.DIFFERENT_SIZE):
            self.linear_regression.linear_regression(x, y)

    def test_linear_regression_no_noise(self):
        parameters_estimated = self.linear_regression.linear_regression(
            self.x, self.y_0)
        assert self.parameters == pytest.approx(parameters_estimated,
            rel=self.threshold)

    def test_linear_regression(self):
        parameters_estimated = self.linear_regression.linear_regression(
            self.x, self.y)
        assert self.parameters == pytest.approx(parameters_estimated,
            rel=self.threshold)