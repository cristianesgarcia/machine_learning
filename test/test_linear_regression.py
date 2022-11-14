import pytest
import numpy as np

from supervised_learning.linear_regression import linear_regression
from utils import messages

class TestLinearRegression:
    # print("Module name", linear_regression.__name__)
    def setup(self):
        self.linear_regression = linear_regression.LinearRegression()
        self.msg = messages.Messages()

    # def test_linear_regression_empty_input_output(self):
    #     x = np.array([])
    #     y = np.array([1,2,3,4,5,6])
    #     with pytest.raises(Exception, match=self.msg.EMPTY_ARRAY):
    #         self.linear_regression.normal_equation(x, y)

    # def test_linear_regression_different_size_arrays(self):
    #     x = np.array([1,2,3,4,5,6])
    #     y = np.array([1,2,3])
    #     with pytest.raises(Exception, match=self.msg.DIFFERENT_SIZE):
    #         self.linear_regression.normal_equation(x, y)

    # def test_linear_regression_normal_equation_no_noise(self):
    #     x = 2*np.random.rand(100,1)
    #     y = 4 + 3*x
    #     parameters = np.array([[4.0],[3.0]])
    #     parameters_estimated = self.linear_regression.normal_equation(x, y)
    #     assert parameters == pytest.approx(parameters_estimated, rel=1e-3)
    
    # def test_linear_regression_normal_equation(self):
    #     x = 2*np.random.rand(100,1)
    #     eta = np.random.rand(100,1)
    #     y = 4 + 3*x + eta
    #     parameters = np.array([[4.0],[3.0]])
    #     parameters_estimated = self.linear_regression.normal_equation(x, y)
    #     assert parameters == pytest.approx(parameters_estimated, rel=5e-1)