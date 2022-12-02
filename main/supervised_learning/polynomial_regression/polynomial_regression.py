import numpy as np
from utils import messages
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRegression:
    """Class to implement the polynomial regression"""

    def __init__(self):
        self.msg = messages.Messages()

    def polynomial_regression(self, x, y, degree):
        if x.size == 0 or y.size == 0:
            raise Exception(self.msg.EMPTY_ARRAY)
        elif x.size != y.size:
            raise Exception(self.msg.DIFFERENT_SIZE)
        
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly_features.fit_transform(x)
        lin_reg = LinearRegression()
        lin_reg.fit(x_poly, y)
        result = lin_reg.coef_[0].tolist()
        result.reverse()
        for i in lin_reg.intercept_.tolist():
            result.append(i)
        return result