import numpy as np
from utils import messages

class LinearRegression:
    """Class to linear regression algorithms"""

    def __init__(self):
        self.msg = messages.Messages()

    def normal_equation(self, x, y):
        if x.size == 0 or y.size == 0:
            raise Exception(self.msg.EMPTY_ARRAY)
        elif x.size != y.size:
            raise Exception(self.msg.DIFFERENT_SIZE)
        
        N = x.size
        Xb = np.c_[np.ones((N,1)), x]
        theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)

        return theta