# Package imports
import numpy as np

# GRADED FUNCTION: sigmoid
class Sigmoid:
    
    def sigmoid(X):
        Y = 1 / (1 + np.exp(-X))
        return Y