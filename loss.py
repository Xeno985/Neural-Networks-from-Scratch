import numpy as np

class Quadloss():
    
    def loss(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def grad_loss(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    
class CrossEntropy():
    
    def loss(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def grad_loss(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)