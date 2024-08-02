import numpy as np
from layers import layer

class Activation(layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    




class sigmoid(Activation):
    
    def __init__(self):
        def derv_sigma(x):
            return sigmoid(x)*(1-sigmoid(x))

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        super().__init__(sigmoid, derv_sigma)





class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def derv_tanh(x):
            return 1 - np.tanh(x) ** 2
        super().__init__(tanh, derv_tanh)
