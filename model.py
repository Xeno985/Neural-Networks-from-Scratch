from layers import layer,dense,ReLU
from activation import sigmoid,Tanh
from loss import Quadloss,CrossEntropy
from conv2d import Convolutional

class sequential():
   
    def __init__(self,arch,loss_fn):
        self.architecture=arch        
        self.Loss_func=loss_fn
        
    def predict(self, input):
        output = input
        for layer in self.architecture:
            output = layer.forward(output)
        return output
    
    def train(self, x_train, y_train, epochs = 1000, verbose = True):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
            
                output = self.predict(x)

            
                error += self.Loss_func.loss(y, output)

            
                grad = self.Loss_func.grad_loss(y, output)
                for layer in reversed(self.architecture):
                    grad = layer.backward(grad)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")# see progress