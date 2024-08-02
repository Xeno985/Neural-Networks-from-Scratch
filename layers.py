import numpy as np
def genweight(i,j):
    return np.random.rand(j,i)
def genbias(j):
    return np.random.rand(j,1)
global learning_rate 


class layer:
    
    def __init__(self,inputs):
        
        self.layers = inputs
        self.input = None
        self.output = None

    def forward(self,inputs):
        raise NotImplementedError
    def backward():
        raise NotImplementedError
class dense(layer):
    W=None
    B=None
    input=None
    def __init__(self,i,j):
        self.W=genweight(i,j)
        self.B=genbias(j)
    
    def forward(self,input_f):
        self.input=input_f
        return np.dot(self.W,input_f)+self.B
    
    def backward(self, input_b):
        learning_rate= 0.01
        weights_gradient = np.dot( input_b,self.input.T)
        input_gradient = np.dot( self.W.T,input_b)
        self.W -= learning_rate * weights_gradient
        self.B -= learning_rate * input_b
        return input_gradient
class ReLU(layer):
      def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
      def backward(self, dvalues):
       
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs