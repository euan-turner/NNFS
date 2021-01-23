import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

##Class defining a single layer, with parameteres for number of inputs and number of neurons
class Layer_Dense:   
    def __init__(self,n_inputs,n_neurons):
        ##Initialise random weights of shape n_inputs * n_neurons, shrink around 0
        ##Shape of inputs * neurons 'transposes' at creation
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

##ReLU object
class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

##Softmax object
class Activation_Softmax():
    def forward(self,inputs):
        ##Make every value negative, to normalise range between 0 and 1 - this protects against math errors for very high inputs
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        ##Normalise values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100,classes=3)

##Input layer - 2 inputs per example - 3 neurons in next layer
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

##Output layer - 3 inputs - 3 outputs (3 classes)
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])






