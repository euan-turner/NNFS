import numpy as np

np.random.seed(0)

##Batch of 3 sets of inputs
X = [[1.0,2.0,3.0,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

##Class defining a single layer, with parameteres for number of inputs and number of neurons
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):   
        ##Initialise random weights of shape n_inputs * n_neurons, shrink around 0
        ##Shape of inputs * neurons 'transposes' at creation
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

##4 inputs per set from batch X
layer1 = Layer_Dense(4,5)
##5 neurons in layer means 5 outputs
layer2 = Layer_Dense(5,2)

##Forward pass through layer 1 with batch X
layer1.forward(X)
##Forward pass through layer 2 with batch layer1.output
layer2.forward(layer1.output)

print(layer1.output)
print(layer2.output)
