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

##ReLU object
class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

##Dataset function
def create_data(points,classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0,1,points) #radius
        t = np.linspace(class_number*4, (class_number+1)*4, points)+np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5),r*np.cos(t*2.5)]
        y[ix] = class_number
    return X,y

##100 feature sets(100 points in the spiral) of 3 classes(3 'arms')
##There are two features per set(co-ordinates) and the classes are the classifying target
X, y = create_data(100,3)

##Two inputs per example
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print("~~~~~~~~~~~~~~~~~~~~~~~~~")
print(activation1.output)
