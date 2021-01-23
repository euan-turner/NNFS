import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import vertical_data, spiral_data

nnfs.init()

##Dense Layer Class
class Dense_Layer():

    def __init__(self, n_inputs : int, n_neurons : int):
        ##Initialise random weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs : np.ndarray) -> np.ndarray:
        ##Forward propagate values through layer
        self.output = np.dot(inputs,self.weights) + self.biases


##ReLU Activation Class
class Act_ReLU():

    def forward(self,inputs : np.ndarray) -> np.ndarray:
        self.output = np.maximum(0,inputs)


##Softmax Activation Class
class Act_Softmax():

    def forward(self,inputs : np.ndarray) -> np.ndarray:
        ##Processing batch values, hence keepdims
        ##Make every value negative prior to exponentiation, so range is 0 to 1
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        ##Normalise values into probability distribution
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)

        self.output = probabilities

##Base Loss Class
class Loss():

    def calculate(self,output : np.ndarray, target : np.ndarray) -> int:
        sample_losses = self.forward(output,target)
        mean_loss = np.mean(sample_losses)
        return mean_loss

##Cross-Entropy Loss Class
class CCE_Loss(Loss):

    def forward(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        num_samples = len(y_pred)
        ##Clip data to prevent division by 0
        ##Clip from both sides so mean does not move
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        ##Isolate probabilites for target values
        if len(y_true.shape) == 1:
            ##Categorical labels point to index of target in each input
            confs = y_pred_clip[range(num_samples),y_true]
        elif len(y_true.shape) == 2:
            ##One-hot encoded labels for targets
            confs = np.sum(y_pred_clip*y_true, axis=1)
        
        neg_log = -np.log(confs)
        return neg_log

        
##X,y = vertical_data(samples=100,classes=3)
##plt.scatter(X[:,0],X[:,1], c=y, s=40, cmap='brg')
##plt.show()
##Model learns well for the simple, vertical data
X,y= spiral_data(samples=100,classes=3)


##Create model
dense1 = Dense_Layer(2,3)
act1 = Act_ReLU()
dense2 = Dense_Layer(3,3)
act2 = Act_Softmax()

loss_func = CCE_Loss()

##Helper variables to track best loss, and associated weights and biases
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):

    ##Generate new, random weights and biases for each iteration
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense1.biases += 0.05 * np.random.randn(1,3)

    ##Forward pass of data
    dense1.forward(X)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    act2.forward(dense2.output)

    ##Calculate loss for model
    loss = loss_func.calculate(act2.output,y)

    ##Calculate accuracy for model
    predictions = np.argmax(act2.output, axis=1)
    accuracy = np.mean(predictions==y)

    ##If loss is smaller than lowest loss, print and save weights and biases
    if loss < lowest_loss:
        print("New lowest loss on iteration:", iteration, "\tloss:", loss, "\tacc:",accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    ##Revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

