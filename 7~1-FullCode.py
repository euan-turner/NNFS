import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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

##Create dataset
X,y = spiral_data(samples=100,classes=3)

##Dense layer with 2 inputs and 3 outputs
dense1 = Dense_Layer(2,3)
##ReLU activation to be used with first dense layer
activation1 = Act_ReLU()

##Dense layer with 3 inputs and 3 outputs
dense2 = Dense_Layer(3,3)
##Softmax activation to be used with second dense layer
activation2 = Act_Softmax()

##Loss function
loss_func = CCE_Loss()

##Forward pass through first layer
dense1.forward(X)
##Pass through ReLU
activation1.forward(dense1.output)

##Forward pass through second layer
dense2.forward(activation1.output)
##Pass through Softmax
activation2.forward(dense2.output)

##Examine output for first 5 samples
print(activation2.output[:5])

##Calculate loss of network
loss = loss_func.calculate(activation2.output,y)

print("Loss:",loss)