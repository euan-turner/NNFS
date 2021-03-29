import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Dense_Layer():

    def __init__(self,n_inputs : int, n_neurons : int):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs : np.ndarray) -> np.ndarray:
        ##Forward propagate value through layer
        self.output = np.dot(inputs,self.weights) + self.biases
        ##Store inputs for use in partial derivative
        self.inputs = inputs
    
    def backward(self, dValues : np.ndarray):
        ##Gradients on weights and biases
        self.dWeights = np.dot(self.inputs.T, dValues)
        self.dBiases = np.sum(dValues, axis=0, keepdims=True)
        ##Gradients on inputs values
        self.dInputs = np.dot(dValues, self.weights.T)


class Act_ReLU():

    def forward(self, inputs : np.ndarray) -> np.ndarray:
        self.output = np.maximum(0,inputs)
        ##Store inputs for back partial derivative
        self.inputs = inputs
    
    def backward(self, dValues : np.ndarray):
        self.dInputs = dValues.copy()
        self.dInputs[self.inputs <= 0] = 0


class Loss():

    def calculate(self,output : np.ndarray, target : np.ndarray) -> int:
        sample_losses = self.forward(output,target)
        mean_loss = np.mean(sample_losses)
        return mean_loss


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

    def backward(self, y_pred : np.ndarray, y_true : np.ndarray):

        '''
        Gradients of Categorical Cross-Entropy Loss are:
        -(targets/predictions)
        '''

        ##Number of samples
        samples = len(y_pred)
        

        ##If target labels are sparse, convert into one-hot encoded vectors
        if len(y_true.shape) == 1:
            ##Number of labels per samples
            labels = len(y_pred[0])
            y_true = np.eye(labels)[y_true]
        
        ##Gradients on predictions
        self.dInputs = -(y_true/y_pred)
        ##Normalize gradients - prevents excessively large gradients during optimisation
        self.dInputs = self.dInputs / samples


class Act_Softmax():

    def forward(self, inputs : np.ndarray) -> np.ndarray:
        ##Processing batch values, hence keepdims
        ##Make every value negative prior to exponentiation, so range is 0 to 1
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        ##Normalise values into probability distribution
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)

        self.output = probabilities

    def backward(self, dValues : np.ndarray):

        '''
        Gradient of Softmax is:
        ∂ S[i,j] / ∂ z[i,k] = S[i,j].(δ[j,k] - S[i,k])
                            = S[i,j]δ[j,k] - S[i,j]S[i,k]

        S[i,j] = j-th output of i-th sample
        z[i,k] = k-th input of i-th sample

        Every inputs affects every output due to normalisation, so the derivative wrt to an input
        returns a vector of partial derivatives.
        These vectors form rows in the Jacobian matrix.
        There is a Jacobian matrix for each set of input values in the batch
        '''

        ##Create uninitilazed array
        self.dInputs = np.empty_like(dValues)

        ##Enumerate (output,gradient) values with an index
        for index, (single_output,single_dValues) \
            in enumerate(zip(self.output,dValues)):
                ##Flatten single_output into a list of single-item lists
                single_output = single_output.reshape(-1,1)
                ##Calculate Jacobian matrix
                jacobian_matrix = np.diagflat(single_output) - \
                                np.dot(single_output, single_output.T)

                ##This line is equivalent to S[i,j]δ[j,k] - S[i,j]S[i,k]
                
                ##Calculate sample-wise gradients for the batch
                self.dInputs[index] = np.dot(jacobian_matrix, single_dValues)


##Combining softmax and cce loss for more simplistic back propagation
class Act_Softmax_CCE_Loss():

    def __init__(self):
        self.act_softmax = Act_Softmax()
        self.cce_loss = CCE_Loss()

    def forward(self, inputs : np.ndarray, y_true : np.ndarray) -> int:
        self.act_softmax.forward(inputs)
        self.output = self.act_softmax.output ##Model's prediction values
        ##Calculate and return loss value
        return self.cce_loss.calculate(self.output, y_true)

    def backward(self, y_pred, y_true):

        '''
        Derivative for the combined softmax and cce loss functions is:
        predicted values - target values
        '''
        ##Number of samples
        samples = len(y_pred)

        ##If targets are one-hot encoded, convert to sparse labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        ##Calculate derivative wrt inputs
        self.dInputs = y_pred.copy()
        self.dInputs[range(samples), y_true] -= 1 ##Subtract one (target value) from the target index for each input
        ##Normalize
        self.dInputs = self.dInputs/samples


##Stochastic Gradient Descent Optimizer
class SGD_Optimizer():

    def __init__(self, learning_rate = 1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum

    ##Call before updating params, sets the learning rate for current epoch
    def pre_update(self):
        if self.decay:
            ##Decaying learning rate over epochs should help to find deeper minima
            ##As the model can escape lower minima at the start, and then focus on 
            ##Deeper minima at the end
            self.current_learning_rate = \
                self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, dense_layer):
        ##If using momentum
        if self.momentum:

            ##Check if layers have momentum
            if not hasattr(dense_layer, 'weight_momentums'):
                dense_layer.weight_momentums = np.zeros_like(dense_layer.weights)
                dense_layer.bias_momentums = np.zeros_like(dense_layer.biases)

            ##Build weight updates with momentum
            weight_updates = self.momentum * dense_layer.weight_momentums - \
                            self.current_learning_rate * dense_layer.dWeights
            dense_layer.weight_momentums = weight_updates
            ##Build bias updates with momentum
            bias_updates = self.momentum * dense_layer.bias_momentums - \
                            self.current_learning_rate * dense_layer.dBiases
            dense_layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * dense_layer.dWeights
            bias_updates = - self.current_learning_rate * dense_layer.dBiases

        dense_layer.weights += weight_updates
        dense_layer.biases += bias_updates

    def post_update(self):
        self.iterations += 1


##Adaptive Gradient Optimizer
class AdaGrad_Optimizer():

    def __init__(self, learning_rate = 1.0, decay = 0.0, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon

    ##Call before updating params, sets the learning rate for current epoch
    def pre_update(self):
        if self.decay:
            ##Decaying learning rate over epochs should help to find deeper minima
            ##As the model can escape lower minima at the start, and then focus on 
            ##Deeper minima at the end
            self.current_learning_rate = \
                self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, dense_layer):

        ##If layer does not contain array cache, initialise with 0s
        if not hasattr(dense_layer, 'weight_cache'):
            dense_layer.weight_cache = np.zeros_like(dense_layer.weights)
            dense_layer.bias_cache = np.zeros_like(dense_layer.biases)

        ##Update cache with the square of current gradients - lose negatives
        dense_layer.weight_cache += dense_layer.dWeights**2
        dense_layer.bias_cache += dense_layer.dBiases**2

        ##Update weights and biases
        ##(Learning rate x Gradient )/(square root of cache + epsilon)
        dense_layer.weights += - self.current_learning_rate * dense_layer.dWeights / \
                                (np.sqrt(dense_layer.weight_cache) + self.epsilon)
        dense_layer.biases += - self.current_learning_rate * dense_layer.dBiases / \
                                (np.sqrt(dense_layer.bias_cache) + self.epsilon)

    def post_update(self):
        self.iterations += 1


##Root Mean Square Propagation Optimizer
class RMSProp_Optimizer():

    ##Initial learning rate is much lower than in SGD or AdaGrad
    ##rho is the decay rate of the cache 
    def __init__(self, learning_rate = 0.001, decay = 0.0, epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    ##Call before updating params, sets the learning rate for current epoch
    def pre_update(self):
        if self.decay:
            ##Decaying learning rate over epochs should help to find deeper minima
            ##As the model can escape lower minima at the start, and then focus on 
            ##Deeper minima at the end
            self.current_learning_rate = \
                self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, dense_layer):

        ##If layer does not contain cace, initialise with 0s
        if not hasattr(dense_layer, 'weight_cache'):
            dense_layer.weight_cache = np.zeros_like(dense_layer.weights)
            dense_layer.bias_cache = np.zeros_like(dense_layer.biases)

        ##Update cache with square of current gradients
        ##Cache decay rate x cache + (1 - cache decay rate) x gradient^2
        dense_layer.weight_cache = self.rho * dense_layer.weight_cache + \
                                   (1 - self.rho) * dense_layer.dWeights**2
        dense_layer.bias_cache = self.rho * dense_layer.bias_cache + \
                                   (1 - self.rho) * dense_layer.dBiases**2

        ##Update weights and biases with normalised changes
        ##(Learning rate x Gradient )/(square root of cache + epsilon)
        dense_layer.weights += -self.current_learning_rate * dense_layer.dWeights / \
                               (np.sqrt(dense_layer.weight_cache) + self.epsilon)
        dense_layer.biases += -self.current_learning_rate * dense_layer.dBiases / \
                               (np.sqrt(dense_layer.bias_cache) + self.epsilon)

    def post_update(self):
        self.iterations += 1


##Adaptive Momentum Optimizer
##Combines momentum from SGD with RMSProp
class Adam_Optimizer():

    ##Initial learning rate is much lower than in SGD or AdaGrad
    ##rho is the decay rate of the cache 
    def __init__(self, learning_rate = 0.001, decay = 0.0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2 ##beta_2 is equivalent to rho in RMSProp
    
    ##Call before updating params, sets the learning rate for current epoch
    def pre_update(self):
        if self.decay:
            ##Decaying learning rate over epochs should help to find deeper minima
            ##As the model can escape lower minima at the start, and then focus on 
            ##Deeper minima at the end
            self.current_learning_rate = \
                self.learning_rate * (1 / (1 + self.decay * self.iterations))
    
    def update_parameters(self, dense_layer):
        
        ##If layer does not contain cache, initialise with 0s
        if not hasattr(dense_layer, 'weight_cache'):
            dense_layer.weight_momentums = np.zeros_like(dense_layer.weights)
            dense_layer.weight_cache = np.zeros_like(dense_layer.weights)
            dense_layer.bias_momentums = np.zeros_like(dense_layer.biases)
            dense_layer.bias_cache = np.zeros_like(dense_layer.biases)

        ##Update momentums with current gradients
        dense_layer.weight_momentums = self.beta_1 * dense_layer.weight_momentums + \
                                        (1 - self.beta_1) * dense_layer.dWeights
        dense_layer.bias_momentums = self.beta_1 * dense_layer.bias_momentums + \
                                        (1 - self.beta_1) * dense_layer.dBiases

        ##Get corrected momentums
        ##Adjusts self.iterations as first pass will be at 0
        weight_momentums_corrected = dense_layer.weight_momentums / \
                                        (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = dense_layer.bias_momentums / \
                                        (1 - self.beta_1 ** (self.iterations + 1))
        
        ##Update cache with squared gradients
        dense_layer.weight_cache = self.beta_2 * dense_layer.weight_cache +\
                                    (1 - self.beta_2) * dense_layer.dWeights**2
        dense_layer.bias_cache = self.beta_2 * dense_layer.bias_cache + \
                                    (1 - self.beta_2) * dense_layer.dBiases**2
        
        ##Get corrected cache
        weight_cache_corrected = dense_layer.weight_cache / \
                                    (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = dense_layer.bias_cache / \
                                    (1 - self.beta_2 ** (self.iterations + 1))
        
        ##Update weights and biases with normalised changes
        ##(Learning rate x Corrected momentums)/(square root of  corrected cache + epsilon)
        dense_layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                                (np.sqrt(weight_cache_corrected) + self.epsilon)
        dense_layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                                (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1

