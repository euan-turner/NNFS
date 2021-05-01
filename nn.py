import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

##Layer Classes
class Dense_Layer():

    def __init__(self,n_inputs : int, n_neurons : int, weight_scale_factor : float = 0.01,
                    weight_reg_l1 : int = 0, weight_reg_l2 : int = 0,
                    bias_reg_l1 : int = 0, bias_reg_l2 : int = 0):

        self.weights = weight_scale_factor * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

        ##Set regularisation strength
        self.weight_reg_l1 = weight_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l1 = bias_reg_l1
        self.bias_reg_l2 = bias_reg_l2
        ##Often, only l2 regularisation will be used, as l1 can affect small
        ##values too much
    
    ##Training is redundant, for consistency
    def forward(self, inputs : np.ndarray, training : bool) -> np.ndarray:
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

        ##Gradients on regularisation
        ##L1 on weights
        if self.weight_reg_l1 > 0:
            ##Partial derivative for each weight:
            ##lambda x (-1 < 0, 1 > 0)
            dWL1 = np.ones_like(self.weights)
            dWL1[self.weights < 0] = -1
            self.dWeights += self.weight_reg_l1 * dWL1
        
        ##L2 on weights
        if self.weight_reg_l2 > 0:
            ##Partial derivative for each weight:
            ##2 x lambda x weight
            dWL2 = 2 * self.weight_reg_l2 * self.weights
            self.dWeights += dWL2
        
        ##L1 on biases
        if self.bias_reg_l1 > 0:
            dBL1 = np.ones_likes(self.biases)
            dBL1[self.biases < 0] = -1
            self.dBiases += self.bias_reg_l1 * dBL1
        
        ##L2 on biases
        if self.bias_reg_l2 > 0:
            dBL2 = 2 * self.bias_reg_l2 * self.biases
            self.dBiases += dBL2


##Helps to protect against overfitting and reliance on neurons
##by randomly setting neuron outputs to 0.
##Encourages more neurons to pick up on underlying patterns.
##Will not be used on test data or actual predictions, only in training.
class Dropout_Layer():

    def __init__(self, rate : float):
        ##Rate represents dropout rate
        ##So invert to get success rate
        self.rate = 1 - rate

    ##Training deactivates the dropout layer in testing and use
    def forward(self, inputs : np.ndarray, training : bool):
        self.input = inputs

        if not training:
            self.output = inputs.copy()
            return

        ##Generate mask for output
        ##With scale value to compensate for lower
        ##output with fewer neurons
        self.binary_mask = np.random.binomial(1, self.rate,
                            size = inputs.shape) / self.rate
        ##Apply mask
        self.output = inputs * self.binary_mask
    
    def backward(self, dValues : np.ndarray):
        ##Gradients on input values
        self.dInputs = dValues * self.binary_mask


##Activation Function Classes
class Act_ReLU():

    ##Training is redundant, for consistency
    def forward(self, inputs : np.ndarray, training : bool):
        self.output = np.maximum(0,inputs)
        ##Store inputs for back partial derivative
        self.inputs = inputs
    
    def backward(self, dValues : np.ndarray):
        self.dInputs = dValues.copy()
        self.dInputs[self.inputs <= 0] = 0
    
    ##Calculate predictions for outputs
    def predictions(self, outputs : np.ndarray) -> np.ndarray:
        return outputs


class Act_Sigmoid():

    ##Training is redundant, for consistency
    def forward(self, inputs: np.ndarray, training : bool):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dValues : np.ndarray):
        ##Sigmoid derivative is sigmoid * (1 - sigmoid)
        self.dInputs = dValues * (1 - self.output) * self.output
    
    ##Calculate predictions for output (binary)
    def predictions(self, outputs : np.ndarray) -> np.ndarray:
        return (outputs > 0.5) * 1


class Act_Softmax():

    ##Training is redundant, for consistency
    def forward(self, inputs : np.ndarray, training : bool):
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
    
    ##Calculate predictions for outputs
    def predictions(self, outputs : np.ndarray) -> np.ndarray:
        return np.argmax(outputs, axis = 1)


class Act_Linear():
    ##In reality this has little functionality, 
    ##included for consistency

    ##Training is redundant, for consistency
    def forward(self, inputs : np.ndarray, training : bool):
        self.input = inputs
        self.output = inputs
    
    def backward(self, dValues : np.ndarray):
        self.dInputs = dValues

    ##Calculate predictions for outputs
    def predictions(self, outputs : np.ndarray) -> np.ndarray:
        return outputs


##Loss Classes

##Base class
class Loss():

    ##Remember trainable layers in network model
    def remember_trainable(self, trainable : list):
        self.trainable = trainable

    def calculate(self,output : np.ndarray, target : np.ndarray,
                     *, include_regularisation : bool = False):
        ##Calculate sample losses
        sample_losses = self.forward(output,target)
        ##Mean loss
        mean_loss = np.mean(sample_losses)

        ##Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if include_regularisation == False:
            return mean_loss
        
        return mean_loss, self.regularisation_loss()

    ##Regularisation loss calculation
    ##Standard for all inheriting loss functions
    def regularisation_loss(self) -> float:
        #0 by default
        reg_loss = 0

        ##Iterate over all trainable layers for model
        for layer in self.trainable:
            ##L1 regularisation on weights
            ##lambda x sum of absolute values of weights
            if layer.weight_reg_l1 > 0:
                reg_loss += layer.weight_reg_l1 * np.sum(np.abs(layer.weights))
            
            ##L2 regularisation on weights
            ##lambda x sum of squares of weights
            ##penalises large values more, and small values less
            if layer.weight_reg_l2 > 0:
                reg_loss += layer.weight_reg_l2 * np.sum(layer.weights * layer.weights)
            
            ##L1 regularisation on biases
            if layer.bias_reg_l1 > 0:
                reg_loss += layer.bias_reg_l1 * np.sum(np.abs(layer.biases))
            
            ##L2 regularisation on biases
            if layer.bias_reg_l2 > 0:
                reg_loss += layer.bias_reg_l2 * np.sum(layer.biases * layer.biases)
            
        return reg_loss
    
    ##Calculate accumulated loss
    def calculate_accumulated(self, *, include_regularisation : bool = False):

        ##Calculate mean loss
        mean_loss = self.accumulated_sum / self.accumulated_count

        if include_regularisation == False:
            return mean_loss
        
        return mean_loss, self.regularisation_loss()
    
    ##Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


##Categorical Cross Entropy Loss
##For use when the network is classifying the input
##and looking to have the correct neuron as 1, and others as 0
##Data has class labels
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

        ##Gradients of Categorical Cross-Entropy Loss are:
        ## -(targets/predictions)

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


##Binary Cross Entropy Loss
##For use in binary logistic regression
##where each output neuron predicts one of two characteristics
##e.g. indoors or outdoors, human or not human.
##Data has class labels
class BCE_Loss(Loss):

    def forward(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        ##y_pred is both the predictions, and derivative being passed backwards
        ##Clip data to prevent division by 0
        ##Clip from both sides so mean does not move
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        ##Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                            (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis = -1)

        return sample_losses
    
    def backward(self, y_pred : np.ndarray, y_true : np.ndarray):
        ##y_pred is both the predictions, and derivative being passed backwards
        ##Gradient of BCE is:
        ##  -1/num outputs x(targets/predictions - (1 - targets)/(1-predictions))

        num_samples = len(y_pred)
        num_outputs = len(y_pred[0])

        ##Clip data to prevent division by 0
        ##Clip from both sides so mean does not move
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        self.dInputs = - (y_true / y_pred_clipped - 
                            (1 - y_true) / (1 - y_pred_clipped)) / num_outputs
        ##Normalise wrt number of sampels
        self.dInputs = self.dInputs / num_samples


##Mean Squared Error Loss (L2 Loss)
##For use in regression
##Squares the difference between predictions and targets
##for each individual output, and averages
##Data has value targets, e.g. temperature
class MSE_Loss(Loss):
    '''
    (1 / num. outputs) * sum (targets - predictions)
    '''

    def forward(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        sample_losses = np.mean((y_true - y_pred)**2, axis = -1)
        return sample_losses
    
    def backward(self, y_pred : np.ndarray, y_true: np.ndarray):
        num_samples = len(y_pred)
        num_outputs = len(y_pred[0])

        self.dInputs = -2 * (y_true - y_pred) / num_outputs
        ##Normalise
        self.dInputs = self.dInputs / num_samples


##Mean Absolute Error Loss (L1 Loss)
##For use in regression
##Takes the absolute difference between predictions and targets
##for each individual output, and averages
##Use less frequently than L2 Loss
class MAE_Loss(Loss):
    '''
    (1 / num. outputs) * sum(absolute(targets - predictions))
    '''

    def forward(self, y_pred : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)
        return sample_losses
    
    def backward(self, y_pred : np.ndarray, y_true : np.ndarray):
        num_samples = len(y_pred)
        num_outputs = len(y_pred[0])

        self.dInputs = np.sign(y_true - y_pred) / num_outputs
        ##Normalise
        self.dInputs = self.dInputs / num_samples


##Combining softmax and cce loss for more simplistic back propagation
class Act_Softmax_CCE_Loss():
    '''
    def __init__(self):
        self.act_softmax = Act_Softmax()
        self.cce_loss = CCE_Loss()
        '''

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


##Optimizer classes

##Stochastic Gradient Descent Optimizer
class SGD_Optimizer():

    def __init__(self, learning_rate : float = 1.0, decay : float = 0.0, momentum : float = 0.0):
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

    def update_parameters(self, dense_layer : Dense_Layer):
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

    def __init__(self, learning_rate : float = 1.0, decay : float = 0.0, epsilon : float = 1e-7):
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

    def update_parameters(self, dense_layer : Dense_Layer):

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
    def __init__(self, learning_rate : float = 0.001, decay : float= 0.0, 
        epsilon : float= 1e-7, rho : float= 0.9):

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

    def update_parameters(self, dense_layer : Dense_Layer):

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
    def __init__(self, learning_rate : float = 0.001, decay : float = 0.0, 
        epsilon : float = 1e-7, beta_1 : float = 0.9, beta_2 : float = 0.999):

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
    
    def update_parameters(self, dense_layer : Dense_Layer):
        
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


##Accuracy classes

##Base Class
class Accuracy():

    ##Calculates an accuracy
    ##on predictions and targets
    def calculate(self, preds : np.ndarray, targets : np.ndarray) -> float:
        ##Method will vary depending on type of model
        comparisons = self.compare(preds, targets)
        accuracy = np.mean(comparisons)

        ##Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    
    ##Calculate accumulated accuracy
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    ##Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


##Accuracy for regression model
class Regression_Acc(Accuracy):

    def __init__(self):
        self.precision = None
    
    ##Calculates precision value based on targets
    def init(self, targets : np.ndarray, reinit : bool = False, factor : int = 250):
        if self.precision == None or reinit == True:
            self.precision = np.std(targets) / factor
    
    ##Compares predictions and targets for regression
    def compare(self, preds : np.ndarray, targets : np.ndarray) -> np.ndarray:
        return np.absolute(preds - targets) < self.precision


##Accuracy for classification model
class Classification_Acc(Accuracy):

    def __init__(self, *, binary = False):
        ##Flag for binary classification (disables one-hot to sparse label conversion)
        self.binary = binary
    
    ##Only needed for consistency in model
    def init(self, y):
        pass

    ##Compares predictions to targets
    def compare(self, preds : np.ndarray, targets : np.ndarray) -> np.ndarray:
        if self.binary == False and len(targets.shape) == 2:
            ##Convert from one-hot to sparse labels
            targets = np.argmax(targets, axis = 1)
        return preds == targets


##Turning the whole network into a model

##Input layer for consistency in model
##has no backward as will not actually be trained in the network
class Input_Layer():

    ##Training is only for consistency
    def forward(self, inputs : np.ndarray, training : bool):
        self.output = inputs


##Model class to contain all parts of the neural network
class Model():

    def __init__(self):
        ##Will contain layer and activation objects
        self.layers = []
        ##Softmax classifier's output object
        self.softmax_classifier_output = None
    
    ##Add layer or activation object to model
    def add(self, obj):
        self.layers.append(obj)
    
    ##Set loss and optimizer objects
    def set(self, *, loss, optimizer, accuracy):
        self.loss_func = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    ##Finalise the model
    def finalise(self):
        ##Create input layer
        self.input_layer = Input_Layer()

        ##Number of objects in model
        obj_count = len(self.layers)

        ##Identify layers, not activations
        self.trainable = []

        ##Set up prev and next points (like a doubly linked list)
        for i in range(obj_count):

            ##First layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            ##Except last layer
            elif i < obj_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            ##Last layer
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss_func
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable.append(self.layers[i])
        
        self.loss_func.remember_trainable(self.trainable)

        ##Check if Softmax and CCE loss can be combined
        ##to speed up backward pass
        if isinstance(self.layers[-1], Act_Softmax) and \
           isinstance(self.loss_func, CCE_Loss):
           ##Combined softmax and cce function
           self.softmax_classifier_output = Act_Softmax_CCE_Loss()

    ##Train the model
    def train(self, X : np.ndarray, y : np.ndarray, batch_size = None,
                epochs : int = 1, print_every : int = 1, test_data : np.ndarray = None):
        print("Training")
        ##Set accuracy precision
        self.accuracy.init(y)

        ##Default value if batch size is not being set
        train_steps = 1

        ##If test data exists, set default step size
        if test_data != None:
            test_steps = 1

            ##Unpack
            X_test, y_test = test_data
        
        ##Calculate numbers of steps per epoch if needed
        if batch_size != None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if test_data != None:
                test_steps = len(X_test) // batch_size

                if test_steps * batch_size < len(X_test):
                    test_steps += 1

        ##Main training loop
        for epoch in range(1, epochs+1):
            
            ##Epoch number
            print(f'Epoch {epoch}')

            ##Reset accumulated values in loss and accuracy objects
            self.loss_func.new_pass()
            self.accuracy.new_pass()

            ##Iterate over steps
            for step in range(train_steps):

                ##If batch size is not set
                ##train in one step
                if batch_size == None:
                    batch_X = X
                    batch_y = y
                ##Otherwise slice batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                ##Forward pass
                output = self.forward(batch_X, training = True)

                ##Calculate loss
                data_loss, reg_loss = self.loss_func.calculate(output, batch_y, include_regularisation = True)
                loss = data_loss + reg_loss

                ##Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                ##Backward pass
                self.backward(output, batch_y)

                ##Optimise parameters
                self.optimizer.pre_update()
                for layer in self.trainable:
                    self.optimizer.update_parameters(layer)
                self.optimizer.post_update()

                ##Step summary
                if step%print_every == 0 or step == train_steps - 1:
                    print(f'step: {step}, '+
                        f'acc: {accuracy:.3f}, '+
                        f'loss: {loss:.3f}, ('+
                        f'data_loss: {data_loss:.5f}, '+
                        f'reg_loss: {reg_loss:5f}), '+
                        f'lr: {self.optimizer.current_learning_rate:.10f}'
                    )
                
            ##Epoch loss and accuracy
            epoch_data_loss, epoch_reg_loss = self.loss_func.calculate_accumulated(
                include_regularisation = True
            )
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            ##Epoch summary
            print(f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f}, ' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_reg_loss:.3f}, ' +
                  f'lr: {self.optimizer.current_learning_rate}'
            )
        
        ##Test the model
        if test_data != None:

            ##Evaluate the model
            self.evaluate(*test_data, batch_size = batch_size)


    ##Forward pass on all objects
    def forward(self, X : np.ndarray, training : bool):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output
    
    ##Backward pass on all objectes
    def backward(self, output : np.ndarray, targets : np.ndarray):
        ##If using softmax and cce loss for classification
        if self.softmax_classifier_output != None:
            ##Backward method on combined object
            self.softmax_classifier_output.backward(output, targets)
            ##Set dInputs for softmax in self.layers
            self.layers[-1].dInputs = self.softmax_classifier_output.dInputs
            ##Backward pass on all object except last softmax activation
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dInputs)
            return
        self.loss_func.backward(output, targets)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dInputs)
    
    ##Evaluate the model
    def evaluate(self, X_val : np.ndarray, y_val : np.ndarray, *, batch_size = None):
        ##Calculate number of steps
        test_steps = 1

        if batch_size != None:
            test_steps = len(X_val) // batch_size
            ##Step for remaining data
            if test_steps * batch_size < len(X_val):
                test_steps += 1
        
        ##Reset accumulated values in loss and accuracy objects
        self.loss_func.new_pass()
        self.accuracy.new_pass()

        for step in range(test_steps):
            ##If batch size is not set
            ##test using full dataset
            if batch_size == None:
                batch_X = X_val
                batch_y = y_val

            ##Otherwise slice batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            
            ##Forward pass
            output = self.forward(batch_X, training = False)

            ##Calculate loss
            loss = self.loss_func.calculate(output, batch_y)

            ##Predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
        
        ##Accumulated loss and accuracy
        test_loss = self.loss_func.calculate_accumulated()
        test_accuracy = self.accuracy.calculate_accumulated()

        ##Summary
        print(f'Validation: ' +
                f'acc: {test_accuracy:.3f}, ' +
                f'loss: {test_loss:.3f}'
        )

        
    
