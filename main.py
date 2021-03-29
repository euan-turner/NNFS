import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nn import *

nnfs.init()

##Create dataset
X, y = spiral_data(samples = 100, classes = 3)

##Dense layer - 2 inputs, 64 outputs
dense1 = Dense_Layer(2,64)
##ReLU activation to be used with first dense layer
activation1 = Act_ReLU()

##Dense layer with 64 inputs and 3 outputs
dense2 = Dense_Layer(64,3)

##Softmax/Loss functions
act_loss = Act_Softmax_CCE_Loss()

##Optimizer
##optimizer = SGD_Optimizer(decay = 1e-3, momentum = 0.9)
##optimizer = AdaGrad_Optimizer(decay = 1e-4)
##optimizer = RMSProp_Optimizer(decay = 1e-4)
optimizer = Adam_Optimizer(learning_rate = 0.05, decay = 5e-7)

for epoch in range(10001):
    ##Forward passes
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = act_loss.forward(dense2.output, y)

    ##Calculate accuracy of predictions form output of act_loss
    preds = np.argmax(act_loss.output, axis = 1)
    if len(y.shape) == 2:
        ##Convert from one-hot to index labels
        y = np.argmax(y, axis=1)

    acc = np.mean(preds == y)
    
    if epoch%100 == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {acc:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lrate: {optimizer.current_learning_rate}')

    ##Backward passes
    act_loss.backward(act_loss.output, y)
    dense2.backward(act_loss.dInputs)
    activation1.backward(dense2.dInputs)
    dense1.backward(activation1.dInputs)

    #Update weights and biases
    optimizer.pre_update()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update()


##Validate the model on test data

##Create test dataset
X_test, y_test = spiral_data(samples = 100, classes = 3)

dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = act_loss.forward(dense2.output, y_test)

##Calculate accuracy
predictions = np.argmax(act_loss.output, axis = 1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis = 1)
accuracy = np.mean(predictions == y_test)

print(f'Validation, Acc: {accuracy:.3f}, loss: {loss:.3f}')



##Test data should not be used to tune hyper-parameters
##A portion of data can be set aside as validation data
##to tune them.
##Or k-fold tuning can be used, to tune on chunks of the 
##training data at a time.