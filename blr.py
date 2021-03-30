##Binary Logistic Regression - Chapter 16
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nn import *

nnfs.init()

##Create training dataset
X, y = spiral_data(samples = 100, classes = 2)
##Reshape labels with inner lists
y = y.reshape(-1,1)

dense1 = Dense_Layer(2, 64, weight_reg_l2= 5e-4, bias_reg_l2 = 5e-4)
activation1 = Act_ReLU()

##1 output per neuron - binary classification
dense2 = Dense_Layer(64,1)
activation2 = Act_Sigmoid()

loss_func = BCE_Loss()
optimizer = Adam_Optimizer(decay = 5e-7)

for epoch in range(10001):

    ##Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_func.calculate(activation2.output, y)
    reg_loss = loss_func.regularisation_loss(dense1) + loss_func.regularisation_loss(dense2)

    loss = data_loss + reg_loss

    ##Outputs greater than 0.5 become 1, less become 0
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if epoch%100 == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {reg_loss:.3f}, ' +
              f'lrate: {optimizer.current_learning_rate:.10f}')
    
    ##Backward pass
    loss_func.backward(activation2.output, y)
    activation2.backward(loss_func.dInputs)
    dense2.backward(activation2.dInputs)
    activation1.backward(dense2.dInputs)
    dense1.backward(activation1.dInputs)

    ##Update weights and biases
    optimizer.pre_update()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update()

##Validate model
X_test, y_test = spiral_data(samples = 100, classes = 2)
y_test = y_test.reshape(-1,1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_func.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'Validation, Acc: {accuracy:.3f}, loss: {loss:.3f}')

