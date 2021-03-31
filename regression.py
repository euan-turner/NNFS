##Regression model
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
from nn import *

nnfs.init()

##Create dataset
X, y = sine_data()

##Dense layer - 1 input, 64 outputs
dense1 = Dense_Layer(1,64, weight_scale_factor = 0.1)
##ReLU for first layer
activation1 = Act_ReLU()

##Dense layer - 64 inputs, 64 outputs
dense2 = Dense_Layer(64,64,  weight_scale_factor = 0.1)
activation2 = Act_ReLU()

##Dense layer - 64 inputs, 1 output
dense3 = Dense_Layer(64,1,  weight_scale_factor = 0.1)
##Linear activation for output - regression
activation3 = Act_Linear()

##MSE (L2) loss function
loss_func = MSE_Loss()

##Adaptive Momentum optimizer
optimizer = Adam_Optimizer(learning_rate = 0.005, decay = 1e-3)

##Value to use as gauge for accuracy compared to targets
acc_precision = np.std(y) / 250

for epoch in range(10001):
    ##Forward passes
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    ##Losses
    data_loss = loss_func.calculate(activation2.output, y)
    reg_loss = loss_func.regularisation_loss(dense1) + loss_func.regularisation_loss(dense2)
    loss = data_loss + reg_loss

    ##Calculate accuracy by taking absolute difference
    ##between predictions and targets
    ##and compare if within accuracy precision
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < acc_precision)

    if epoch%100 == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {reg_loss:.3f}, ' +
              f'lrate: {optimizer.current_learning_rate:.10f}')

    ##Backward passes
    loss_func.backward(activation3.output, y)
    activation3.backward(loss_func.dInputs)
    dense3.backward(activation3.dInputs)
    activation2.backward(dense3.dInputs)
    dense2.backward(activation2.dInputs)
    activation1.backward(dense2.dInputs)
    dense1.backward(activation1.dInputs)

    ##Optimisation
    optimizer.pre_update()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.update_parameters(dense3)
    optimizer.post_update()

##Validation
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()