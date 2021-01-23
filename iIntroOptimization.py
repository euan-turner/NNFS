import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import vertical_data, spiral_data
from gFullCode import Dense_Layer, Act_ReLU, Act_Softmax, CCE_Loss
nnfs.init()

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

