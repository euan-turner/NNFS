import numpy as np

##Batches allow computation in parallel - utilising GPUs
##They also allow for generalisation of the algorithm
##Input matrices are now batches, with each row representing one set of inputs

##First layer - 3 neurons (3 rows of weights)- 4 inputs per neuron (4 columns per row)
inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

  
weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

##Transpose weights array - columns become rows, rows become columns
##So that shapes align for dot product
weights = np.array(weights).T
           
biases = [2,3,0.5]

##Second layer - inputs are first layer outputs - 3 neurons(3 rows of weights) - 3 inputs per neuron(3 columns per row)
weights2 = [[0.1,-0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]
weights2 = np.array(weights2).T

biases2 = [-1,2,-0.5]


layer1_outputs = np.dot(inputs,weights) + biases
layer2_outputs = np.dot(layer1_outputs,weights2) + biases2
print(layer1_outputs)
print(layer2_outputs)
