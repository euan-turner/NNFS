import numpy as np

##Passed in gradients from next layer, assuming easy values
dvalues = np.array([[1,1,1],[2,2,2],[3,3,3]])

##3 input samples
inputs = np.array([[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]])

##3 sets of 4 weights
weights = np.array([[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]).T

##One bias per neuron
biases = np.array([[2,3,0.5]])

##Forward pass
layer_outputs = np.dot(inputs,weights) + biases
relu_outputs = np.maximum(0,layer_outputs)

##Derivative of ReLU
dRelu = relu_outputs.copy()
dRelu[layer_outputs <= 0] = 0

##Derivative of layer wrt inputs, multiply by weights -> "averages across batch"
dInputs = np.dot(dRelu, weights.T)

##Derivative of layer wrt weights, multiply by inputs -> "averages across batch"
dWeights = np.dot(inputs.T, dRelu)

##Derivative of layer wrt biases, sum over first axis -> "averages across batch"
dBiases = np.sum(dRelu, axis=0, keepdims=True)

print(dInputs)
print(dWeights)
print(dBiases)

##Update parameters - not targeting anything, just illustrating
weights += -0.001 * dWeights
biases += -0.001 * dBiases

new_outputs = np.maximum(0,np.dot(inputs,weights) + biases)
print(relu_outputs,"\n",new_outputs)
