##One layer - 3 neurons - 4 inputs per neuron
layerinputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
           
biases = [2,3,0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights,biases): ##List of weights/bias pairs for each neuron, generated sequentially
    neuron_output = 0
    for n_input,weight in zip(inputs,neuron_weights): ##List of input/weight pairs for an individual neuron
        neuron_output+= n_input*weight
    neuron_output += neuron_bias  ##Add bias to total value of input*weight sum
    layer_outputs.append(neuron_output) 

    
print(layer_outputs)   
