##Classification Model
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nn import *

nnfs.init()

##Create training and test datasets
X, y = spiral_data(samples = 1000, classes = 3)
X_test, y_test = spiral_data(samples = 100, classes = 3)

##Instantiate model
model = Model()

##Add layers
model.add(Dense_Layer(2,512, weight_reg_l2 = 5e-4, bias_reg_l2 = 5e-4))
model.add(Act_ReLU())
model.add(Dropout_Layer(0.1))
model.add(Dense_Layer(512,3))
model.add(Act_Softmax())

##Set loss, optimizer and accuracy
model.setup(
    loss = CCE_Loss(),
    optimizer = Adam_Optimizer(learning_rate = 0.05, decay = 5e-5),
    accuracy = Classification_Acc()
)

##Finalise model
model.finalise()
##Train model
model.train(X,y, epochs = 10000, print_every = 100,
                test_data = (X_test, y_test))



##Test data should not be used to tune hyper-parameters
##A portion of data can be set aside as validation data
##to tune them.
##Or k-fold tuning can be used, to tune on chunks of the 
##training data at a time.