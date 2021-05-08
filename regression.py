##Regression Model
import nnfs
from nnfs.datasets import sine_data
from nn import *

##Create dataset
X,y = sine_data()

##Instantiate model
model = Model()

##Add layers
model.add(Dense_Layer(1,64))
model.add(Act_ReLU())
model.add(Dense_Layer(64,64))
model.add(Act_ReLU())
model.add(Dense_Layer(64,1))
model.add(Act_Linear())

##Set loss, optimizer and accuracy objects
model.setup(
    loss = MSE_Loss(),
    optimizer = Adam_Optimizer(learning_rate= 0.005, decay = 1e-3),
    accuracy = Regression_Acc()
)

##Finalise
model.finalise()

##Train
model.train(X, y, epochs = 10000, print_every = 100)