##Binary Logistic Regression -Model
import nnfs
from nnfs.datasets import spiral_data
from nn import *

##Training and test datasets
X, y = spiral_data(samples = 100, classes = 2)
X_test, y_test = spiral_data(samples = 100, classes = 2)

##Reshape labels into list of one-item lists
y = y.reshape(-1,1)
y_test = y_test.reshape(-1,1)

##Instantiate model
model = Model()

##Add layers
model.add(Dense_Layer(2, 64, weight_reg_l2 = 5e-4, bias_reg_l2 = 5e-4))
model.add(Act_ReLU())
model.add(Dense_Layer(64, 1))
model.add(Act_Sigmoid())

##Set loss, optimizer and accuracy
model.setup(
    loss = BCE_Loss(),
    optimizer = Adam_Optimizer(decay = 5e-7),
    accuracy = Classification_Acc(binary = True)
)

##Finalize set-up
model.finalise()

##Train
model.train(X, y, test_data = (X_test, y_test), epochs = 10000, print_every = 100)
