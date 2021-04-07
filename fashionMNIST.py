import numpy as np
import cv2
import os
import nnfs
from nn import *

nnfs.init()

##Load dataset
def load_mnist_dataset(dataset, path):
    ##Scan directory and make list of labels
    labels = os.listdir(os.path.join(path, dataset))

    ##Samples and labels
    X = []
    y = []

    ##Iterate over class labels
    for label in labels:
        ##Iterate over images
        for file in os.listdir(os.path.join(path, dataset, label)):
            ##Read image
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
                )
            
            X.append(image)
            y.append(label)
    
    ##Convert to numpy arrays
    return np.array(X), np.array(y).astype('uint8')

##Create train and test data
def create_mnist_data(path):
    ##Load train data
    X, y = load_mnist_dataset('train', path)
    ##Load test data
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


##Create dataset
X, y, X_test, y_test = create_mnist_data('fashion_mnist_images')

##Scale data between -1 and 1
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

##Reshape to 1D vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

##Shuffle train data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

##Instantiate model
model = Model()

##Add layers
model.add(Dense_Layer(X.shape[1], 64))
model.add(Act_ReLU())
model.add(Dense_Layer(64,64))
model.add(Act_ReLU())
model.add(Dense_Layer(64,10))
model.add(Act_Softmax())

##Add loss and optimizer
model.set(
    loss = CCE_Loss(),
    optimizer = Adam_Optimizer(decay = 5e-5),
    accuracy = Classification_Acc()
)

##Finalise model
model.finalise()

##Train
model.train(X, y, test_data = (X_test, y_test), 
    epochs = 5, batch_size = 128, print_every = 100)






