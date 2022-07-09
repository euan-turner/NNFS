from fashionMNIST import create_mnist_data
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nn import *

nnfs.init()


##Labels
fashion_mnist_labels = {
    0: 'Tishirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

tshirt_data = cv2.imread('pred_images/tshirt.png', cv2.IMREAD_GRAYSCALE)
tshirt_data = cv2.resize(tshirt_data, (28,28))
plt.imshow(tshirt_data, cmap='gray')
plt.show()
##Load model


'''
##Create dataset
X, y, X_test, y_test = create_mnist_data('fashion_mnist_images')

##Scale and reshape
X_test = (X_test.reshape(X_test.shape[0],-1).astype(np.float32) - 127.5) / 127.5

##Load model
model = Model.load('fashion_mnist.model')

##Predict on first 5 samples
confs = model.predict(X_test[:5])
preds = model.output_layer_activation.predictions(confs)

##First 5 labels
print(y_test[:5])
##First 5 predictions
for p in preds:
    print(fashion_mnist_labels[p])
##Print confidences on predictions
print(confs)
'''