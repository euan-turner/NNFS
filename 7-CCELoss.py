import numpy as np

##Example output - probabilities for 3 samples
softmax_outputs = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
##Ground truth - targets -> targeting 0th, 1st and 1st indices of outputs
class_targets = [0,1,1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(neg_log)

##One-hot encoded class targets
onehot_targets = np.array([[1,0,0],[0,1,0],[0,1,0]])

if len(onehot_targets.shape) == 1: ##Covers the class_targets format case
    correct_confidences = softmax_outputs[range(len(softmax_outputs)),class_targets]
elif len(onehot_targets.shape) == 2: ##Covers the onehot_targets format case
    correct_confidences = np.sum(softmax_outputs*onehot_targets, axis=1)

onehot_neg_loss = -np.log(correct_confidences)
onehot_avg_loss = np.mean(onehot_neg_loss)
print(onehot_avg_loss)


##Common Loss Class 
class Loss:

    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        mean_loss = np.mean(sample_losses)
        return mean_loss

##Cross-Entropy Loss Class
class Loss_CategoricalCrossEntropy(Loss):

    def forward(self,y_pred,y_true):
        num_samples = len(y_pred)
        ##Clip data to prevent division by 0, clip both sides so mean does not move
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        ##Deal with categorical labels or one hot labels
        if len(y_true.shape) == 1:
            correct_confs = y_pred_clipped[range(num_samples,y_true)]
        elif len(y_true.shape) == 2:
            correct_confs = np.sum(y_pred_clipped*y_true,axis=1)
        
        neg_log = -np.logs(correct_confs)
        return neg_log