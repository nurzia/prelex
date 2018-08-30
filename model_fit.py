from PreProcessClass import *
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support


#loading the three sets
train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
dev_X = np.load('dev_X.npy')
dev_Y = np.load('dev_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')

#keeping a version of the test-set labels in current shape to use it later to compute f1-score
y_true = test_Y

#preparing the three sets' labels to fit model to
train_Y = to_categorical(train_Y, num_classes = 2)
dev_Y = to_categorical(dev_Y, num_classes = 2)
test_Y = to_categorical(test_Y, num_classes = 2)

#build model
model = build_model1(learning_rate = 0.0001, recurrent_dim=256, dr_ou = None, num_layers=2)

#define sample weights
sample_weight=np.array([1., 88.])

#fit model to data and store values of loss and validation loss
history = model.fit(train_X, train_Y, epochs = 15, validation_data = (dev_X, dev_Y), sample_weight=sample_weight)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

#evaluate model on test set 
test_loss, test_acc = model.evaluate(test_X, test_Y)

#build array of predicted labels
test_pred = model.predict(x=test_X)

#reshape predicted and true arrays to compute precision and recall
test_pred = np.argmax(test_pred, axis=-1).flatten()
y_true=y_true.reshape(y_true.shape[0], y_true.shape[1]).flatten()

#compute precision, recall and f1-score
pr= precision_recall_fscore_support(y_true,test_pred,average=None, labels=[0,1])
f1 = f1_score(y_true=y_true, y_pred=test_pred, average=None)


print("------------------------------------------")
print(y_true)
print(test_pred)
print("y_true")
print(y_true.shape)
print("test_pred")
print(test_pred.shape)
print("LOSS")
print(loss_values)
print("VAL_LOSS")
print(val_loss_values)
print("ACCURACY")
print(test_acc)
print("F1-TEST")
print(f1)
print("PRECISION AND RECALL")
print(pr)


