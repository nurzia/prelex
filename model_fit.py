from PreProcessClass import *
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support


myclass = PreProcessClass()

train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
dev_X = np.load('dev_X.npy')
dev_Y = np.load('dev_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')

y_true = test_Y

train_Y = to_categorical(train_Y, num_classes = 2)
dev_Y = to_categorical(dev_Y, num_classes = 2)
test_Y = to_categorical(test_Y, num_classes = 2)

model = build_model1(learning_rate = 0.0001, recurrent_dim=256, dr_ou = None, num_layers=2)


class_weight = {0: 1.,
               1: 88.}

sample_weight=np.array([1., 88.])

history = model.fit(train_X, train_Y, epochs = 15, validation_data = (dev_X, dev_Y), sample_weight=sample_weight)

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']



test_pred = model.predict(x=test_X)

#y_true.flatten()
print("shape0")
print(y_true.shape[0])
print("shape1")
print(y_true.shape[1])


y_true=y_true.reshape(y_true.shape[0],y_true.shape[1])

#test_pred.reshape((test_pred.shape[0]*test_pred.shape[1], test_pred.shape[2]))

test_pred = np.argmax(test_pred, axis=-1)


y_true=y_true.flatten()
test_pred=test_pred.flatten()
#print("y_true")
#print(y_true)



print("y_true")
print(y_true.shape)
print("test_pred")
print(test_pred.shape)


i = 0
true_percent=0.
pred_percent=0.

weird_num=0
weird_num_des=0
weird_num_sin=0
norma=0
while i<len(y_true):
    if y_true[i]==1:
        true_percent += 1
        if y_true[i+1]==0 and y_true[i-1]==0:
            weird_num+=1        
        elif y_true[i+1]==0 and y_true[i-1]==1:
            weird_num_des+=1
        elif y_true[i+1]==1 and y_true[i-1]==0:
            weird_num_sin+=1
        elif y_true[i+1]==1 and y_true[i-1]==1:
            norma+=1
    if test_pred[i]==1:
        pred_percent += 1
    i+=1

true_percent/=len(y_true)
pred_percent/=len(test_pred)

print("WEIRD_NUM")
print(weird_num)
print("DESTRA")
print(weird_num_des)
print("SINISTRA")
print(weird_num_sin)
print("BUONI")
print(norma)
print("true_percent")
print(true_percent)
print("pred_percent")
print(pred_percent)


print("------------------------------------------")
print(y_true)
print(test_pred)


pr= precision_recall_fscore_support(y_true,test_pred,average=None, labels=[0,1])

f1 = f1_score(y_true=y_true, y_pred=test_pred, average=None)

print("y_true")
print(y_true.shape)
print("test_pred")
print(test_pred.shape)
test_loss, test_acc = model.evaluate(test_X, test_Y)

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


