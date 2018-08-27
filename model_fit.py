from PreProcessClass import *
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

myclass = PreProcessClass()
print(os.getcwd())

train_X = np.load('sets/train_X.npy')
train_Y = np.load('sets/train_Y.npy')
dev_X = np.load('sets/dev_X.npy')
dev_Y = np.load('sets/dev_Y.npy')
test_X = np.load('sets/test_X.npy')
test_Y = np.load('sets/test_Y.npy')

print("train_X")
print(train_X.shape)
print("train_Y")
print(train_Y.shape)
print("dev_X")
print(dev_X.shape)
print("dev_Y")
print(dev_Y.shape)
print("test_X")
print(test_X.shape)
print("test_Y")
print(test_Y.shape)

train_Y = to_categorical(train_Y, num_classes = 2)
dev_Y = to_categorical(dev_Y, num_classes = 2)
test_Y = to_categorical(test_Y, num_classes = 2)

model = build_model1(learning_rate = 0.01, recurrent_dim=1024, dr_ou = None, num_layers=2)
print(model.summary())

history = model.fit(train_X, train_Y, epochs = 15, validation_data = (dev_X, dev_Y))

history_dict=history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']
epochs = np.arange(len(loss_values))

test_loss, test_acc = model.evaluate(test_X, test_Y)

print("LOSS")
print(loss_values)
print("VAL_LOSS")
print(val_loss_values)
print("ACCURACY")
print(test_acc)

loss_file = open("loss_layers.dat","w+")
val_loss_file = open("val_loss_layers.dat","w+")
accuracy_file = open("accuracy_layers.dat","w+")


loss_file.write(str(loss_values))

val_loss_file.write(str(val_loss_values))

accuracy_file.write(str(test_acc))






