from PreProcessClass import *
from keras.utils import to_categorical
import matplotlib.pyplot as plt

myclass = PreProcessClass()

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = myclass.get_data(num_frames = 44100, hop_length = 44100, num_freq = 128)
