import argparse
import random
import glob
import os
import shutil

import numpy as np

from keras.utils import to_categorical
from keras.layers import *
from keras.models import  Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

import preprocess
import utils
import modelling


def main():
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--audio_dir', type=str, default='assets/AUDIO')
    parser.add_argument('--chat_dir', type=str, default='assets/TRANSCRIPTION')
    parser.add_argument('--data_dir', type=str, default='assets/preprocessed')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing
    parser.add_argument('--frames', type=int, default=256) # 256
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--num_freq', type=int, default=64)
    parser.add_argument('--max_children', type=int, default=1)
    parser.add_argument('--max_files', type=int, default=2)

    # model
    parser.add_argument('--bptt', type=int, default=10)
    parser.add_argument('--train_size', type=float, default=.75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dense_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--dropout', type=float, default=.5)
    
    args = parser.parse_args()
    print(args)

    base_model = load_model(args.model_prefix + '.model')
    x = base_model.output
    x = Concatenate(axis=-1)(x)
    x = TimeDistributed(Dense(args.dense_dim, activation='relu'))(x)
    x = Dropout(args.dropout)(x)
    predictions = TimeDistributed(Dense(2, activation='relu'))(x)
    
    # this is the model we will train
    fine_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    
    fine_model.summary()
    fine_model.save(args.model_prefix + '-fine.model')
    return


if __name__ == '__main__':
    main()