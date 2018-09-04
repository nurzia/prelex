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
import keras.backend as K
from keras.optimizers import Adam, SGD
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical


import preprocess
import utils
import modelling


def combine_(x):
    forward, backward = x

    # reverse the output:
    backward = backward[:, ::-1, :]

    left_pattern = [[0, 0], [1, 0], [0, 0]]
    forward = tf.pad(forward, left_pattern,
                     mode='CONSTANT', constant_values=0.0)

    right_pattern = [[0, 0], [0, 1], [0, 0]]
    backward = tf.pad(backward, right_pattern,
                      mode='CONSTANT', constant_values=0.0)

    return Concatenate(axis=-1)([forward, backward])


def combine_output_shape(input_shape):
    shape = list(input_shape[0])
    shape[-1] *= 2 # we double the actual feature dimension
    shape[-2] += 1 # we add a timestep with respect to the LM
    return tuple(shape)


class BatchGenerator(object):

    def __init__(self, filenames, batch_size, frames,
                hop, num_freq, chat_dir, bptt):
        self.filenames = filenames
        self.bptt = bptt
        self.batch_size = batch_size
        self.frames = frames
        self.hop = hop
        self.num_freq = num_freq
        self.chat_dir = chat_dir
        self.num_batches = None

    def generate_batches(self, endless=False):
        forward_in_batch, backward_in_batch, out_batch = [], [], []

        batch_cnt = 0

        while True:

            for file_idx, audio_file in enumerate(self.filenames):
                # extract audio
                wave, sample_rate = preprocess.load_file(audio_file)
                spectrogram = preprocess.spectrogram(wave, sample_rate,
                                                      num_frames=self.frames,
                                                      hop_length=self.hop,
                                                      num_freq=self.num_freq)

                child_name = audio_file.split('/')[2]

                # extract transcription
                bn = os.path.basename(audio_file).replace('.wav', '')
                p = f'{self.chat_dir}/{child_name}/{bn}_S.cha'
                intervals = preprocess.extract_intervals(p)
                markers = preprocess.apply_intervals(wave, sample_rate, intervals)
                ints = preprocess.mark_spectro(spectrogram.shape[0], num_frames=self.frames,
                                    hop_length=self.hop, markers=markers)

                X = np.array(spectrogram, dtype=np.float32)
                Y = np.array(ints, dtype=np.int32)

                # batchify the data:
                num_series = X.shape[0] // self.bptt
                X = X[:num_series * self.bptt]
                X = X.reshape((num_series, -1, X.shape[1]))
                Y = Y[:num_series * self.bptt]
                Y = Y.reshape((num_series, -1))
                
                series_sums = Y.sum(axis=1)
                non_empty = np.where(series_sums > 0)[0]
                empty = np.where(series_sums == 0)[0]
                empty = np.random.choice(empty, len(non_empty), replace=False)
                keep = tuple(sorted(list(non_empty) + list(empty)))

                X = X[keep, :, :]
                Y = Y[keep, :]

                #print('1-ratio: ', Y.sum() / len(Y.ravel()))

                Y_cat = to_categorical(Y, num_classes=2)

                for series_idx, (series, y) in enumerate(zip(X, Y_cat)):

                    l, r = series[:-1, :], series[1:, :]

                    forward_in_batch.append(l)
                    backward_in_batch.append(r[::-1, :])

                    out_batch.append(y)

                    if len(forward_in_batch) >= self.batch_size or (series_idx == (X.shape[0] - 1)):
                        yield ({'forward_in': np.array(forward_in_batch, dtype=np.float32),
                                'backward_in': np.array(backward_in_batch, dtype=np.float32)},
                               {'output_': np.array(out_batch, dtype=np.int32)})
                        batch_cnt += 1
                        forward_in_batch, backward_in_batch, out_batch = [], [], []

            if not endless:
                break

        if not self.num_batches:
            self.num_batches = batch_cnt


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
    parser.add_argument('--max_files', type=int, default=10)
    parser.add_argument('--preprocess', action='store_true', default=False)

    # model
    parser.add_argument('--bptt', type=int, default=11) # needs to be one higher than the LM
    parser.add_argument('--train_size', type=float, default=.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--burn_in_epochs', type=int, default=1)
    parser.add_argument('--dense_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.0000001) # the learning rate has to be tiny to avoid underflow (NaN)
    parser.add_argument('--dropout', type=float, default=.0)
    
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    children = sorted(glob.glob(f'{args.audio_dir}/*'))
    if args.max_children:
        children = children[:args.max_children]

    audio_files = []
    for audio_folder in children:
        for audio_file in glob.glob(f'{audio_folder}/*.wav'):
            audio_files.append(audio_file)

    if args.max_files:
        audio_files = audio_files[:args.max_files]

    train_fns, rest_fns = train_test_split(audio_files, train_size=args.train_size,
                                           random_state=args.seed)
    dev_fns, test_fns = train_test_split(rest_fns, train_size=.5,
                                        random_state=args.seed)

    train_generator = BatchGenerator(train_fns, batch_size=args.batch_size, frames=args.frames,
                                     hop=args.hop, num_freq=args.num_freq, chat_dir=args.chat_dir,
                                     bptt=args.bptt)
    for in_, out_ in train_generator.generate_batches(endless=False):
        pass

    dev_generator = BatchGenerator(dev_fns, batch_size=args.batch_size, frames=args.frames,
                                    hop=args.hop, num_freq=args.num_freq, chat_dir=args.chat_dir,
                                    bptt=args.bptt)
    for in_, out_ in dev_generator.generate_batches(endless=False):
        pass

    base_model = load_model(args.model_prefix + '.model')

    x = Lambda(combine_, output_shape=combine_output_shape)(base_model.output)
    x = TimeDistributed(Dense(args.dense_dim, activation='relu'))(x)
    x = Dropout(args.dropout)(x)

    predictions = TimeDistributed(Dense(2, activation='relu'), name='output_')(x)
    
    # this is the model we will train
    fine_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    
    fine_model.summary()

    optim = SGD(lr=args.lr)
    fine_model.compile(optimizer=optim,
                  loss={'output_': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    fine_model.save(args.model_prefix + '-fine.model')

    # define callbacks for model fitting:
    checkpoint = ModelCheckpoint(args.model_prefix + '-fine.model', monitor='val_loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, min_delta=0.03)

    try:
        fine_model.fit_generator(generator=train_generator.generate_batches(endless=True),
                            steps_per_epoch=train_generator.num_batches,
                            epochs=args.burn_in_epochs,
                            validation_data=dev_generator.generate_batches(endless=True),
                            validation_steps=dev_generator.num_batches,
                            callbacks=[checkpoint, reduce_lr])
    except KeyboardInterrupt:
        pass

    # unfreeze after burn-in training:
    for layer in base_model.layers:
        layer.trainable = True

    optim = SGD(lr=args.lr)
    fine_model.compile(optimizer=optim,
                  loss={'output_': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    # define callbacks for model fitting:
    checkpoint = ModelCheckpoint(args.model_prefix + '-fine.model', monitor='val_loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, min_delta=0.03)

    try:
        fine_model.fit_generator(generator=train_generator.generate_batches(endless=True),
                            steps_per_epoch=train_generator.num_batches,
                            epochs=args.epochs,
                            validation_steps=dev_generator.num_batches,
                            validation_data=dev_generator.generate_batches(endless=True),
                            callbacks=[checkpoint, reduce_lr])
    except KeyboardInterrupt:
        pass

    #predictions = fine_model.predict(X_dev).argmax(axis=-1).ravel()
    #print(classification_report(Y_dev.ravel(), predictions))


if __name__ == '__main__':
    main()