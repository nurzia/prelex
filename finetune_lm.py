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
    shape[-1] *= 2
    shape[-2] += 1
    return tuple(shape)


class BatchGenerator(object):

    def __init__(self, X, Y, batch_size):
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.num_batches = None

    def generate_batches(self, endless=False):
        forward_in_batch, backward_in_batch, out_batch = [], [], []

        batch_cnt = 0

        while True:
            for batch, y in zip(self.X, self.Y):

                l, r = batch[:-1, :], batch[1:, :]

                forward_in_batch.append(l)
                backward_in_batch.append(r[::-1, :])

                out_batch.append(y)

                if len(forward_in_batch) >= self.batch_size:
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
    parser.add_argument('--max_files', type=int, default=2)
    parser.add_argument('--preprocess', action='store_true', default=False)

    # model
    parser.add_argument('--bptt', type=int, default=11) # needs to be one higher than the LM
    parser.add_argument('--train_size', type=float, default=.75)
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
 
    if args.preprocess:
        X, Y = [], []
        for file_idx, audio_file in enumerate(audio_files):
            print(audio_file)
            # extract audio
            wave, sample_rate = preprocess.load_file(audio_file)
            spectrogram = preprocess.spectrogram(wave, sample_rate,
                                                  num_frames=args.frames,
                                                  hop_length=args.hop,
                                                  num_freq=args.num_freq)
            print(spectrogram.shape)

            child_name = audio_file.split('/')[2]

            # extract transcription
            bn = os.path.basename(audio_file).replace('.wav', '')
            p = f'{args.chat_dir}/{child_name}/{bn}_S.cha'
            intervals = preprocess.extract_intervals(p)
            markers = preprocess.apply_intervals(wave, sample_rate, intervals)
            ints = preprocess.mark_spectro(spectrogram.shape[0], num_frames=args.frames,
                                hop_length=args.hop, markers=markers)
            X.extend(spectrogram)
            Y.extend(ints)
            
            if args.max_files and len(X) >= args.max_files:
                break

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int32)

        try:
            shutil.rmtree(args.data_dir)
        except FileNotFoundError:
            pass
        os.mkdir(args.data_dir)

        np.save(f'{args.data_dir}/X.npy', X)
        np.save(f'{args.data_dir}/Y.npy', Y)

    else:
        X = np.load(f'{args.data_dir}/X.npy')
        Y = np.load(f'{args.data_dir}/Y.npy')


    # batchify the data:
    num_batches = X.shape[0] // args.bptt
    X = X[:num_batches * args.bptt]
    X = X.reshape((num_batches, -1, X.shape[1]))
    Y = Y[:num_batches * args.bptt]
    Y = Y.reshape((num_batches, -1))
    
    batch_sums = Y.sum(axis=1)
    non_empty = np.where(batch_sums > 0)[0]
    empty = np.where(batch_sums == 0)[0]
    empty = np.random.choice(empty, len(non_empty), replace=False)
    keep = tuple(sorted(list(non_empty) + list(empty)))

    X = X[keep, :, :]
    Y = Y[keep, :]

    print('1-ratio: ', Y.sum() / len(Y.ravel()))

    # split train from rest (dev + test)
    sums = [int(b.sum() > 0) for b in Y]
    splits = train_test_split(X, Y, train_size=args.train_size,
                              stratify=sums,
                              random_state=args.seed)
    X_train, X_rest, Y_train, Y_rest = splits

    # dev from test
    sums = [int(b.sum() > 0) for b in Y_rest]
    splits = train_test_split(X_rest, Y_rest, train_size=.5,
                              stratify=sums,
                              random_state=args.seed)
    X_dev, X_test, Y_dev, Y_test = splits
    
    Y_train_cat = to_categorical(Y_train, num_classes=2)
    Y_dev_cat = to_categorical(Y_dev, num_classes=2)

    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)

    train_generator = BatchGenerator(X_train, Y_train_cat, args.batch_size)
    for in_, out_ in train_generator.generate_batches(endless=False):
        pass

    dev_generator = BatchGenerator(X_dev, Y_dev_cat, args.batch_size)
    for in_, out_ in dev_generator.generate_batches(endless=False):
        pass

    base_model = load_model(args.model_prefix + '.model')

    x = Lambda(combine_, output_shape=combine_output_shape)(base_model.output)
    print(x.shape, '++++++++++')
    x = TimeDistributed(Dense(args.dense_dim, activation='relu'))(x)
    print(x.shape, '-----------')
    x = Dropout(args.dropout)(x)

    predictions = TimeDistributed(Dense(2, activation='relu'), name='output_')(x)
    
    # this is the model we will train
    fine_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    
    fine_model.summary()

    optim = Adam(lr=args.lr)
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
                            validation_data=dev_generator.generate_batches(endless=False),
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
                            validation_data=dev_generator.generate_batches(endless=False),
                            callbacks=[checkpoint, reduce_lr])
    except KeyboardInterrupt:
        pass

    #predictions = fine_model.predict(X_dev).argmax(axis=-1).ravel()
    #print(classification_report(Y_dev.ravel(), predictions))


if __name__ == '__main__':
    main()