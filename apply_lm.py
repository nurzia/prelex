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
import modelling


def main():
    # NOTE: the parameters specified here must EXACTLY match
    # those with which the language model was trained.

    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--audio_infile', type=str, default='assets/AUDIO/BRA/BRA000600_OV_01.wav')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--num_freq', type=int, default=64)

    # model
    parser.add_argument('--bptt', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    print(args)

    # load pretrained lm:
    lm = load_model(args.model_prefix + '.model')
    lm.summary()

    # get spectrogram:
    wave, sample_rate = preprocess.load_file(args.audio_infile)
    spectrogram = preprocess.spectrogram(wave, sample_rate,
                                         num_frames=args.frames,
                                         hop_length=args.hop,
                                         num_freq=args.num_freq)
    X = np.array(spectrogram, dtype=np.float32)
                
    # divide into time series:
    num_series = X.shape[0] // (args.bptt + 1)
    X = X[:num_series * (args.bptt + 1)]
    print('shape of the spectrogram:', X.shape)

    X = X.reshape((num_series, -1, X.shape[1]))
    print('shape of the batched spectrogram:', X.shape)

    forward_reprs, backward_reprs = [], []
    forward_in_batch, backward_in_batch = [], []

    filler = np.zeros(args.num_freq)

    for series_idx, series in enumerate(X):
        l, r = series[:-1, :], series[1:, :]

        forward_in_batch.append(l)
        backward_in_batch.append(r[::-1, :])

        if (len(forward_in_batch) == args.batch_size) or (series_idx == (X.shape[0] - 1)):
            in_  = {'forward_in': np.array(forward_in_batch, dtype=np.float32),
                    'backward_in': np.array(backward_in_batch, dtype=np.float32)}

            forward_out, backward_out = lm.predict(in_)

            # reverse backward output!
            backward_out = backward_out[:, ::-1, :]

            # flatten the time dimension:
            for series_idx in range(forward_out.shape[0]):

                # fill the little "holes" at the end and beginning of the timesteps:
                forward_reprs.append(filler)
                
                for timestep in range(args.bptt):
                    forward_reprs.append(forward_out[series_idx, timestep, :])
                    backward_reprs.append(backward_out[series_idx, timestep, :])
                
                backward_reprs.append(filler)

            # reset:
            forward_in_batch, backward_in_batch = [], []

    # don't forget last batch:
    if len(forward_in_batch):
        in_  = {'forward_in': np.array(forward_in_batch, dtype=np.float32),
                        'backward_in': np.array(backward_in_batch, dtype=np.float32)}

        forward_out, backward_out = lm.predict(in_)
        # reverse backward output!
        backward_out = backward_out[:, ::-1, :]
        # flatten the time dimension:
        for series_idx in range(forward_out.shape[0]):
            # fill the little "holes" at the end and beginning of the timesteps:
            forward_reprs.append(filler)
            
            for timestep in range(args.bptt):
                forward_reprs.append(forward_out[series_idx, timestep, :])
                backward_reprs.append(backward_out[series_idx, timestep, :])
            
            backward_reprs.append(filler)

    print(np.array(forward_reprs).shape)
    print(np.array(backward_reprs).shape)

    # concatenate forward and backward representations for each timestep:
    reprs = np.hstack((forward_reprs, backward_reprs))

    print('shape of the representations:', reprs.shape)


if __name__ == '__main__':
    main()