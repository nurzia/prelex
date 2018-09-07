import argparse
import random
import glob
import os
import shutil

import numpy as np

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

import preprocess
import modelling


class GenerationCallback(Callback):

    def __init__(self, bptt, max_len, num_freq, window, hop):
        super(GenerationCallback, self).__init__()
        self.bptt = bptt
        self.max_len = max_len
        self.num_freq = num_freq
        self.window = window
        self.hop = hop

    def on_epoch_end(self, epoch, logs):
        timeseries = np.zeros((self.bptt + 1, self.num_freq))

        # Question: not sure whether we can actually generate for the backward model?
        spectrogram_ = []
        for timestep in range(self.max_len):
            l, r = timeseries[:-1, :], timeseries[1:, :]

            forward_in_batch = [l]
            backward_in_batch = [r[::-1, :]]

            input_data = {'forward_in': np.array(forward_in_batch, dtype=np.float32),
                          'backward_in': np.array(backward_in_batch, dtype=np.float32)}

            forward, backward = self.model.predict(input_data)
            forward = forward[0, -1, :] # only one item in batch
            spectrogram_.append(forward)

            timeseries = np.vstack((timeseries[:-1, :], forward))

        spectrogram_ = np.array(spectrogram_)
        print('generated spectogram shape:', spectrogram_.shape)

        # STUB
        #wave = preprocess.inverse_spectrogram(spectrogram_, self.window, self.hop)
        #print(wave.shape)

        # TO-DO: reconvert the spectrogram to an actual sound wave and save it to file.


class AudioGenerator(object):

    def __init__(self, audio_dir, frames, hop,
                 num_freq, bptt, batch_size,
                 max_children=None, max_files=None):
        self.audio_dir = audio_dir
        self.frames = frames
        self.hop = hop
        self.num_freq = num_freq
        self.max_children = max_children
        self.max_files = max_files
        self.bptt = bptt
        self.batch_size = batch_size
        self.num_batches = None

        children = sorted(glob.glob(self.audio_dir + '/*'))
        if self.max_children:
            children = children[:self.max_children]

        self.audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(audio_folder + '/*.wav'):
                self.audio_files.append(audio_file)

    def get_batches(self, endless=False):
        batch_cnt = 0

        while True:
            for file_idx, audio_file in enumerate(self.audio_files):
                print('parsing:', audio_file)

                # get spectrogram:
                wave, sample_rate = preprocess.load_file(audio_file)
                spectrogram = preprocess.spectrogram(wave, sample_rate,
                                                num_frames=self.frames,
                                                hop_length=self.hop,
                                                num_freq=self.num_freq)
                X = np.array(spectrogram, dtype=np.float32)
                
                # divide into time series:
                num_time_series = X.shape[0] // (self.bptt + 1)
                X = X[:num_time_series * (self.bptt + 1)]
                X = X.reshape((num_time_series, -1, X.shape[1]))

                # start yielding batches (truncates last series that don't fit):
                forward_in_batch, forward_out_batch = [], []
                backward_in_batch, backward_out_batch = [], []

                for series_idx, series in enumerate(X):

                    l, r = series[:-1, :], series[1:, :]

                    forward_in_batch.append(l)
                    forward_out_batch.append(r)

                    backward_in_batch.append(r[::-1, :])
                    backward_out_batch.append(l[::-1, :])

                    if len(forward_in_batch) >= self.batch_size or (series_idx == (X.shape[0] - 1)):
                        yield ({'forward_in': np.array(forward_in_batch, dtype=np.float32),
                                'backward_in': np.array(backward_in_batch, dtype=np.float32)},
                               {'forward_out': np.array(forward_out_batch, dtype=np.float32),
                                'backward_out': np.array(backward_out_batch, dtype=np.float32)})

                        # reset:
                        forward_in_batch, forward_out_batch = [], []
                        backward_in_batch, backward_out_batch = [], []
                        batch_cnt += 1
                
                if self.max_files and file_idx >= (self.max_files - 1):
                    break

            if self.num_batches is None:
                self.num_batches = batch_cnt

            if not endless:
                break


def main():
    parser = argparse.ArgumentParser()

    # data:
    parser.add_argument('--audio_dir', type=str, default='/home/nurzia/AUDIO')
    parser.add_argument('--chat_dir', type=str, default='/home/nurzia/TRANSCRIPTION')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing:
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--num_freq', type=int, default=64)
    parser.add_argument('--max_children', type=int, default=1)
    parser.add_argument('--max_files', type=int, default=1)

    # model:
    parser.add_argument('--bptt', type=int, default=10)
    parser.add_argument('--train_size', type=float, default=.75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--max_gen_len', type=int, default=30)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    # build the bidirectional language model:
    model = modelling.build_lm(bptt=args.bptt, input_dim=args.num_freq,
                               recurrent_dim=args.hidden_dim,
                               lr=args.lr, num_layers=args.num_layers)
    model.summary()

    # create a "lazy" batch generator:
    generator = AudioGenerator(audio_dir=args.audio_dir,
                               frames=args.frames,
                               hop=args.hop,
                               num_freq=args.num_freq,
                               max_children=args.max_children,
                               max_files=args.max_files,
                               bptt=args.bptt,
                               batch_size=args.batch_size)
    
    print('-> idle loop over data...')
    # idle loop over the generator to know how many batches we have:
    for batch in generator.get_batches(endless=False):
        pass

    # define callbacks for model fitting:
    checkpoint = ModelCheckpoint(args.model_prefix + '.model', monitor='loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, min_delta=0.03)
    generate = GenerationCallback(bptt=args.bptt, max_len=args.max_gen_len,
                                  num_freq=args.num_freq, window=args.frames,
                                  hop=args.hop)

    # fit the model:
    try:
        model.fit_generator(generator=generator.get_batches(endless=True),
                            steps_per_epoch=generator.num_batches,
                            epochs=args.epochs,
                            callbacks=[checkpoint, reduce_lr, generate])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()