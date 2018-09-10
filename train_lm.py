import argparse
import random
import glob
import os
import shutil

import numpy as np
import librosa

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback

import preprocess
import modelling
import audio_utilities

# for pre-emphasis etc: LOG OF MEL
# https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb


class GenerationCallback(Callback):

    def __init__(self, bptt, max_len, num_freq, fft_size,
                 filterbank, hop, sample_rate=44100):
        super(GenerationCallback, self).__init__()
        self.bptt = bptt
        self.max_len = max_len
        self.num_freq = num_freq
        self.fft_size = fft_size
        self.filterbank = filterbank
        self.hop = hop
        self.sample_rate = sample_rate

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

        mel_spectrogram_ = np.array(spectrogram_)
        print('generated spectogram shape:', mel_spectrogram_.shape)

        inverted_spectrogram_ = np.dot(mel_spectrogram_, self.filterbank)

        wave_ = preprocess.invert_spectrogram(inverted_spectrogram_,
                                             fft_size=self.fft_size,
                                             hop=self.hop)
        print('generated wave shape:', wave_.shape)
        audio_utilities.save_audio_to_file(wave_,
                                           self.sample_rate,
                                           outfile=f'epoch{epoch}.wav')


class AudioGenerator(object):

    def __init__(self, audio_dir, fft_size, hop, num_freq, bptt, batch_size,
                 filterbank, lowcut=70, highcut=8000, sample_rate=44100,
                 max_children=None, max_files=None):
        self.audio_dir = audio_dir
        self.fft_size = fft_size
        self.hop = hop
        self.lowcut = lowcut
        self.highcut = highcut
        self.sample_rate = sample_rate
        self.num_freq = num_freq
        self.max_children = max_children
        self.max_files = max_files
        self.bptt = bptt
        self.batch_size = batch_size
        self.filterbank = filterbank
        self.num_batches = None

        children = sorted(glob.glob(self.audio_dir + '/*'))
        if self.max_children:
            children = children[:self.max_children]

        self.audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(audio_folder + '/*.wav'):
                self.audio_files.append(audio_file)

    def fit(self):
        mean, std = [], []

        for batch in self.get_batches(endless=False, normalize=False):
            unrolled = batch[0]['forward_in'].reshape((-1, self.num_freq))

            mean.append(unrolled.mean(axis=0))
            mean.append(unrolled.std(axis=0))

        self.mean = np.array(mean, dtype='float32').mean(axis=0)
        self.std = np.array(mean, dtype='float32').std(axis=0)


    def spectrogram(self, audio_file, normalize=True, eps=1e-8):
        """
        magnitute spectrogram > mel filters > log compression
        """
        signal = audio_utilities.get_signal(audio_file, expected_fs=self.sample_rate)
        stft_full = audio_utilities.stft_for_reconstruction(signal, self.fft_size, self.hop)
        stft_mag = abs(stft_full) ** 2.0
        mel_spectrogram = np.dot(self.filterbank, stft_mag.T).T

        mel_spectrogram = np.log(mel_spectrogram + eps)

        if normalize:
            mel_spectrogram -= self.mean
            mel_spectrogram /= (self.std + eps)

        return mel_spectrogram



    def get_batches(self, normalize=True, endless=False, sample_rate=44100):
        batch_cnt = 0

        while True:
            for file_idx, audio_file in enumerate(self.audio_files):
                print(audio_file)
                
                mel_spectrogram = self.spectrogram(audio_file, normalize)
                X = np.array(mel_spectrogram, dtype=np.float32)
                
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
    parser.add_argument('--audio_dir', type=str, default='assets/AUDIO')#'/home/nurzia/AUDIO')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing:
    parser.add_argument('--fft_size', type=int, default=256)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--num_freq', type=int, default=150) # https://blogs.technet.microsoft.com/machinelearning/2018/01/30/hearing-ai-getting-started-with-deep-learning-for-audio-on-azure/
    parser.add_argument('--max_children', type=int, default=1)
    parser.add_argument('--max_files', type=int, default=1)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--lowcut', type=int, default=70)
    parser.add_argument('--highcut', type=int, default=8000)

    # model:
    parser.add_argument('--bptt', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.0001)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    # create mel filterbank:
    linear_bin_count = 1 + args.fft_size // 2
    filterbank = audio_utilities.make_mel_filterbank(args.lowcut, args.highcut, args.num_freq,
                                                     linear_bin_count , 44100)

    # build the bidirectional language model:
    model = modelling.build_lm(bptt=args.bptt, input_dim=args.num_freq,
                               recurrent_dim=args.hidden_dim,
                               lr=args.lr, num_layers=args.num_layers)
    model.summary()

    # create a "lazy" batch generator:
    generator = AudioGenerator(audio_dir=args.audio_dir,
                               fft_size=args.fft_size,
                               hop=args.hop,
                               num_freq=args.num_freq,
                               max_children=args.max_children,
                               filterbank=filterbank,
                               max_files=args.max_files,
                               bptt=args.bptt,
                               batch_size=args.batch_size)
    generator.fit()

    # define callbacks for model fitting:
    checkpoint = ModelCheckpoint(args.model_prefix + '.model', monitor='loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, min_delta=0.03)
    generate = GenerationCallback(bptt=args.bptt, max_len=args.max_gen_len,
                                  num_freq=args.num_freq, fft_size=args.fft_size,
                                  hop=args.hop, filterbank=filterbank)

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