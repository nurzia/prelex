import glob

import numpy as np

import torch
import torch.nn as nn

import audio_utilities
import preprocess

# https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb
# pipeline: wave > spectrogram > mel > log > (normalize)


class AudioGenerator(object):

    def __init__(self, audio_dir, fft_size, hop, num_freq, bptt, batch_size,
                 filterbank, lowcut=70, highcut=8000, sample_rate=44100,
                 max_children=None, max_files=None, eps=1e-8):
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
        self.eps = eps
        self.num_batches = None

        children = sorted(glob.glob(self.audio_dir + '/*'))
        if self.max_children:
            children = children[:self.max_children]

        self.audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(audio_folder + '/*.wav'):
                self.audio_files.append(audio_file)

    def fit(self, normalize=False):
        if normalize:
            mean, std = [], []

            for batch in self.get_batches(normalize=False):
                unrolled = batch['forward_in'].reshape((-1, self.num_freq))

                mean.append(unrolled.mean(axis=0))
                mean.append(unrolled.std(axis=0))

            self.mean = np.array(mean, dtype='float32').mean(axis=0)
            self.std = np.array(mean, dtype='float32').std(axis=0)
        else:
            for batch in self.get_batches(normalize=False):
                pass


    def generate(self, epoch, model, max_len, normalize):
        spectrogram_ = []

        model.eval()

        with torch.no_grad():
            print('-> generating...')

            hidden = model.init_hidden(1)
            inp = torch.zeros((1, 1, self.num_freq))

            for p in range(max_len):
                inp, hidden = model(inp, hidden)
                spectrogram_.append(inp.squeeze().numpy())
            
        mel_spectrogram_ = np.array(spectrogram_)
        print('generated spectogram shape:', mel_spectrogram_.shape)

        # 1. unnormalize:
        if normalize:
            mel_spectrogram_ += self.mean
            mel_spectrogram_ *= (self.std + self.eps)

        # 2. out of log domain:
        mel_spectrogram_ = np.exp(mel_spectrogram_)

        # 3. invert mel filtering:
        inverted_spectrogram_ = np.dot(mel_spectrogram_, self.filterbank)

        # 4. inverse the spectrogram and back to wave:
        wave_ = preprocess.invert_spectrogram(inverted_spectrogram_,
                                             fft_size=self.fft_size,
                                             hop=self.hop)
        print('generated wave shape:', wave_.shape)
        audio_utilities.save_audio_to_file(wave_,
                                           self.sample_rate,
                                           outfile=f'epoch{epoch}.wav')


    def spectrogram(self, audio_file, normalize=True):
        """
        magnitute spectrogram > mel filters > log compression
        """
        signal = audio_utilities.get_signal(audio_file, expected_fs=self.sample_rate)
        stft_full = audio_utilities.stft_for_reconstruction(signal, self.fft_size, self.hop)
        stft_mag = abs(stft_full) ** 2.0
        mel_spectrogram = np.dot(self.filterbank, stft_mag.T).T

        mel_spectrogram = np.log(mel_spectrogram + self.eps)

        if normalize:
            mel_spectrogram -= self.mean
            mel_spectrogram /= (self.std + self.eps)

        return mel_spectrogram



    def get_batches(self, normalize=True, sample_rate=44100):
        batch_cnt = 0

        for file_idx, audio_file in enumerate(self.audio_files):
            print(audio_file)
            
            mel_spectrogram = self.spectrogram(audio_file, normalize)
            X = np.array(mel_spectrogram, dtype=np.float32)
            
            # divide into time series:
            num_time_series = X.shape[0] // (self.bptt + 1)
            X = X[:num_time_series * (self.bptt + 1)]
            X = X.reshape((num_time_series, -1, X.shape[1]))
            
            # start yielding batches:
            forward_in_batch, forward_out_batch = [], []
            for series_idx, series in enumerate(X):
                l, r = series[:-1, :], series[1:, :]
                forward_in_batch.append(l)
                forward_out_batch.append(r)
                if len(forward_in_batch) >= self.batch_size or (series_idx == (X.shape[0] - 1)):
                    yield {'forward_in': np.array(forward_in_batch, dtype=np.float32),
                           'forward_out': np.array(forward_out_batch, dtype=np.float32)}
                    forward_in_batch, forward_out_batch = [], []
                    batch_cnt += 1
            
            if self.max_files and file_idx >= (self.max_files - 1):
                break

        if self.num_batches is None:
            self.num_batches = batch_cnt