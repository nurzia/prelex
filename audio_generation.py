import glob
import json

import numpy as np

import torch
import torch.nn as nn

import preprocess
import librosa
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb
# pipeline: wave > spectrogram > mel > log > (normalize)


class AudioGenerator(object):

    def __init__(self, audio_dir, fft_size, hop, num_mel, bptt, batch_size,
                 device, lowcut=70, highcut=8000, sample_rate=44100,
                 max_children=None, max_files=None, eps=1e-8):
        self.audio_dir = audio_dir
        self.fft_size = fft_size
        self.hop = hop
        self.lowcut = lowcut
        self.highcut = highcut
        self.sample_rate = sample_rate
        self.num_mel = num_mel
        self.max_children = max_children
        self.max_files = max_files
        self.bptt = bptt
        self.batch_size = batch_size
        self.eps = eps
        self.device = device
        self.num_batches = None

        children = sorted(glob.glob(self.audio_dir + '/*'))
        if self.max_children:
            children = children[:self.max_children]

        self.audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(audio_folder + '/*.wav'):
                self.audio_files.append(audio_file)

        self.melW = librosa.filters.mel(sr=self.sample_rate, n_fft=self.fft_size,
                                        n_mels=self.num_mel,
                                        fmin=self.lowcut, fmax=self.highcut)
        self.ham_win = np.hamming(self.fft_size)

    def dump(self, model_prefix):
        with open(model_prefix + '_params.json', 'w') as f:
            json.dump({'fft_size': self.fft_size,
                       'hop': self.hop,
                       'lowcut': self.lowcut,
                       'highcut': self.highcut,
                       'sample_rate': self.sample_rate,
                       'num_mel': self.num_mel,
                       'bptt': self.bptt, f,
                       indent=4)
            joblib.dump(self.scaler, model_prefix + '_scaler.pkl')

    @classmethod
    def load(self, model_prefix, **kwargs):
        with open(model_prefix + '_params.json', 'r') as f:
            params = json.load(f)
            for k, v in kwargs:
                params[k] = v
            ag = AudioGenerator(**params)
            ag.scaler = joblib.load(model_prefix + '_scaler.pkl')
            return ag

    def fit(self, normalize=False):
        if normalize:
            self.scaler = StandardScaler()
            for batch in self.get_batches(normalize=False):
                flat_batch = batch.reshape((-1, self.num_mel))
                self.scaler.partial_fit(flat_batch)
        else:
            for batch in self.get_batches(normalize=False):
                pass

    def generate(self, epoch, model, max_len, normalize):
        spectrogram_ = []

        model.eval()

        with torch.no_grad():
            print('-> generating...')

            hidden = model.init_hidden(1)
            hidden = tuple([t.to(self.device) for t in hidden])

            inp = torch.zeros((1, 1, self.num_mel)).to(self.device)

            for p in range(max_len):
                inp, hidden = model(inp, hidden)
                spectrogram_.append(inp.squeeze().cpu().numpy())
            
        mel_spectrogram_ = np.array(spectrogram_)
        #print('generated spectogram shape:', mel_spectrogram_.shape)

        # 1. unnormalize:
        if normalize:
            mel_spectrogram_ = self.scaler.inverse_transform(mel_spectrogram_)

        # 2. out of log domain:
        mel_spectrogram_ = np.exp(mel_spectrogram_)

        # 3. invert mel filtering:
        inverted_spectrogram_ = np.dot(self.melW.T, mel_spectrogram_.T) # still not sure whether this is correct
        #print('inversed spectrogram:', inverted_spectrogram_.shape)

        # 4. inverse the spectrogram and back to wave:
        wave_ = preprocess.griffinlim(inverted_spectrogram_, n_iter=50, window='hann',
                                      n_fft=self.fft_size, win_length=self.fft_size,
                                      hop_length=self.hop, verbose=False)

        #print('generated wave shape:', wave_.shape)
        librosa.output.write_wav(y=wave_, sr=self.sample_rate,
                                 path=f'epoch{epoch}.wav', norm=True)


    def get_batches(self, normalize=True, sample_rate=44100):
        batch_cnt = 0

        for file_idx, audio_path in enumerate(self.audio_files):
            #print(audio_path)

            sound_clip, fn_fs = preprocess.read_audio(audio_path, target_fs=sample_rate, duration=None) #!!!
            
            ### sanity checks: ################################################################
            if sound_clip.shape[0] < self.fft_size:
                print("File %s is shorter than fft size - DISCARDING" % audio_path)
                continue

            if sound_clip.shape[0] == 0:
                print("File %s is corrupted!" % fn)
                continue

            assert (int(fn_fs) == sample_rate)
            ####################################################################################

            # Compute spectrogram                
            [f, t, X] = signal.spectral.spectrogram(
                x=sound_clip,
                window=self.ham_win,
                nperseg=self.fft_size,
                noverlap=self.hop,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
            X = np.dot(X.T, self.melW.T)
            X = np.log(X + self.eps)
            X = X.astype(np.float32)
            
            # divide into time series:
            num_time_series = X.shape[0] // (self.bptt + 1)
            X = X[:num_time_series * (self.bptt + 1)]
            start_idx, end_idx, batch_series = 0, self.bptt + 1, []

            while end_idx <= X.shape[0]:
                batch_series.append(X[start_idx : end_idx])

                if len(batch_series) == self.batch_size or end_idx == X.shape[0] - 1:
                    batch = np.array(batch_series, dtype='float32')

                    if normalize:
                        flat_batch = batch.reshape((-1, self.num_mel))
                        flat_batch = self.scaler.transform(flat_batch)
                        yield flat_batch.reshape((batch.shape[0], self.bptt + 1, self.num_mel))
                    else:
                        yield batch

                    # reset:
                    batch_cnt += 1
                    batch_series = []

                # make sure that last timestep is first of next batch
                start_idx += self.bptt
                end_idx += self.bptt
                
            
            if self.max_files and file_idx >= (self.max_files - 1):
                break

        if self.num_batches is None:
            self.num_batches = batch_cnt