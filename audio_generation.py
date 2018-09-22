import glob
import os
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import preprocess
import librosa
from scipy.io import wavfile
import audioread
from scipy import signal

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import cluster
from sklearn import svm
from sklearn.decomposition import PCA

# https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb
# pipeline: wave > spectrogram > mel > log > normalize


class AudioGenerator(object):

    def __init__(self, audio_dir, fft_size, hop, num_mel, bptt, batch_size,
                 device, mean_=None, var_=None, lowcut=70, highcut=8000,
                 sample_rate=44100, max_children=None, max_files=None, eps=1e-8,
                 **kwargs):
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

        if mean_ is not None and var_ is not None:
            self.mean_ = torch.FloatTensor(mean_).to(self.device)
            self.var_ = torch.FloatTensor(var_).to(self.device)

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
                       'bptt': self.bptt}, f,
                       indent=4)

            scale = np.array((self.mean_.cpu().numpy(),
                              self.var_.cpu().numpy()))
            np.save(model_prefix + '_scale.npy', scale)

    @classmethod
    def load(self, model_prefix, args):
        args = vars(args)

        with open(model_prefix + '_params.json', 'r') as f:
            params = json.load(f)
            for k, v in params.items():
                args[k] = v
        
        scale = np.load(model_prefix + '_scale.npy')
        args['mean_'], args['var_'] = scale

        return AudioGenerator(**args)

    def fit(self):
        scaler = StandardScaler() # only used during fitting because of handy partial_fit method

        for batch, _ in self.get_batches():
            batch = batch.cpu().numpy()
            flat_batch = batch.reshape((-1, self.num_mel))
            scaler.partial_fit(flat_batch)

        self.mean_ = torch.FloatTensor(scaler.mean_).to(self.device)
        self.var_ = torch.FloatTensor(scaler.var_).to(self.device)

    def generate(self, epoch, model, max_len):
        spectrogram_ = []

        model.eval()

        with torch.no_grad():

            hidden = model.init_hidden(1)
            hidden = tuple([t.to(self.device) for t in hidden])

            inp = torch.zeros((1, 1, self.num_mel)).to(self.device)

            for p in range(max_len):
                inp, hidden = model(inp, hidden)
                spectrogram_.append(inp.squeeze().cpu().numpy())
            
        mel_spectrogram_ = np.array(spectrogram_)
        #print('generated spectogram shape:', mel_spectrogram_.shape)

        # 1. unnormalize:
        mel_spectrogram_ *= self.var_.cpu().numpy()
        mel_spectrogram_ += self.mean_.cpu().numpy()

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

    def cluster(self, model, mode='mean'):
        assert mode in ('mean', 'last')
        model.eval()
        labels_true, reprs = [], []

        chi_files = list(glob.glob('/home/nurzia/chi_utt/*.wav'))[:100]
        adu_files = list(glob.glob('/home/nurzia/adu_utt/*.wav'))[:100]

        for fp in chi_files + adu_files:
            cat = os.path.basename(fp).split('_')[0]
            labels_true.append(cat)

            X = torch.FloatTensor(self.mel_spectrogram(fp))
            X = X.to(self.device)

            X.sub_(self.mean_).div_(self.var_)

            hidden_states = []
            with torch.no_grad():
                hidden = model.init_hidden(1)
                hidden = tuple([t.to(self.device) for t in hidden])
                for x in X:
                    _, hidden = model.lstm(x.view(1, 1, -1), hidden)
                    hidden_states.append(hidden[0].squeeze().cpu().numpy())

            hidden_states = np.array(hidden_states)
            if mode == 'mean':
                reprs.append(hidden_states.mean(axis=0))
            elif mode == 'last':
                reprs.append(hidden_states[-1])

        reprs = np.array(reprs)
        label_encoder = LabelEncoder()
        labels_true_int = label_encoder.fit_transform(labels_true)

        clust = cluster.KMeans(n_clusters=2)
        labels_pred = clust.fit_predict(reprs)
        score = metrics.adjusted_rand_score(labels_true_int, labels_pred)
        print('Adjusted rand-score:', score)

        pca = PCA(n_components=2)
        pca_X = pca.fit_transform(reprs)
        loadings = pca.components_.transpose()
        var_exp = pca.explained_variance_ratio_

        clf = svm.SVC(kernel='linear').fit(pca_X, labels_true_int)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = pca_X[:, 0].min() - 1, pca_X[:, 0].max() + 1
        y_min, y_max = pca_X[:, 1].min() - 1, pca_X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        ax1.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        x1, x2 = pca_X[:, 0], pca_X[:, 1]
        plt.scatter(x1, x2, edgecolors='none', facecolors='none')
        for p1, p2, a in zip(x1, x2, labels_true):
            plt.text(p1, p2, a.lower()[:3], ha='center',
                va='center', fontdict={'family': 'Arial', 'size': 12})

        ax1.set_xlabel('PC1 ('+ str(round(var_exp[0] * 100, 2)) +'%)')
        ax1.set_ylabel('PC2 ('+ str(round(var_exp[1] * 100, 2)) +'%)')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()

        plt.savefig('pca_mesh.pdf')
        plt.clf()


    def batchify(self, data):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(-1, self.batch_size, self.num_mel)
        return data.to(self.device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len]
        return data, target

    def mel_spectrogram(self, audio_path):
        sound_clip, fn_fs = preprocess.read_audio(audio_path, target_fs=None, duration=None)

        ### sanity checks: ################################################################
        if sound_clip.shape[0] < self.fft_size:
            print("File %s is shorter than fft size - DISCARDING" % audio_path)

        if sound_clip.shape[0] == 0:
            print("File %s is corrupted!" % fn)

        assert (int(fn_fs) == self.sample_rate)
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
        return X.astype(np.float32)


    def get_batches(self):
        batch_cnt = 0

        for file_idx, audio_path in enumerate(self.audio_files):
            X = self.mel_spectrogram(audio_path)

            batchified = self.batchify(torch.FloatTensor(X))

            for batch, i in enumerate(range(0, batchified.size(0) - 1, self.bptt)):
                source, targets = self.get_batch(batchified, i)

                source.sub_(self.mean_).div_(self.var_)
                targets.sub_(self.mean_).div_(self.var_)

                yield (source, targets)

                batch_cnt += 1
                
            
            if self.max_files and file_idx >= (self.max_files - 1):
                break

        if self.num_batches is None:
            self.num_batches = batch_cnt