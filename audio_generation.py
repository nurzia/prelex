import glob
import os
import shutil
import json
from tqdm import tqdm
import random
random.seed(13455)

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import preprocess
import librosa
from scipy import signal

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import cluster
from sklearn import svm, linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb
# pipeline: wave > spectrogram > mel > log > normalize


class AudioGenerator(object):

    def __init__(self, audio_dir, fft_size, hop, num_mel, bptt, batch_size,
                 device, spectro_dir, mean_=None, var_=None, lowcut=70, highcut=8000,
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
        self.spectro_dir = spectro_dir

        if mean_ is not None and var_ is not None:
            self.mean_ = torch.FloatTensor(mean_).to(self.device)
            self.var_ = torch.FloatTensor(var_).to(self.device)

        self.melW = librosa.filters.mel(sr=self.sample_rate, n_fft=self.fft_size,
                                        n_mels=self.num_mel,
                                        fmin=self.lowcut, fmax=self.highcut)
        self.ham_win = np.hamming(self.fft_size)

        # declare a number of properties:
        self.num_batches = None
        self.train_files = None
        self.test_files = None
        self.fitted_ = False

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
        scaler = StandardScaler() # only used for fitting because of handy partial_fit method

        print('Fitting scaler...')
        tqdm_ = tqdm(self.get_batches(), total=self.num_batches)
        for batch, _ in tqdm_:
            batch = batch.cpu().numpy()
            flat_batch = batch.reshape((-1, self.num_mel))
            scaler.partial_fit(flat_batch)

        self.mean_ = torch.FloatTensor(scaler.mean_).to(self.device)
        self.var_ = torch.FloatTensor(scaler.var_).to(self.device)

        self.fitted_ = True

        return self

    def dump_spectrograms(self):
        children = sorted(glob.glob(self.audio_dir + '/*'))
        if self.max_children:
            children = children[:self.max_children]

        audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(audio_folder + '/*.wav'):
                audio_files.append(audio_file)
        audio_files = audio_files[:self.max_files]

        try:
            shutil.rmtree(self.spectro_dir)
        except FileNotFoundError:
            pass
        os.mkdir(self.spectro_dir)

        print('Preprocessing to mel-spectrograms...')
        for file_idx, audio_path in enumerate(tqdm(audio_files)):
            X = self.mel_spectrogram(audio_path)
            newp = os.path.basename(audio_path).replace('.wav', '.mel')
            newp = os.sep.join((self.spectro_dir, newp))
            np.save(newp, X)


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

    def melspec_from_file(self, fp, mode='all', min_len=75):
        X = self.mel_spectrogram(fp)

        if len(X) < min_len:
            return None
        else:
            X = torch.FloatTensor(X).to(self.device)
            X.sub_(self.mean_).div_(self.var_)        
            return X.cpu().numpy()

    def hidden_from_file(self, model, fp, mode='all', min_len=75):
        X = torch.FloatTensor(self.mel_spectrogram(fp))

        if len(X) < min_len:
            return None

        X = X.to(self.device)
        X.sub_(self.mean_).div_(self.var_)        

        hidden_states = []
        with torch.no_grad():
            hidden = None
            for x in X:
                _, hidden = model.lstm(x.view(1, 1, -1), hidden)
                hidden_states.append(hidden[0][-1].squeeze().cpu().numpy())

        hidden_states = np.array(hidden_states)
        if mode == 'mean':
            return hidden_states.mean(axis=0)
        elif mode == 'last':
            reprs.append(hidden_states[-1])
        elif mode == 'all':
            return hidden_states

    def baseline(self, mode='all'):
        assert self.train_files is not None
        assert mode in ('all', 'mean')

        train_labels_true, train_reprs = [], []
        for fp in self.train_files:
            cat = os.path.basename(fp).split('_')[0]
            hidden = self.melspec_from_file(fp)
            if hidden is None:
                continue

            if mode == 'mean':
                train_reprs.append(hidden.mean(axis=0))
                train_labels_true.append(cat)
            else:
                train_reprs.extend(hidden)
                train_labels_true.extend([cat] * len(hidden))    

        train_reprs = np.array(train_reprs)
        label_encoder = LabelEncoder()
        train_labels_true_int = label_encoder.fit_transform(train_labels_true)

        clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
        print('training logreg...')
        clf.fit(train_reprs, train_labels_true_int)

        print('predicting...')
        test_labels_true, test_labels_pred = [], []
        for fp in self.test_files:
            hidden = self.melspec_from_file(fp)

            if hidden is None:
                continue

            cat = os.path.basename(fp).split('_')[0]
            test_labels_true.append(cat)

            if mode == 'all':
                proba = clf.predict_proba(np.array(hidden)).mean(axis=0)
            else:
                proba = clf.predict([np.array(hidden).mean(axis=0)])[0]

            lab = label_encoder.classes_[proba.argmax()]
            test_labels_pred.append(lab)
            if lab != cat:
                print(f'    - wrongly predicted file: ({lab} instead of {cat}): {fp}')

        print(classification_report(test_labels_true, test_labels_pred))


    def external_validation(self, model, mode='all'):
        assert mode in ('mean', 'last', 'all')
        model.eval()

        if self.train_files is None:
            chi_files = list(glob.glob('assets/UTT/chi_utt/*.wav'))#[:100]
            adu_files = list(glob.glob('assets/UTT/adu_utt/*.wav'))#[:100]

            both_files = chi_files + adu_files
            random.shuffle(both_files)
            self.train_files = sorted(both_files[:int(len(both_files) / 100 * 90)])
            self.test_files = sorted(both_files[int(len(both_files) / 100 * 90):])

        train_labels_true, train_reprs = [], []
        for fp in self.train_files:
            cat = os.path.basename(fp).split('_')[0]
            hidden = self.hidden_from_file(model, fp, mode)

            if hidden is None:
                continue

            if mode in ('mean', 'last'):
                train_reprs.append(hidden)
                train_labels_true.append(cat)
            else:
                train_reprs.extend(hidden)
                train_labels_true.extend([cat] * len(hidden))    

        train_reprs = np.array(train_reprs)
        label_encoder = LabelEncoder()
        train_labels_true_int = label_encoder.fit_transform(train_labels_true)

        clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
        print('training logreg...')
        clf.fit(train_reprs, train_labels_true_int)

        print('predicting...')
        test_labels_true, test_labels_pred = [], []
        for fp in self.test_files:
            hidden = self.hidden_from_file(model, fp, mode)
            if hidden is None:
                continue

            cat = os.path.basename(fp).split('_')[0]
            test_labels_true.append(cat)

            if mode == 'all':
                proba = clf.predict_proba(np.array(hidden)).mean(axis=0)
            else:
                proba = clf.predict([np.array(hidden)])[0]

            lab = label_encoder.classes_[proba.argmax()]
            test_labels_pred.append(lab)
            if lab != cat:
                print(f'    - wrongly predicted file: ({lab} instead of {cat}): {fp}')

        print(classification_report(test_labels_true, test_labels_pred))

    def mel_spectrogram(self, audio_path):
        sound_clip, fn_fs = preprocess.read_audio(audio_path, target_fs=None)

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

    def batchify(self, data):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(self.batch_size, -1, self.num_mel).transpose(0, 1).contiguous()
        return data.to(self.device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len]
        return data, target

    def get_batches(self):
        batch_cnt = 0

        for file_idx, mel_path in enumerate(glob.glob(self.spectro_dir + '/*.mel.npy')):
            X = np.load(mel_path)

            batchified = self.batchify(torch.FloatTensor(X))

            for batch, i in enumerate(range(0, batchified.size(0) - 1, self.bptt)):
                source, targets = self.get_batch(batchified, i)

                if self.fitted_:
                    source.sub_(self.mean_).div_(self.var_)
                    targets.sub_(self.mean_).div_(self.var_)

                yield (source, targets)

                batch_cnt += 1
                
            
            if self.max_files and file_idx >= (self.max_files - 1):
                break

        if self.num_batches is None:
            self.num_batches = batch_cnt