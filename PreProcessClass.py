from __future__ import print_function
import numpy as np
import librosa
import librosa.display

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob, os
import itertools

from numpy import linalg as LA

from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import Dense, Dropout, Bidirectional,\
    Input, Lambda, Embedding, LSTM, Flatten, TimeDistributed,\
    Activation

from keras.utils import to_categorical
from keras import regularizers
from keras import layers




class PreProcessClass:
    
    

    def __init__(self):
  

        "constructor function"

        return
        





    def get_signal (self, filename, duration = None):


        "function that gets data out of a .wav file that is passed as input and makes it mono"

        y, sr = librosa.load(path=filename, sr=None, mono=False)
        mono_signal = librosa.to_mono(y)

        if duration != None and isinstance(duration, int) == True: 
            mono_signal = mono_signal[0:sr*duration]

        return mono_signal, sr






    def get_markers (self, signal, sr, intervals):


        "function that marks each time-step as either relevant or non-relevant"
        
        mark_incr = 1000./float(sr)

        markers = np.zeros(len(signal))

        for i, interval in enumerate(intervals):
            first = int(interval[0]/mark_incr)
            last = int(interval[1]/mark_incr)
            markers[first:last]=1

        return markers 






    def pre_emphasis (self, signal):


        "function that applies a pre-emphasis filter to amplify high frequencies"

        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        return emphasized_signal






    def spectrogram (self, signal, sr, num_frames, hop_length, num_freq):


        "function that creates a spectrogram of the input signal"

        spectrum = librosa.core.stft(signal, n_fft=num_frames, hop_length=hop_length, win_length=None, window='hann', center=False)

        S = librosa.feature.melspectrogram(y=None, sr=sr, S=spectrum, n_fft=num_frames, hop_length=hop_length, n_mels=num_freq)
        
        log_S = librosa.core.power_to_db(S)

        return log_S






    def mark_spectro (self, spectro_length, sr, num_frames, hop_length, markers, threshold = 0.5):


        "function that calculates the overlap from the previously computed markers"

        overlaps = np.array([])

        #compute which percentage of each spectrogram overlaps a babble
        for i in range(0, len(markers), hop_length):
            overlaps = np.append(overlaps, np.mean(markers[i:i+num_frames]))

        #if spectrogram overlaps a babble for more than a given threshold label as "yes" otherwise as "no"
        for overlap in overlaps:
            if overlap < threshold:
                overlap = 0
            else:
                overlap = 1

        if spectro_length < len(markers):
            overlaps=overlaps[:spectro_length]

        return overlaps






    def audio_file_names (self, ratio, main_folder="/home/nurzia/AUDIO/"):


        "function that gets the names of the audio-files for the 'RANDOM' scenario"

        files = glob.glob(main_folder+"**/*.wav")
        train_len = int(ratio*len(files))

        train_files = files[:train_len]

        dev_len = train_len + int((len(files) - train_len)/2)
        dev_files = files[train_len:dev_len]

        test_files = files[dev_len:len(files)]

        return train_files, dev_files, test_files






    def audio_file_names_dir (self, ratio, main_folder="/home/nurzia/AUDIO/"):


        "function that gets the names of the audio-files for che 'CHILDREN' scenario"

        all_dirs = os.listdir(main_folder)

        directories = []
        for all_dir in all_dirs:
            direc = all_dir
            all_dir = main_folder+direc+'/'
            directories.append(all_dir)

        train_len = int(ratio*len(all_dirs))
        train_dirs = directories[:train_len]

        dev_len = train_len + int((len(all_dirs) - train_len)/2)
        dev_dirs = directories[train_len:dev_len]

        test_dirs = directories[dev_len:len(all_dirs)]
 
        train_files = []
        for train_dir in train_dirs:
            train_files.append(glob.glob(train_dir+'*.wav'))
        train_files = [item for sublist in train_files for item in sublist]

        dev_files = []
        for dev_dir in dev_dirs:
            dev_files.append(glob.glob(dev_dir+'*.wav'))
        dev_files = [item for sublist in dev_files for item in sublist]

        test_files = []
        for test_dir in test_dirs:
            test_files.append(glob.glob(test_dir+'*.wav'))
        test_files = [item for sublist in test_files for item in sublist]

        return train_files, dev_files, test_files






    def transcription_file_names (self, audio_files, main_folder = "/home/nurzia/TRANSCRIPTION/"):


        "function that gets the names of the transcriptions"

        transcriptions = np.array([])
        
        for audio_file in audio_files:
             path = main_folder+audio_file[19:-4]+'_S.cha'
             transcriptions = np.append(transcriptions, path)

        return transcriptions






    def normalize (self, S):


        "function that mean-normalizes the input"

        norm_S = S
        mean_vector=np.mean(norm_S, axis=0)
        variance=np.var(norm_S, axis=0)
        
        for frame in norm_S:
            frame -= mean_vector
            frame = np.divide(norm_S, variance) 

        return norm_S






    def process_data (self, audio_files, seq_len, num_frames, hop_length, num_freq):


        "function that gets all the input from a set of audiofile names"

        transcriptions = self.transcription_file_names (audio_files)
        all_spectros = []
        all_overlaps = []
        all_means = []

        #loop over all audiofile names in set
        for index, audio_file in enumerate(audio_files):
            
            # get waveform out of audiofile
            signal, sr = self.get_signal(filename=audio_file) 

            #get spectrogram of audiofile
            log_S = np.transpose(np.array(self.spectrogram(signal, sr, num_frames, hop_length, num_freq))) 

            #normalize spectrogram
            log_S = self.normalize(log_S) 

            #cut signal short so that its length is divisible by seq_len
            if not len(log_S)%seq_len==0:
                log_S = log_S[:-(len(log_S)%seq_len)]

            #reshape spectrogram sequence as matrix of seq_len-long sequences of spectrograms
            log_S = log_S.reshape((int(len(log_S)/seq_len),seq_len,num_freq))
            log_S = log_S.astype(np.float32)

            #get intervals out of transcription file
            intervals = self.get_intervals(transcriptions[index])

            #build binary vector as long as waveform to mark what is in or out of a babble
            markers = self.get_markers(signal, sr, intervals)

            #label each spectrogram as babble or non-babble
            overlaps = self.mark_spectro(len(log_S)*seq_len, sr, num_frames, hop_length, markers)

            #reshape labels sequence as matrix of seq_len-long sequences of labels
            overlaps = overlaps.reshape((int(len(overlaps)/seq_len), seq_len,1))
            overlaps = overlaps.astype(np.int32)

            #append matrix obtained from a single file to general matrix obtained from all preceding files
            all_spectros.append(log_S)
            all_overlaps.append(overlaps)

            #compute average length of babble for the given file
            av_blen = self.average_blength(markers=markers, sr = sr)
            all_means.append(av_blen)

        #transform X and Y sets into numpy entity
        all_spectros = np.vstack(all_spectros)
        all_overlaps = np.vstack(all_overlaps)

        #compute average length of babble over the whole set
        all_means = np.array(all_means)
        mean = np.mean(all_means)

        return all_spectros, all_overlaps






    def get_data(self, scenario = 'children', ratio = 0.8, seq_len = 50, num_frames = 44100, hop_length = 44100, num_freq = 128):


        "container function that calls all the other ones"

        #use a different function to get the audiofile names in train-, dev- and test- sets depending which scenario one sets
        if scenario == 'random':
            train_files, dev_files, test_files = self.audio_file_names (ratio=ratio)

        elif scenario == 'children':
            train_files, dev_files, test_files = self.audio_file_names_dir (ratio=ratio)


        #get a matrix of spectrogram sequences and one of labels for each of the 3 sets
        train_X, train_Y = self.process_data(audio_files = train_files, seq_len = seq_len, 
                                              num_frames = num_frames, hop_length = hop_length, 
                                              num_freq = num_freq)

        dev_X, dev_Y = self.process_data(audio_files = dev_files, seq_len = seq_len, 
                                              num_frames = num_frames, hop_length = hop_length, 
                                              num_freq = num_freq)

        test_X, test_Y = self.process_data(audio_files = test_files, seq_len = seq_len, 
                                              num_frames = num_frames, hop_length = hop_length, 
                                              num_freq = num_freq)

        #save the 3 sets on 3 different files
        np.save('train_X', train_X)
        np.save('train_Y', train_Y)
        np.save('dev_X', dev_X)
        np.save('dev_Y', dev_Y)
        np.save('test_X', test_X)
        np.save('test_Y', test_Y)

        return train_X, train_Y, dev_X, dev_Y, test_X, test_Y






    def get_intervals(self, fn, allowed={'I0', 'I1', 'I2', 'U0', 'U1', 'U2'}):

 
        "function that gets all the time intervals out of the transcription"

        
        chi_started = None

        all_intervals = []

        for line in open(fn, 'r'):

            if line =='@End':
                break

            line = line.strip()

            if not line or line.startswith('@'):
                continue

            if line.startswith('*CHI:\tv@.'):
                try:
                    chi_started = [int(i) for i in line[:-1].split('_')[-2:]]
                except (ValueError, IndexError):
                    print("parsing error " + line)
                    pass

            elif line.startswith(('*MOT:\t', '*FAT:\t')):
                chi_started = None

            elif line.startswith('%voc:\t'):
                code = line.split('\t')[-1].strip()

                if chi_started and code in allowed:
                    all_intervals.append(chi_started)
                    chi_started = None

        return np.array(all_intervals)






    def average_blength(self, markers, sr = 44100):


        "function that computes the average length of a babble"

        condition = markers == 1

        mean_len = np.mean([ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ])
 
        return mean_len/sr






def build_model(seq_len=50,
               fft_dim=128,
               recurrent_dim=100,
               learning_rate=0.001,
               dr_ou = None,
               num_layers=1):


    "function building a model with n BLSTMs with m parameters and n BLSTMs with m/2 parameters and a final TDD that gives the output"
    
    #define optimizer
    optim = Adam(lr=learning_rate)

    #define model
    model = Sequential()

    for i in range(num_layers):
        #add num_layers layers to the model with recurrent_dim number of parameters
        model.add(Bidirectional(LSTM(recurrent_dim, return_sequences=True), input_shape = (seq_len, fft_dim)))
        #execute dropout if desired
        if dr_ou != None:
            model.add(layers.Dropout(rate = dr_ou))
    for i in range(num_layers):
        #add num_layers layers to the model with recurrent_dim/2 number of parameters
        model.add(Bidirectional(LSTM(int(recurrent_dim/2), return_sequences=True), input_shape = (seq_len, fft_dim)))
        #execute dropout if desired
        if dr_ou != None:
            model.add(layers.Dropout(rate = dr_ou))

    #add final time-distributed dense layer that gives the output
    model.add(TimeDistributed(Dense(2, activation = 'softmax', name='out')))

    #compile model
    model.compile(optimizer=optim,
                  metrics=['accuracy'],
                  loss={'categorical_crossentropy'})
    return model






