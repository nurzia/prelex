import argparse
import random
import glob
import os
import shutil

import numpy as np

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import preprocess
import utils
import modelling

# TO DO:
# - statefulness?
# - check truncation issue
# - sample from the model and convert generated spectrogram back to audio


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

        children = sorted(glob.glob(f'{self.audio_dir}/*'))
        if self.max_children:
            children = children[:self.max_children]

        self.audio_files = []
        for audio_folder in children:
            for audio_file in glob.glob(f'{audio_folder}/*.wav'):
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
                
                # divide into batches:
                num_batches = X.shape[0] // (self.bptt + 1)
                X = X[:num_batches * (self.bptt + 1)]
                X = X.reshape((num_batches, -1, X.shape[1]))

                # start yielding batches:
                forward_in_batch, forward_out_batch = [], []
                backward_in_batch, backward_out_batch = [], []

                for batch_idx, batch in enumerate(X):

                    l, r = batch[:-1, :], batch[1:, :]

                    forward_in_batch.append(l)
                    forward_out_batch.append(r)

                    backward_in_batch.append(r[::-1, :])
                    backward_out_batch.append(l[::-1, :])

                    if len(forward_in_batch) >= self.batch_size:
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
    parser.add_argument('--audio_dir', type=str, default='assets/AUDIO')
    parser.add_argument('--chat_dir', type=str, default='assets/TRANSCRIPTION')
    parser.add_argument('--data_dir', type=str, default='assets/preprocessed')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing:
    parser.add_argument('--frames', type=int, default=256) # 256
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--num_freq', type=int, default=64)
    parser.add_argument('--max_children', type=int, default=1)
    parser.add_argument('--max_files', type=int, default=2)

    # model:
    parser.add_argument('--bptt', type=int, default=10)
    parser.add_argument('--train_size', type=float, default=.75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.001)

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
    
    # idle loop over the generator to know how many batches we have:
    for batch in generator.get_batches(endless=False):
        pass

    # define callbacks for model fitting:
    checkpoint = ModelCheckpoint(args.model_prefix + '.model', monitor='loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, min_delta=0.03)

    # fit the model:
    try:
        model.fit_generator(generator=generator.get_batches(endless=True),
                            steps_per_epoch=generator.num_batches,
                            epochs=args.epochs,
                            callbacks=[checkpoint, reduce_lr])
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()