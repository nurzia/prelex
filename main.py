import argparse
import random
import glob
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical

import preprocess
import modelling

def main():
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--audio_dir', type=str, default='assets/AUDIO')
    parser.add_argument('--chat_dir', type=str, default='assets/TRANSCRIPTION')
    parser.add_argument('--data_dir', type=str, default='assets/preprocessed')
    parser.add_argument('--model_prefix', type=str, default='model')

    # preprocessing
    parser.add_argument('--frames', type=int, default=25)
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--hop', type=int, default=25)
    parser.add_argument('--max_freq', type=int, default=8000)
    parser.add_argument('--max_children', type=int, default=0)
    parser.add_argument('--max_timesteps', type=int, default=None)
    parser.add_argument('--preprocess', action='store_true', default=False)

    # model
    parser.add_argument('--bptt', type=int, default=30)
    parser.add_argument('--train_size', type=float, default=.75)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    children = sorted(glob.glob(f'{args.audio_dir}/*'))
    if args.max_children:
        children = children[:args.max_children]

    if args.preprocess:
        X, Y = [], []
        for audio_folder in children:
            child_name = os.path.basename(audio_folder)
            for audio_file in glob.glob(f'{audio_folder}/*.wav'):
                print(audio_file)
                # extract audio
                wave, sample_rate = preprocess.load_file(audio_file)
                spectrogram = preprocess.build_spectrogram(wave, sample_rate,
                                                          window=args.frames,
                                                          step=args.hop,
                                                          max_freq=args.max_freq)
                # extract transcription
                bn = os.path.basename(audio_file).replace('.wav', '')
                p = f'{args.chat_dir}/{child_name}/{bn}_S.cha'
                intervals = preprocess.extract_intervals(p)
                markers = preprocess.apply_intervals(wave, sample_rate, intervals)
                ints = preprocess.mark_spectro(spectrogram.shape[0], num_frames=args.frames,
                                    hop_length=args.hop, markers=markers)

                X.extend(spectrogram)
                Y.extend(ints)

                if args.max_timesteps and len(X) >= args.max_timesteps:
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

    print(X.shape)
    
    batch_sums = Y.sum(axis=1)
    non_empty = np.where(batch_sums > 0)[0]
    empty = np.where(batch_sums == 0)[0]
    empty = np.random.choice(empty, len(non_empty), replace=False)
    keep = tuple(sorted(list(non_empty) + list(empty)))

    X = X[keep, :, :]
    Y = Y[keep, :]

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

    model = modelling.build_model(bptt=args.bptt, input_dim=X.shape[-1],
                                  recurrent_dim=args.hidden_dim,
                                  lr=args.lr, num_layers=args.num_layers)
    model.summary()
    model.fit(X_train, Y_train_cat, epochs=args.epochs,
              validation_data=(X_dev, Y_dev_cat))

    predictions = model.predict(X_dev).argmax(axis=-1).ravel()
    print(classification_report(Y_dev.ravel(), predictions))


if __name__ == '__main__':
    main()