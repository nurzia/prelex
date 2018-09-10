import argparse
import random
import glob
import os
import shutil
import time

import numpy as np

import torch
import torch.nn as nn

from modelling import LanguageModel, repackage_hidden
import audio_utilities
from audio_generation import AudioGenerator



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
    parser.add_argument('--max_files', type=int, default=10)
    parser.add_argument('--max_gen_len', type=int, default=3000)
    parser.add_argument('--lowcut', type=int, default=70)
    parser.add_argument('--highcut', type=int, default=8000)

    # model:
    parser.add_argument('--bptt', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--log-interval', type=int, default=3)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #device = torch.device('cuda' if args.cuda else 'cpu')

    # build the bidirectional language model:
    model = LanguageModel(bptt=args.bptt, input_dim=args.num_freq,
                       hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # create mel filterbank:
    linear_bin_count = 1 + args.fft_size // 2
    filterbank = audio_utilities.make_mel_filterbank(args.lowcut, args.highcut, args.num_freq,
                                                     linear_bin_count, 44100)

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
    generator.fit(normalize=False)

    try:
        model.train()

        for epoch in range(args.epochs):
            total_loss = 0.
            epoch_losses = []
            start_time = time.time()        

            hidden = model.init_hidden(args.batch_size)

            for batch_idx, batch in enumerate(generator.get_batches(normalize=False)):
                in_ = torch.FloatTensor(batch['forward_in'])
                out_ = torch.FloatTensor(batch['forward_out'])

                # last batch might have deviant size:
                if in_.shape[0] != hidden[0].shape[0]:
                    hidden = model.init_hidden(in_.shape[0])
                
                bsz = in_.shape[0]

                hidden = repackage_hidden(hidden)

                model.zero_grad()
                output, hidden = model(in_, hidden)
                output = output.view(bsz * args.bptt, -1)

                loss = criterion(output, out_.view(bsz * args.bptt, -1))
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()

                total_loss += loss.item()
                epoch_losses.append(loss.item())

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    cur_loss = total_loss / args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                            'loss {:5.2f}'.format(
                        epoch, batch_idx, generator.num_batches,
                        elapsed * 1000 / args.log_interval, cur_loss))
                    total_loss = 0
                    start_time = time.time()

            print(f'Average loss in epoch {epoch}: {np.mean(epoch_losses)}')
            generator.generate(epoch, model, args.max_gen_len, normalize=False)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()