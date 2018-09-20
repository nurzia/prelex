import argparse
import random
import glob
import os
import shutil
import time

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

import modelling
from modelling import repackage_hidden
from audio_generation import AudioGenerator


def main():
    parser = argparse.ArgumentParser()

    # data:
    parser.add_argument('--audio_dir', type=str, default='assets/AUDIO')#'/home/nurzia/AUDIO')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing:
    parser.add_argument('--fft_size', type=int, default=1024) # expressed in samples, not miliseconds!
    parser.add_argument('--hop', type=int, default=512) # expressed in samples, not miliseconds!
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--num_mel', type=int, default=128) # feature dim of the log mel spectrogram https://blogs.technet.microsoft.com/machinelearning/2018/01/30/hearing-ai-getting-started-with-deep-learning-for-audio-on-azure/
    parser.add_argument('--max_children', type=int, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_gen_len', type=int, default=300) # expressed in frames, consisting of fft size samples
    parser.add_argument('--lowcut', type=int, default=0)
    parser.add_argument('--highcut', type=int, default=8000)
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--norm', action='store_true', default=False)

    # model:
    parser.add_argument('--bptt', type=int, default=256)  # expressed in frames, consisting of fft size samples
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--clip', type=float, default=5.)
    parser.add_argument('--log-interval', type=int, default=100)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = modelling.LanguageModel(input_dim=args.num_mel, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # create a "lazy" batch generator:
    generator = AudioGenerator(audio_dir=args.audio_dir,
                               fft_size=args.fft_size,
                               hop=args.hop,
                               num_mel=args.num_mel,
                               max_children=args.max_children,
                               max_files=args.max_files,
                               bptt=args.bptt,
                               batch_size=args.batch_size,
                               device=device)
    generator.fit(normalize=args.norm)

    try:
        lowest_loss = np.inf

        for epoch in range(args.epochs):
            total_loss = 0.
            epoch_losses = []
            start_time = time.time()        

            hidden = model.init_hidden(args.batch_size)

            for batch_idx, (source, targets) in enumerate(tqdm(generator.get_batches(normalize=args.norm),
                                                   total=generator.num_batches)):
                model.train()
                
                hidden = repackage_hidden(hidden)
                hidden = tuple([t.to(device) for t in hidden])

                model.zero_grad()

                output, hidden = model(source, hidden)

                output = output.view(-1, args.num_mel)
                loss = criterion(output, targets.view(-1, args.num_mel))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                optimizer.step()

                total_loss += loss.item()
                epoch_losses.append(loss.item())

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    cur_loss = total_loss / args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} series | ms/batch {:5.2f} | '
                            'loss {:5.5f}'.format(
                        epoch, batch_idx, generator.num_batches,
                        elapsed * 1000 / args.log_interval, cur_loss))
                    total_loss = 0
                    start_time = time.time()

            epoch_loss = np.mean(epoch_losses)
            print(f'Average loss in epoch {epoch}: {epoch_loss}')
            generator.generate(epoch, model, args.max_gen_len, normalize=args.norm)

            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                torch.save(model, args.model_prefix + '_best.pth')

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()