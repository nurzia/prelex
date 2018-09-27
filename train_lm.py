import argparse
import random
import glob
import os
import shutil

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

import modelling
from modelling import repackage_hidden
from audio_generation import AudioGenerator

# GRowNG BPTT?


def main():
    parser = argparse.ArgumentParser()

    # data:
    parser.add_argument('--audio_dir', type=str, default='assets/TRIMMED_AUDIO')
    parser.add_argument('--spectro_dir', type=str, default='assets/SPECTROGRAMS')
    parser.add_argument('--model_prefix', type=str, default='lm')

    # preprocessing:
    parser.add_argument('--fft_size', type=int, default=640) # expressed in samples, not milliseconds!
    parser.add_argument('--hop', type=int, default=320) # expressed in samples, not milliseconds!
    parser.add_argument('--seed', type=int, default=26711)
    parser.add_argument('--num_mel', type=int, default=100) # feature dim of the log mel spectrogram https://blogs.technet.microsoft.com/machinelearning/2018/01/30/hearing-ai-getting-started-with-deep-learning-for-audio-on-azure/
    parser.add_argument('--max_children', type=int, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_gen_len', type=int, default=300) # expressed in frames, consisting of fft size samples
    parser.add_argument('--lowcut', type=int, default=0)
    parser.add_argument('--highcut', type=int, default=8000)
    parser.add_argument('--extract', action='store_true', default=False)
    parser.add_argument('--sample_rate', type=int, default=32000)

    # model:
    parser.add_argument('--bptt', type=int, default=128)  # expressed in frames, consisting of fft size samples
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--clip', type=float, default=30.)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    model = modelling.LanguageModel(input_dim=args.num_mel, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers)
    model = model.to(args.device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss() #nn.MSELoss()

    generator = AudioGenerator(audio_dir=args.audio_dir,
                               fft_size=args.fft_size,
                               hop=args.hop,
                               num_mel=args.num_mel,
                               max_children=args.max_children,
                               max_files=args.max_files,
                               bptt=args.bptt,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate,
                               device=args.device,
                               spectro_dir=args.spectro_dir)
    if args.extract:
        generator.dump_spectrograms()
    generator.fit()

    # test whether we can dump and load the vectorizer:
    #generator.dump(args.model_prefix + '_vect')
    #del generator
    #generator = AudioGenerator.load(args.model_prefix + '_vect', args)

    print('==== RANDOM =====')
    generator.external_validation(model)
    print('==== BASELINE ===')
    generator.baseline()
    print('===========================')

    try:
        lowest_loss = np.inf

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []

            hidden = None

            tqdm_ = tqdm(generator.get_batches(),
                         total=generator.num_batches)
            tqdm_.set_description(f'Epoch {epoch + 1}')

            for batch_idx, (source, target) in enumerate(tqdm_):

                optimizer.zero_grad()

                output, hidden = model(source, hidden)

                output = output.view(-1, args.num_mel)
                target = target.view(-1, args.num_mel)

                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                epoch_losses.append(loss.item())
                tqdm_.set_postfix(loss=str(round(np.mean(epoch_losses), 6)))

                hidden = repackage_hidden(hidden)

            epoch_loss = np.mean(epoch_losses)

            generator.generate(epoch, model, args.max_gen_len)
            generator.external_validation(model)

            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                torch.save(model, args.model_prefix + '_best.pth')

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()