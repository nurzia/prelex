import torch
import torch.nn as nn
import torch.nn.functional as F


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class LanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_layers=1):
        super(LanguageModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_, hidden):
        output, hidden = self.lstm(input_, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

