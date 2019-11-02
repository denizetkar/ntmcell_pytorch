import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gru import GRU
from ..lstm import LSTM


class Controller(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Controller, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def size(self):
        return self.input_dim, self.output_dim


class LSTMController(Controller):
    """An NTM controller based on LSTM."""

    def __init__(self, input_dim, output_dim, num_layers=1, dropout=0.1):
        super(LSTMController, self).__init__(input_dim, output_dim)

        self.num_layers = num_layers
        self.lstm = LSTM(input_dim,
                         output_dim,
                         num_layers=num_layers,
                         dropout=dropout)

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = torch.zeros((self.num_layers, batch_size, self.output_dim))
        lstm_c = torch.zeros((self.num_layers, batch_size, self.output_dim))
        return lstm_h, lstm_c

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


class GRUController(Controller):
    """An NTM controller based on GRU."""

    def __init__(self, input_dim, output_dim, num_layers=1, dropout=0.1):
        super(GRUController, self).__init__(input_dim, output_dim)

        self.num_layers = num_layers
        self.gru = GRU(input_dim,
                       output_dim,
                       num_layers=num_layers,
                       dropout=dropout)

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        gru_h = torch.zeros((self.num_layers, batch_size, self.output_dim))
        return gru_h

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.gru(x, prev_state)
        return outp.squeeze(0), state


class MLPController(Controller):
    """An NTM controller based on multilayer perceptron."""

    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super(MLPController, self).__init__(input_dim, output_dim)

        if hidden_layers is None:
            hidden_layers = [(128, 0.1, True), (128, 0.1, True)]

        hidden_layer_sizes, dropout_rates, use_batch_layers = zip(*hidden_layers)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for dropout in dropout_rates])
        # define network layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_layer_sizes[0])] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                                                             for i in range(len(hidden_layers) - 1)])
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_layer_size) if use_batch_layers[layer_num] else nn.Identity()
             for layer_num, hidden_layer_size in enumerate(hidden_layer_sizes)])
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_dim)

    def create_new_state(self, batch_size):
        return None

    def forward(self, x, prev_state):
        try:
            for hidden_layer, bn_layer, dropout in zip(self.hidden_layers, self.bn_layers, self.dropouts):
                x = F.selu(hidden_layer(x))
                x = dropout(bn_layer(x))
        except ValueError:
            # Error is most likely due to giving 1 sample to batchnorm layer
            return None, None

        return self.output_layer(x), None
