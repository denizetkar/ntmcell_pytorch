import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gru import GRU
from ..lstm import LSTM


class Controller(nn.Module):
    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def size(self):
        return self.input_size, self.output_size


class LSTMController(Controller):
    """An NTM controller based on LSTM."""

    def __init__(self, input_size, output_size, **other_controller_params):
        super(LSTMController, self).__init__(input_size, output_size)

        if not other_controller_params:
            other_controller_params = dict(num_layers=1, dropout=0.1)
        self.lstm = LSTM(input_size, output_size, **other_controller_params)

    def create_new_state(self, batch_size):
        return self.lstm.create_new_state(batch_size)

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


class GRUController(Controller):
    """An NTM controller based on GRU."""

    def __init__(self, input_size, output_size, **other_controller_params):
        super(GRUController, self).__init__(input_size, output_size)

        if not other_controller_params:
            other_controller_params = dict(num_layers=1, dropout=0.1)
        self.gru = GRU(input_size, output_size, **other_controller_params)

    def create_new_state(self, batch_size):
        return self.gru.create_new_state(batch_size)

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.gru(x, prev_state)
        return outp.squeeze(0), state


class MLPController(Controller):
    """An NTM controller based on multilayer perceptron."""

    def __init__(self, input_size, output_size, hidden_layers=None):
        super(MLPController, self).__init__(input_size, output_size)

        if hidden_layers is None:
            hidden_layers = [(128, 0.1, True), (128, 0.1, True)]

        hidden_layer_sizes, dropout_rates, use_batch_layers = zip(*hidden_layers)
        self.dropouts = nn.ModuleList([nn.AlphaDropout(dropout) for dropout in dropout_rates])
        # define network layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layer_sizes[0])] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                                                             for i in range(len(hidden_layers) - 1)])
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_layer_size) if use_batch_layers[layer_num] else nn.Identity()
             for layer_num, hidden_layer_size in enumerate(hidden_layer_sizes)])
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)

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
