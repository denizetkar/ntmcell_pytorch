from .lstmcell import LSTMCell
from ..cell_stacker import CellStacker


class LSTM(CellStacker):
    def __init__(self, input_dim, hidden_dim, num_layers=1, use_bias=True, batch_first=False, dropout=0.0,
                 use_layer_norm=True):
        super(LSTM, self).__init__(input_dim, hidden_dim, LSTMCell, num_layers, use_bias, batch_first, dropout,
                                   use_layer_norm)
