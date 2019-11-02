from .grucell import GRUCell
from ..cell_stacker import CellStacker


class GRU(CellStacker):
    def __init__(self, input_dim, hidden_dim, num_layers=1, use_bias=True, batch_first=False, dropout=0.0,
                 use_layer_norm=True):
        super(GRU, self).__init__(input_dim, hidden_dim, GRUCell, num_layers, use_bias, batch_first, dropout,
                                  use_layer_norm)
