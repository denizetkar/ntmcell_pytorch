import torch
import torch.nn as nn


class CellStacker(nn.Module):
    def __init__(self, input_size, hidden_size, cell_cls, num_layers=1, use_bias=True, batch_first=False, dropout=0.0,
                 use_layer_norm=None):
        super(CellStacker, self).__init__()
        assert num_layers > 0, 'number of layers is not bigger than 0 (%s)' % str(num_layers)

        if use_layer_norm is None:
            self.layer_cells = nn.ModuleList([cell_cls(input_size, hidden_size, use_bias)] +
                                             [cell_cls(hidden_size, hidden_size, use_bias)
                                              for _ in range(num_layers - 1)])
        else:
            self.layer_cells = nn.ModuleList([cell_cls(input_size, hidden_size, use_bias, use_layer_norm)] +
                                             [cell_cls(hidden_size, hidden_size, use_bias, use_layer_norm)
                                              for _ in range(num_layers - 1)])
        self.batch_first = batch_first
        self.dropout_layer = nn.Dropout(dropout)

    def create_new_state(self, batch_size):
        states = (layer_cell.create_new_state(batch_size) for layer_cell in self.layer_cells)
        return states

    def forward(self, input_seq, prev_states):
        # input_seq: [seq_len, batch_size, input_size] OR [batch_size, seq_len, input_size]
        seq_len_dim_index = int(self.batch_first)
        seq_len = input_seq.shape[seq_len_dim_index]

        output_seq = [None for _ in range(seq_len)]
        states = list(prev_states)
        for i in range(seq_len):
            inp = input_seq.select(dim=seq_len_dim_index, index=i)
            # propagate 'inp' through each layer cell
            inp, states[0] = self.layer_cells[0](inp, states[0])
            for layer_num in range(1, len(self.layer_cells)):
                inp = self.dropout_layer(inp)
                inp, states[layer_num] = self.layer_cells[layer_num](inp, states[layer_num])
            output_seq[i] = inp.unsqueeze(seq_len_dim_index)

        return torch.cat(output_seq, dim=seq_len_dim_index), tuple(states)
