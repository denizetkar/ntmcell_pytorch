import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, use_layer_norm=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_to_gates_layer = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.hidden_to_gates_layer = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        if use_bias:
            self.gates_bias = nn.Parameter(torch.zeros(4 * hidden_dim))
        else:
            self.gates_bias = torch.zeros(4 * hidden_dim)

        if use_layer_norm:
            self.layer_norm_i = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-10) for _ in range(3)])
            self.layer_norm_h = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-10) for _ in range(3)])
        else:
            self.layer_norm_i = nn.ModuleList([nn.Identity() for _ in range(3)])
            self.layer_norm_h = nn.ModuleList([nn.Identity() for _ in range(3)])

    def create_new_state(self, batch_size):
        hx = torch.zeros((batch_size, self.hidden_dim))
        return hx

    def forward(self, inp, hx):
        i_gates = self.input_to_gates_layer(inp)
        h_gates = self.hidden_to_gates_layer(hx)
        i_gate_chunks = list(i_gates.chunk(3, dim=-1))
        h_gate_chunks = list(h_gates.chunk(3, dim=-1))
        gate_bias_chunks = list(self.gates_bias.chunk(4, dim=-1))
        for gate_index in range(3):
            i_gate_chunks[gate_index] = self.layer_norm_i[gate_index](i_gate_chunks[gate_index])
            h_gate_chunks[gate_index] = self.layer_norm_h[gate_index](h_gate_chunks[gate_index])

        resetgate = torch.sigmoid(i_gate_chunks[0] + h_gate_chunks[0] + gate_bias_chunks[0])
        updategate = torch.sigmoid(i_gate_chunks[1] + h_gate_chunks[1] + gate_bias_chunks[1])
        newgate_i = i_gate_chunks[2] + gate_bias_chunks[2]
        newgate_h = h_gate_chunks[2] + gate_bias_chunks[3]
        newgate = torch.tanh(newgate_i + resetgate * newgate_h)

        hy = (1 - updategate) * newgate + updategate * hx

        return hy, hy
