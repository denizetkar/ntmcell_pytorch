import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True, use_layer_norm=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_to_gates_layer = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hidden_to_gates_layer = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        if use_bias:
            self.gates_bias = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.gates_bias = torch.zeros(4 * hidden_size)

        if use_layer_norm:
            self.layer_norm_i = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-10) for _ in range(4)])
            self.layer_norm_h = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-10) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size, eps=1e-10)
        else:
            self.layer_norm_i = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_h = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def create_new_state(self, batch_size):
        hx = torch.zeros((batch_size, self.hidden_size))
        cx = torch.zeros((batch_size, self.hidden_size))
        return hx, cx

    def forward(self, inp, state):
        hx, cx = state
        i_gates = self.input_to_gates_layer(inp)
        h_gates = self.hidden_to_gates_layer(hx)
        i_gate_chunks = list(i_gates.chunk(4, dim=-1))
        h_gate_chunks = list(h_gates.chunk(4, dim=-1))
        gate_bias_chunks = list(self.gates_bias.chunk(4, dim=-1))
        for gate_index in range(4):
            i_gate_chunks[gate_index] = self.layer_norm_i[gate_index](i_gate_chunks[gate_index])
            h_gate_chunks[gate_index] = self.layer_norm_h[gate_index](h_gate_chunks[gate_index])

        ingate = torch.sigmoid(i_gate_chunks[0] + h_gate_chunks[0] + gate_bias_chunks[0])
        forgetgate = torch.sigmoid(i_gate_chunks[1] + h_gate_chunks[1] + gate_bias_chunks[1])
        cellgate = torch.tanh(i_gate_chunks[2] + h_gate_chunks[2] + gate_bias_chunks[2])
        outgate = torch.sigmoid(i_gate_chunks[3] + h_gate_chunks[3] + gate_bias_chunks[3])

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.layer_norm_c(cy))

        return hy, (hy, cy)
