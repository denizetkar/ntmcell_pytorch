import torch
import torch.nn as nn
import torch.nn.functional as F


class NTMHeadBase(nn.Module):
    def __init__(self, N, M, controller_dim):
        super(NTMHeadBase, self).__init__()
        self.N = N
        self.M = M
        self.controller_dim = controller_dim

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros((batch_size, self.N))

    @staticmethod
    def _address_memory(k, β, g, s, γ, w_prev, memory):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        w = memory.address(k, β, g, s, γ, w_prev)

        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, N, M, controller_dim):
        super(NTMReadHead, self).__init__(N, M, controller_dim)
        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_dim, sum(self.read_lengths))

    def forward(self, controller_output, w_prev, memory):
        o = self.fc_read(controller_output)
        k, β, g, s, γ = o.split(self.read_lengths, dim=-1)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev, memory)
        r = memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, N, M, controller_dim):
        super(NTMWriteHead, self).__init__(N, M, controller_dim)
        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_dim, sum(self.write_lengths))

    def forward(self, controller_output, w_prev, memory):
        o = self.fc_write(controller_output)
        k, β, g, s, γ, e, a = o.split(self.write_lengths, dim=-1)

        # e should be in [0, 1]
        e = torch.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev, memory)

        return (w, e, a), w
