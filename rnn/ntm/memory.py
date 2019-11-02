import torch
import torch.nn as nn
import torch.nn.functional as F


class NTMMemory(nn.Module):
    """Memory bank for NTM."""

    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M
        self.memory = None

    def reset(self, batch_size):
        self.memory = torch.zeros((batch_size, self.N, self.M))

    def size(self, i=None):
        if i:
            return self.memory.size(i)
        return self.memory.size()

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(-2), self.memory.unsqueeze(-3)).squeeze(-2)

    def mul(self, multiplier):
        self.memory *= multiplier

    def add(self, addition):
        self.memory += addition

    def write_erase(self, w, e):
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(-2))
        self.mul(1 - erase)

    def write_add(self, w, a):
        addition = torch.matmul(w.unsqueeze(-1), a.unsqueeze(-2))
        self.add(addition)

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        k = k.unsqueeze(-2)
        w = F.softmax(β * F.cosine_similarity(self.memory.unsqueeze(-3) + 1e-16, k + 1e-16, dim=-1), dim=-1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        """Circular convolution implementation."""
        assert s.size(-1) == 3
        t = torch.cat([wg[..., -1:], wg, wg[..., :1]], dim=-1)
        t = t.view(1, -1, t.shape[-1])
        s = s.view(-1, 1, s.shape[-1])
        c = F.conv1d(t, s, groups=t.shape[1]).view_as(wg)
        return c

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=-1, keepdim=True) + 1e-16)
        return w
