import torch
import torch.nn as nn

from .controller import GRUController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory
from .utils import tensor_to_device


class NTMCell(nn.Module):
    def __init__(self, input_dim, output_dim, N=32, M=8, num_read_heads=1, num_write_heads=1, controller_dim=64,
                 controller_cls=GRUController, other_controller_params=None):
        if other_controller_params is None:
            other_controller_params = {}
        assert num_read_heads > 0 and num_write_heads > 0, "heads list must contain at least a single read head"

        super(NTMCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.M = M
        self.read_heads = NTMReadHead(N, M, num_read_heads, controller_dim)
        self.write_heads = NTMWriteHead(N, M, num_write_heads, controller_dim)
        self.controller = controller_cls(input_dim + num_read_heads * M, controller_dim, **other_controller_params)
        self.device = None

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(controller_dim + num_read_heads * M, output_dim)

    def to(self, *args, **kwargs):
        if args and (isinstance(args[0], torch.device) or ('cuda' in args[0]) or ('cpu' in args[0])):
            self.device = args[0]
        elif kwargs and 'device' in kwargs:
            self.device = kwargs['device']

        return super(NTMCell, self).to(*args, **kwargs)

    def create_new_state(self, batch_size):
        initial_reads = torch.zeros((batch_size, len(self.read_heads), self.M))
        controller_state = self.controller.create_new_state(batch_size)
        read_head_states = self.read_heads.create_new_state(batch_size)
        write_head_states = self.write_heads.create_new_state(batch_size)
        ntm_memory = NTMMemory(self.N, self.M)
        ntm_memory.reset(batch_size)

        return tensor_to_device((initial_reads, controller_state, read_head_states, write_head_states, ntm_memory),
                                device=self.device)

    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x input_dim)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_read_head_states, prev_write_head_states, prev_ntm_memory = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x, prev_reads.view(-1, len(self.read_heads) * self.M)], dim=-1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # VECTORIZE !!!!!!!!!!!!!!
        reads, read_head_states = self.read_heads(controller_outp, prev_read_head_states, prev_ntm_memory)
        (w, e, a), write_head_states = self.write_heads(controller_outp, prev_write_head_states, prev_ntm_memory)
        # erases: [B, n, N, M]
        erases = torch.matmul(w.unsqueeze(-1), e.unsqueeze(-2))
        erase_multiplier = torch.ones(prev_ntm_memory.size())
        for i in range(erases.size(1)):
            erase = erases[:, i]
            erase_multiplier *= (1 - erase)
        prev_ntm_memory.mul(erase_multiplier)
        # addition: [B, n, N, M]
        addition = torch.matmul(w.unsqueeze(-1), a.unsqueeze(-2))
        prev_ntm_memory.add(addition.sum(dim=1))

        # Generate Output
        inp2 = torch.cat([controller_outp, reads.view(-1, len(self.read_heads) * self.M)], dim=-1)
        o = self.fc(inp2)

        # Pack the current state
        state = (reads, controller_state, read_head_states, write_head_states, prev_ntm_memory)

        return o, state
