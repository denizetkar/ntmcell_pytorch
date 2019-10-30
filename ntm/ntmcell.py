import torch
import torch.nn as nn

from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory
from .controller import GRUController
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
        self.read_heads = nn.ModuleList([NTMReadHead(N, M, controller_dim) for _ in range(num_read_heads)])
        self.write_heads = nn.ModuleList([NTMWriteHead(N, M, controller_dim) for _ in range(num_write_heads)])
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
        initial_reads = [torch.zeros((batch_size, self.M)) for _ in range(len(self.read_heads))]
        controller_state = self.controller.create_new_state(batch_size)
        read_head_states = [head.create_new_state(batch_size) for head in self.read_heads]
        write_head_states = [head.create_new_state(batch_size) for head in self.write_heads]
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
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads, read_head_states = [], []
        for head, prev_head_state in zip(self.read_heads, prev_read_head_states):
            r, head_state = head(controller_outp, prev_head_state, prev_ntm_memory)
            reads.append(r)
            read_head_states.append(head_state)
        w_e_a_list, write_head_states = [], []
        for head, prev_head_state in zip(self.write_heads, prev_write_head_states):
            w_e_a, head_state = head(controller_outp, prev_head_state, prev_ntm_memory)
            w_e_a_list.append(w_e_a)
            write_head_states.append(head_state)
        # Perform the erase operations
        for w, e, a in w_e_a_list:
            prev_ntm_memory.write_erase(w, e)
        # Perform the addition operations
        for w, e, a in w_e_a_list:
            prev_ntm_memory.write_add(w, a)

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = self.fc(inp2)

        # Pack the current state
        state = (reads, controller_state, read_head_states, write_head_states, prev_ntm_memory)

        return o, state
