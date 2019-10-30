import torch
import torch.nn as nn

from ntm import NTMCell


def main():
    if False:
        rnn = nn.LSTMCell(5, 10)
        # (seq_len, batch_size, embedding_size)
        inp = torch.ones((3, 4, 5), dtype=torch.float)
        hx, cx = torch.zeros((4, 10)), torch.zeros((4, 10))
        out = []
        for i in inp:
            hx, cx = rnn(i, (hx, cx))
            out.append(hx)
        print(out)

    if True:
        rnn = NTMCell(5, 10).to('cpu')
        # (seq_len, batch_size, embedding_size)
        inp = torch.ones((3, 4, 5), dtype=torch.float)
        prev_state = rnn.create_new_state(4)
        out = []
        for i in inp:
            o, prev_state = rnn(i, prev_state)
            out.append(o)
        print(out)


if __name__ == '__main__':
    main()
