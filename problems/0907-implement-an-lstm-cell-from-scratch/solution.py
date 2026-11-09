import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))

        k = 1.0 / math.sqrt(hidden_size)
        for p in (self.W_ih, self.W_hh, self.b_ih, self.b_hh):
            nn.init.uniform_(p, -k, k)

    def forward(self, x, state):
        h_prev, c_prev = state
        g = F.linear(x, self.W_ih, self.b_ih) + F.linear(h_prev, self.W_hh, self.b_hh)
        i, f, c_tilde, o = g.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_tilde = torch.tanh(c_tilde)
        c_new = f * c_prev + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return h_new, c_new