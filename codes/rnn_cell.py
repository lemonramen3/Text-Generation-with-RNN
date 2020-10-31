import torch
from torch import nn
import torch.nn.functional as F

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        #return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        # flag indicates whether the position is valid. 1 for valid, 0 for invalid.
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output # stored for next step
        return output, new_state

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        Z = torch.sigmoid(self.W_z(incoming) + self.U_z(state))
        R = torch.sigmoid(self.W_r(incoming) + self.U_r(state))
        H_head = (self.W_c(incoming) + self.U(R * state))
        H = (1 - Z) * state + Z * H_head
        return H, H
        # TODO END

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state (which can be a tuple)
        return (torch.zeros((batch_size, self.hidden_size), device=device),
                torch.zeros((batch_size, self.hidden_size), device=device))
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state

        I = torch.sigmoid(self.W_i(incoming) + self.U_i(state[0]))
        O = torch.sigmoid(self.W_o(incoming) + self.U_o(state[0]))
        F = torch.sigmoid(self.W_f(incoming) + self.U_f(state[0]))
        C_head = (self.W_c(incoming) + self.U_c(state[0]))
        C = F * state[1] + I * C_head
        H = O * torch.tanh(C)
        return H, (H, C)
        # TODO END