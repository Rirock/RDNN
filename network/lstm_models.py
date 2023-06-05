import torch
from torch import nn
import math

class LSTM_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, device='cpu'):
        super(LSTM_REG, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers).to(device) # rnn
        self.reg = nn.Linear(hidden_size, output_size).to(device) # 回归

    def forward(self, x):
        x = x.float()
        x, _ = self.rnn(x) # (seq, batch, hidden)
        # _, (x, _) = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class GRU_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, device='cpu'):
        super(GRU_REG, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers).to(device) # rnn
        self.reg = nn.Linear(hidden_size, output_size).to(device) # 回归

    def forward(self, x):
        x = x.float()
        x, _ = self.rnn(x) # (seq, batch, hidden)
        # _, (x, _) = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, device='cpu'):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size * num_layers
        self.linear1 = nn.Linear(input_size, hidden_size * num_layers).to(device) # 回归
        self.linear2 = nn.Linear(hidden_size * num_layers, output_size).to(device) # 回归

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)