import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


class DNM_Linear(nn.Module):    # DNM
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = 5 * torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = torch.sigmoid(x)

        # Dendritic
        x = torch.prod(x, 3) #prod 

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = 5 * (x - 0.5)

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class DNM_Linear_M(nn.Module):    # MDNN
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear_M, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = torch.sigmoid(x)

        # Dendritic
        x = torch.prod(x, 3) #prod 
        # x = torch.tanh(x)

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = self.params['k'] * (x - self.params['qs'])

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    
class DNM_multiple(nn.Module):   # MDNN * 2
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear(input_size, hidden_size, M)
        self.DNM_Linear2 = DNM_Linear(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


# DNM_LSTM
class Gate(nn.Module):
    def __init__(self, input_size, hidden_dim, M, device='cpu'):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size+hidden_dim, hidden_dim)

    def forward(self, x, h_pre, active_func):
        xh = torch.cat([x, h_pre], 1)
        h_next = active_func(self.linear(xh))
        return h_next

class Gate_DNM(nn.Module):
    def __init__(self, input_size, hidden_dim, M, device='cpu'):
        super(Gate_DNM, self).__init__()
        self.linear = DNM_Linear_M(input_size+hidden_dim, hidden_dim, M, device=device)

    def forward(self, x, h_pre, active_func):
        xh = torch.cat([x, h_pre], 1)
        h_next = active_func(self.linear(xh))
        return h_next

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_dim, M, device='cpu'):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # self.gate = clones(Gate(input_size, hidden_dim, M, device=device), 4)
        self.gate_f = Gate_DNM(input_size, hidden_dim, M, device=device)
        self.gate_i = Gate_DNM(input_size, hidden_dim, M, device=device)
        self.gate_g = Gate_DNM(input_size, hidden_dim, M, device=device)
        self.gate_o = Gate_DNM(input_size, hidden_dim, M, device=device)

    def forward(self, x, h_pre, c_pre):
        """
        :param x: (batch, input_size)
        :param h_pre: (batch, hidden_dim)
        :param c_pre: (batch, hidden_dim)
        :return: h_next(batch, hidden_dim), c_next(batch, hidden_dim)
        """
        f_t = self.gate_f(x, h_pre, torch.sigmoid)
        i_t = self.gate_i(x, h_pre, torch.sigmoid)
        g_t = self.gate_g(x, h_pre, torch.tanh)
        o_t = self.gate_o(x, h_pre, torch.sigmoid)
        c_next = f_t * c_pre + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next
    
class DNM_LSTM_L2(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, M, device='cpu'):
        super(DNM_LSTM_L2, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.device = device
        self.lstm_cell = LSTMCell(input_size, hidden_dim, M, device=device)
        self.lstm_cell2 = LSTMCell(hidden_dim, hidden_dim, M, device=device)

    def forward(self, x):
        """
        :param x: (seq_len, batch,input_size)
        :return:
           output (seq_len, batch, hidden_dim)
           h_n    (1, batch, hidden_dim)
           c_n    (1, batch, hidden_dim)
        """
        seq_len, batch, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim).to(self.device)
        c = torch.zeros(batch, self.hidden_dim).to(self.device)
        h2 = torch.zeros(batch, self.hidden_dim).to(self.device)
        c2 = torch.zeros(batch, self.hidden_dim).to(self.device)
        output = torch.zeros(seq_len, batch, self.hidden_dim).to(self.device)
        for i in range(seq_len):
            inp = x[i, :, :].to(self.device)
            h, c = self.lstm_cell(inp, h, c)
            h2, c2 = self.lstm_cell2(h, h2, c2)
            output[i, :, :] = h2

        h_n = output[-1:, :, :]
        return output, (h_n, c.unsqueeze(0))

class DNM_LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, M, device='cpu'):
        super(DNM_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.device = device
        self.lstm_cell = LSTMCell(input_size, hidden_dim, M, device=device)

    def forward(self, x):
        """
        :param x: (seq_len, batch,input_size)
        :return:
           output (seq_len, batch, hidden_dim)
           h_n    (1, batch, hidden_dim)
           c_n    (1, batch, hidden_dim)
        """
        seq_len, batch, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim).to(self.device)
        c = torch.zeros(batch, self.hidden_dim).to(self.device)
        output = torch.zeros(seq_len, batch, self.hidden_dim).to(self.device)

        for i in range(seq_len):
            inp = x[i, :, :].to(self.device)
            h, c = self.lstm_cell(inp, h, c)
            output[i, :, :] = h

        h_n = output[-1:, :, :]
        return output, (h_n, c.unsqueeze(0))
    
class RDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, M, device='cpu'):
        super(RDNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = DNM_LSTM(input_size, hidden_size, output_size, M, device=device) # rnn
        self.reg = nn.Linear(hidden_size, output_size).to(device) 

    def forward(self, x):
        x = x.float()
        # _, (x, _) = self.rnn(x) # (seq, batch, hidden)
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

