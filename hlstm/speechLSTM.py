from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from settings import *

class encoderLayer(nn.Module):
    def __init__(self, n_split, input_dim, hidden_dim):
        super(encoderLayer, self).__init__()
        # Hyper parameters
        self.n_split = n_split
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Model Layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x = [batch size, time steps, input dim]
        input_size = x.size()
        self.batch_size = input_size[0]
        assert(input_size[2] == self.input_dim)
        time_steps = input_size[1]

        x = x.view(self.batch_size*self.n_split, -1, self.input_dim)
        # x = [batch_size * n_split, time steps / n_split, input dim]

        out, (hidden, cell) = self.lstm(x)
        # out = [batch_size * n_split, tiem steps / n_split, hid dim * num directions]
        # hidden = [num layers * num directions, batch size * n_split, hid. dim]
        # cell = [num layers * num directions, batch size * n_split, hid. dim]

        hidden = hidden.contiguous().view(self.batch_size, self.n_split, -1)
        # cell = cell.contiguous().view(self.batch_size, self.n_split, -1)
        # hiddens/cells: [batch_size, n_split, hidden_dim]

        return hidden


class decoderLayer(nn.Module):
    def __init__(self, n_split, input_dim, hidden_dim, output_dim, device):
        super(decoderLayer, self).__init__()
        # Hyper parameters
        self.n_split = n_split
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.act = nn.Tanh()

        # Model Layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim) # fc takes in inputs of [batch_size, hidden dim]

    def initCell(self):
        return torch.zeros(1, self.batch_size * self.n_split, self.hidden_dim, device=self.device)

    def forward(self, x, h0, teacher_forcing, output_time_steps):
        # x = [batch size, time steps, input dim]
        # h0/c0 = [1, batch size * n_split, hidden dim]
        input_size = x.size()
        self.batch_size = input_size[0]
        assert(input_size[2] == self.input_dim)
        time_steps = input_size[1]

        x = x.view(self.batch_size*self.n_split, -1, self.input_dim)
        # x = [batch_size * n_split, time steps / n_split, input dim]
        h0 = h0.contiguous().view(1, self.batch_size * self.n_split, self.hidden_dim)
        c0 = self.initCell()
    
        if (teacher_forcing):
            out, (hidden, cell) = self.lstm(x, (h0, c0))
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
            # out = [batch_size * n_split, time steps / n_split, hid dim]
            # hidden/cell = [1, batch size * n_split, hid. dim]
        else:
            out = []
            h, c = h0, c0
            for t in range(output_time_steps):
                #print("Time = %d"%t)
                _, (h, c) = self.lstm(x, (h, c))
                #print("hidden size", h.size())
                x = h.view(-1, 1, self.hidden_dim)
                x = self.fc(x)
                x = self.act(x)
                out.append(x)
            out = torch.cat(out, 1)
            # out = [batch_size * n_split, output_time_steps, hidden dim]
        
        out = out.view(self.batch_size, -1, self.output_dim)
        # out = [batch_size, time steps, hid dim * num directions]

        return out

def test_encoder():
    n_split = 4
    input_dim = 100
    hidden_dim = 128
    encoder = encoderLayer(n_split, input_dim, hidden_dim)
    encoder.to(DEVICE)

if __name__ == '__main__':
    test_encoder()

################# Testing block ######################

# USE_CUDA = torch.cuda.is_available() #torch.cuda.is_available()
# print("cuda available: %d"%USE_CUDA)


# def general_infer(model, inputs):
#     # inputs = [batch_size, sent len, input_dim]
#     outputs = model(inputs)
#     # outputs = [batch_size, sent len, hidden_size * n_directions]

#     outputs = torch.mean(outputs, 2)
#     # outputs = [batch_size, sent len]

#     outputs = torch.mean(outputs, 1)
#     # outputs = [batch_size]
#     return outputs

# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(F.sigmoid(preds))
#     correct = (rounded_preds == y).float()
#     acc = correct.sum()/len(correct)
#     return acc

# INPUT_DIM = 50
# HIDDEN_DIM = 256
# BATCH_SIZE = 64
# N_LAYERS = 2
# BIDIRECTIONAL = 1
# DROPOUT = 0.5
# SENT_LEN = 20
# n_split = 2

# model = speechLSTMLayer(n_split, INPUT_DIM, HIDDEN_DIM, BATCH_SIZE,
#     N_LAYERS, BIDIRECTIONAL, DROPOUT)

# optimizer = optim.Adam(model.parameters())

# criterion = nn.BCEWithLogitsLoss()

# device = torch.device('cuda' if USE_CUDA else 'cpu')

# model = model.to(device)
# criterion = criterion.to(device)


# SENT_LEN = 8
# MAX_VAL = 1000
# inputs = torch.randint(0, MAX_VAL, (BATCH_SIZE, SENT_LEN, INPUT_DIM), dtype = torch.float)
# labels = torch.bernoulli(torch.empty(BATCH_SIZE).uniform_(0, 1))

# inputs = inputs.to(device)
# labels = labels.to(device)

# optimizer.zero_grad()
# predictions = general_infer(model, inputs)
# loss = criterion(predictions, labels)
# acc = binary_accuracy(predictions, labels)
# loss.backward()
# optimizer.step()
# print("loss: %.3f, acc: %.2f%%" %(loss, acc*100))

################# Testing block ######################
