import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output

class QuestionEmbedding1(nn.Module):
    def __init__(self, in_dim):
        """Module for question CNN embedding
        """
        super(QuestionEmbedding1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, in_dim), stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, in_dim), stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(3, in_dim), stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.linear = nn.Linear(1280, 1024)

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        x_input = x.unsqueeze(1)  # [batch, 1, sequence, in_dim]
        tanh1 = self.conv1(x_input) # [batch, 256, sequence, 1]
        tanh1 = torch.tanh(tanh1)
        tanh1, indice1 = torch.max(tanh1, dim=2)  # [batch, 256, 1]

        tanh2 = self.conv2(x_input) # [batch, 256, sequence - 1, 1]
        tanh2 = torch.tanh(tanh2)
        tanh2, indice2 = torch.max(tanh2, dim=2) # [batch, 256, 1]

        tanh3 = self.conv3(x_input) # [batch, 512, sequence - 2, 1]
        tanh3 = torch.tanh(tanh3)
        tanh3, indice3 = torch.max(tanh3, dim=2)  # [batch, 512, 1]

        question_embedding = torch.cat((tanh1, tanh2, tanh3), dim=1).squeeze(2) # [batch, 1024]


        # linear_layer = nn.Linear(in_features=length, out_features=1024)

        # question_embedding = self.linear(question_embedding)

        return question_embedding


