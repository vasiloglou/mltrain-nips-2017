import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers,dropout):
        super(_netE, self).__init__()
        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.fc1 = nn.Linear(self.nhid, self.ninp)


    def forward(self, emb, hidden):

        ques_feat, hidden = self.ques_rnn(emb, hidden)
        concat_feat = F.dropout(ques_feat[-1], self.d, training=self.training)
        encoder_feat = F.tanh(self.fc1(concat_feat))

        return encoder_feat, hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
