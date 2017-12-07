
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
'''
class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = nn.Linear(4096, 512)

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.fc1 = nn.Linear(self.nhid*2, self.nhid)
        self.fc2 = nn.Linear(self.nhid, self.ninp)

    def forward(self, ques_emb, img_raw, ques_hidden, rnd):

        img_emb = F.dropout(F.tanh(self.img_embed(img_raw)), self.d, training=self.training)

        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = F.dropout(ques_feat[-1], self.d, training=self.training)
        concat_feat = torch.cat((ques_feat, img_emb),1)
        encoder_feat = F.dropout(F.tanh(self.fc1(concat_feat)),  self.d, training=self.training)
        encoder_feat =  F.dropout(self.fc2(encoder_feat), 0.3, training=self.training)

        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


'''

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = nn.Linear(512, 512)

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.Wq_2 = nn.Linear(self.nhid, self.nhid)
        self.Wi_2 = nn.Linear(self.nhid, self.nhid)
        self.Wa_2 = nn.Linear(self.nhid, 1)

        self.fc1 = nn.Linear(self.nhid*2, self.ninp)
        #self.fc2 = nn.Linear(self.nhid*2, self.ninp)

    def forward(self, ques_emb, img_raw, ques_hidden, rnd):

        img_emb = F.tanh(self.img_embed(img_raw))

        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        #ques_feat = F.dropout(ques_feat[-1], self.d, training=self.training)
        ques_feat = ques_feat[-1]

        ques_emb_2 = self.Wq_2(ques_feat).view(-1, 1, self.nhid)
        img_emb_2 = self.Wi_2(img_emb).view(-1, 49, self.nhid)

        atten_emb_2 = F.tanh(img_emb_2 + ques_emb_2.expand_as(img_emb_2))

        img_atten_weight = F.softmax(self.Wa_2(F.dropout(atten_emb_2, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, 49))

        img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, 49),
                                        img_emb.view(-1, 49, self.nhid))

        concat_feat = F.dropout(torch.cat((img_attn_feat.view(-1, self.nhid), ques_feat), 1), self.d, training=self.training)
        encoder_feat = F.tanh(self.fc1(concat_feat))


        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

'''

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = nn.Linear(4096, 512)

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        #self.Wq_2 = nn.Linear(self.nhid, self.nhid)
        #self.Wi_2 = nn.Linear(self.nhid, self.nhid)
        #self.Wa_2 = nn.Linear(self.nhid, 1)

        #self.fc1 = nn.Linear(self.nhid, self.nhid)
        self.fc2 = nn.Linear(self.nhid*2, self.ninp)

    def forward(self, ques_emb, img_raw, ques_hidden, rnd):

        img_emb = F.dropout(F.tanh(self.img_embed(img_raw)), self.d, training=self.training)

        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = F.dropout(ques_feat[-1], self.d, training=self.training)
        #ques_feat = ques_feat[-1]

        #ques_emb_2 = self.Wq_2(ques_feat).view(-1, 1, self.nhid)
        #img_emb_2 = self.Wi_2(img_emb).view(-1, 49, self.nhid)

        #atten_emb_2 = F.tanh(img_emb_2 + ques_emb_2.expand_as(img_emb_2))

        #img_atten_weight = F.softmax(self.Wa_2(F.dropout(atten_emb_2, self.d, training=self.training
        #                                        ).view(-1, self.nhid)).view(-1, 49))

        #img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, 49),
        #                                img_emb.view(-1, 49, self.nhid))

        #encoder_feat = F.dropout(F.tanh(self.fc1(img_attn_feat.view(-1, self.nhid))),  self.d, training=self.training)
        concat_feat = torch.cat((img_emb, ques_feat), 1)
        encoder_feat = F.dropout(F.tanh(self.fc2(concat_feat)), self.d, training=self.training)

        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
'''
