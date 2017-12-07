import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, img_feat_size):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = nn.Linear(img_feat_size, nhid)

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.Wq_1 = nn.Linear(self.nhid, self.nhid)
        self.Wh_1 = nn.Linear(self.nhid, self.nhid)
        self.Wa_1 = nn.Linear(self.nhid, 1)

        self.Wq_2 = nn.Linear(self.nhid, self.nhid)
        self.Wh_2 = nn.Linear(self.nhid, self.nhid)
        self.Wi_2 = nn.Linear(self.nhid, self.nhid)
        self.Wa_2 = nn.Linear(self.nhid, 1)

        self.fc1 = nn.Linear(self.nhid*3, self.ninp)

    def forward(self, ques_emb, his_emb, img_raw, ques_hidden, his_hidden, rnd):

        img_emb = F.tanh(self.img_embed(img_raw))

        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = ques_feat[-1]

        his_feat, his_hidden = self.his_rnn(his_emb, his_hidden)
        his_feat = his_feat[-1]

        ques_emb_1 = self.Wq_1(ques_feat).view(-1, 1, self.nhid)
        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)

        atten_emb_1 = F.tanh(his_emb_1 + ques_emb_1.expand_as(his_emb_1))
        his_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb_1, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, rnd))

        his_attn_feat = torch.bmm(his_atten_weight.view(-1, 1, rnd),
                                        his_feat.view(-1, rnd, self.nhid))

        his_attn_feat = his_attn_feat.view(-1, self.nhid)
        ques_emb_2 = self.Wq_2(ques_feat).view(-1, 1, self.nhid)
        his_emb_2 = self.Wh_2(his_attn_feat).view(-1, 1, self.nhid)
        img_emb_2 = self.Wi_2(img_emb).view(-1, 49, self.nhid)

        atten_emb_2 = F.tanh(img_emb_2 + ques_emb_2.expand_as(img_emb_2) + \
                                    his_emb_2.expand_as(img_emb_2))

        img_atten_weight = F.softmax(self.Wa_2(F.dropout(atten_emb_2, self.d, training=self.training
                                                ).view(-1, self.nhid)).view(-1, 49))

        img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, 49),
                                        img_emb.view(-1, 49, self.nhid))

        concat_feat = torch.cat((ques_feat, his_attn_feat.view(-1, self.nhid), \
                                 img_attn_feat.view(-1, self.nhid)),1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
