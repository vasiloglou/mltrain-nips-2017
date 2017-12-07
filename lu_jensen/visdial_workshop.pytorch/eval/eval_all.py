from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
import datetime
from misc.netG import _netG

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_h5', default='vdl_img_vgg.h5', help='')
parser.add_argument('--input_ques_h5', default='visdial_data.h5', help='visdial_data.h5')
parser.add_argument('--input_json', default='visdial_params.json', help='visdial_params.json')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')

opt = parser.parse_args()

print(opt)
opt.manualSeed = 111 #random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################################################################
# Data Loader
####################################################################################

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    data_dir = opt.data_dir
    input_img_h5 = opt.input_img_h5
    input_ques_h5 = opt.input_ques_h5
    input_json = opt.input_json
    opt = checkpoint['opt']
    opt.start_epoch = checkpoint['epoch']
    opt.batchSize = 5
    opt.data_dir = data_dir
    opt.model_path = model_path

input_img_h5 = os.path.join(opt.data_dir, opt.input_img_h5)
input_ques_h5 = os.path.join(opt.data_dir, opt.input_ques_h5)
input_json = os.path.join(opt.data_dir, opt.input_json)

dataset_val = dl.validate(input_img_h5=input_img_h5, input_ques_h5=input_ques_h5,
                input_json=input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=30,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
vocab_size = dataset_val.vocab_size
ques_length = dataset_val.ques_length
ans_length = dataset_val.ans_length + 1
his_length = dataset_val.ans_length + dataset_val.ques_length
itow = dataset_val.itow
img_feat_size = 512


print('init Generative model...')
netE_g = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW_g = model._netW(vocab_size, opt.ninp, opt.dropout)
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)
sampler = model.gumbel_sampler()
critG = model.G_loss(opt.ninp)
critLM = model.LMCriterion()

if  opt.model_path_G != '':
    print('Loading Generative model...')
    netW_g.load_state_dict(checkpoint_G['netW_g'])
    netE_g.load_state_dict(checkpoint_G['netE_g'])
    netG.load_state_dict(checkpoint_G['netG'])


if opt.cuda: # ship to cuda, if has GPU
    netW_g.cuda()
    netE_g.cuda()
    netG.cuda()
    critG.cuda()
    sampler.cuda()
    critLM.cuda()

####################################################################################
# training model
####################################################################################

def val():
    netE_g.eval()
    netW_g.eval()
    netG.eval()

    n_neg = 100
    ques_hidden1 = netE_g.init_hidden(opt.batchSize)
    hist_hidden2 = netE_g.init_hidden(opt.batchSize)
    data_iter_val = iter(dataloader_val)

    count = 0
    i = 0
    rank_G = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                    opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, 512)

        img_input.data.resize_(image.size()).copy_(image)
        for rnd in range(10):

            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answer[:,rnd,:,:].clone().view(-1, ans_length).t()
            opt_tans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]
            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
            opt_ans_target.data.resize_(opt_tans.size()).copy_(opt_tans)
            gt_index.data.resize_(gt_id.size()).copy_(gt_id)

            ques_emb_g = netW_g(ques_input, format = 'index')
            his_emb_g = netW_g(his_input, format = 'index')

            ques_emb_d = netW_d(ques_input, format = 'index')
            his_emb_d = netW_d(his_input, format = 'index')

            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            ques_hidden2 = repackage_hidden(ques_hidden2, batch_size)

            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))
            hist_hidden2 = repackage_hidden(hist_hidden2, his_emb_d.size(1))

            featG, ques_hidden1 = netE_g(ques_emb_g, his_emb_g, img_input, \
                                                ques_hidden1, hist_hidden1, rnd+1)

            featD, _ = netE_d(ques_emb_d, his_emb_d, img_input, \
                                                ques_hidden2, hist_hidden2, rnd+1)
            #featD = l2_norm(featD)
            _, ques_hidden1 = netG(featG.view(1,-1,opt.ninp), ques_hidden1)

            hidden_replicated = []
            for hid in ques_hidden1:
                hidden_replicated.append(hid.view(opt.nlayers, batch_size, 1, \
                    opt.nhid).expand(opt.nlayers, batch_size, 100, opt.nhid).clone().view(opt.nlayers, -1, opt.nhid))
            hidden_replicated = tuple(hidden_replicated)

            ans_emb = netW_g(opt_ans_input, format = 'index')

            output, _ = netG(ans_emb, hidden_replicated)
            logprob = - output
            logprob_select = torch.gather(logprob, 1, opt_ans_target.view(-1,1))

            mask = opt_ans_target.data.eq(0)  # generate the mask
            if isinstance(logprob, Variable):
                mask = Variable(mask, volatile=logprob.volatile)

            logprob_select.masked_fill_(mask.view_as(logprob_select), 0)

            prob = logprob_select.view(ans_length, -1, 100).sum(0).view(-1,100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = prob.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(prob, 1)

            count = sort_score.lt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_G += list(rank.view(-1).data.cpu().numpy())

            opt_ans_emb = netW_d(opt_ans_target, format = 'index')
            opt_hidden = repackage_hidden(opt_hidden, opt_ans_target.size(1))
            opt_feat = netD(opt_ans_emb, opt_ans_target, opt_hidden, vocab_size)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            #ans_emb = ans_emb.view(ans_length, -1, 100, opt.nhid)
            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            gt_score = score.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(score, 1, descending=True)
            count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_D += list(rank.view(-1).data.cpu().numpy())
        i += 1

        if i % 50 == 0:
            R1 = np.sum(np.array(rank_G)==1) / float(len(rank_G))
            R5 =  np.sum(np.array(rank_G)<=5) / float(len(rank_G))
            R10 = np.sum(np.array(rank_G)<=10) / float(len(rank_G))
            ave = np.sum(np.array(rank_G)) / float(len(rank_G))
            mrr = np.sum(1/(np.array(rank_G, dtype='float'))) / float(len(rank_G))
            print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(1, len(dataloader_val), mrr, R1, R5, R10, ave))



    return rank_G, rank_D

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
ans_input = torch.LongTensor(ans_length, opt.batchSize)
ans_target = torch.LongTensor(ans_length, opt.batchSize)
wrong_ans_input = torch.LongTensor(ans_length, opt.batchSize)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

fake_len = torch.LongTensor(opt.batchSize)
fake_diff_mask = torch.ByteTensor(opt.batchSize)
fake_mask = torch.ByteTensor(opt.batchSize)
# answer len
batch_sample_idx = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)

# for evaluation:
opt_ans_input = torch.LongTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)
opt_ans_target = torch.LongTensor(opt.batchSize)

if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    ans_input, ans_target = ans_input.cuda(), ans_target.cuda()
    wrong_ans_input = wrong_ans_input.cuda()
    sample_ans_input = sample_ans_input.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    fake_diff_mask = fake_diff_mask.cuda()
    fake_mask = fake_mask.cuda()

    opt_ans_input = opt_ans_input.cuda()
    gt_index = gt_index.cuda()
    opt_ans_target = opt_ans_target.cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)
sample_ans_input = Variable(sample_ans_input)

noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
fake_diff_mask = Variable(fake_diff_mask)
fake_mask = Variable(fake_mask)

opt_ans_input = Variable(opt_ans_input)
opt_ans_target = Variable(opt_ans_target)
gt_index = Variable(gt_index)


optimizerD = optim.Adam([{'params': netW_d.parameters()},
                        {'params': netE_d.parameters()},
                        {'params': netD.parameters()}], lr=opt.D_lr, betas=(opt.beta1, 0.999))

optimizerG = optim.Adam([{'params': netW_g.parameters()},
                        {'params': netE_g.parameters()},
                        {'params': netG.parameters()}], lr=opt.G_lr, betas=(opt.beta1, 0.999))

optimizerLM = optim.Adam([{'params': netW_g.parameters()},
                        {'params': netE_g.parameters()},
                        {'params': netG.parameters()}], lr=opt.LM_lr, betas=(opt.beta1, 0.999))
history = []
train_his = {}

epoch = 0
print('Evaluating ... ')
rank_G, rank_D = val()
R1 = np.sum(np.array(rank_G)==1) / float(len(rank_G))
R5 =  np.sum(np.array(rank_G)<=5) / float(len(rank_G))
R10 = np.sum(np.array(rank_G)<=10) / float(len(rank_G))
ave = np.sum(np.array(rank_G)) / float(len(rank_G))
mrr = np.sum(1/(np.array(rank_G, dtype='float'))) / float(len(rank_G))
print ('Generator: %d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))

R1 = np.sum(np.array(rank_D)==1) / float(len(rank_D))
R5 =  np.sum(np.array(rank_D)<=5) / float(len(rank_D))
R10 = np.sum(np.array(rank_D)<=10) / float(len(rank_D))
ave = np.sum(np.array(rank_D)) / float(len(rank_D))
mrr = np.sum(1/(np.array(rank_D, dtype='float'))) / float(len(rank_D))
print ('Discriminator: %d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
