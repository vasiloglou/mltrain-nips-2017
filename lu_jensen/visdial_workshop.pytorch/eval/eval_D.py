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

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
import datetime
import h5py
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_h5', default='vdl_img_vgg.h5', help='')
parser.add_argument('--input_ques_h5', default='visdial_data.h5', help='visdial_data.h5')
parser.add_argument('--input_json', default='visdial_params.json', help='visdial_params.json')

parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')

opt = parser.parse_args()

opt.manualSeed = random.randint(1, 10000) # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

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


dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
####################################################################################
# Build the Model
####################################################################################

n_words = dataset_val.vocab_size
ques_length = dataset_val.ques_length
ans_length = dataset_val.ans_length + 1
his_length = ques_length+dataset_val.ans_length
itow = dataset_val.itow
img_feat_size = 512

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)

netW = model._netW(n_words, opt.ninp, opt.dropout)
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, n_words, opt.dropout)
critD = model.nPairLoss(opt.nhid, 2)

if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    print('Loading model Success!')

if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(), netD.cuda()
    critD.cuda()

n_neg = 100
####################################################################################
# Some Functions
####################################################################################

def eval():
    netW.eval()
    netE.eval()
    netD.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    i = 0
    display_count = 0
    average_loss = 0
    rank_all_tmp = []
    result_all = []
    img_atten = torch.FloatTensor(100 * 30, 10, 7, 7)
    while i < len(dataloader_val):#len(1000):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, 512)
        img_input.data.resize_(image.size()).copy_(image)
        save_tmp = [[] for j in range(batch_size)]

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            gt_index.data.resize_(gt_id.size()).copy_(gt_id)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)

            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                                    ques_hidden, hist_hidden, rnd+1)

            #img_atten[i*batch_size:(i+1)*batch_size, rnd, :] = img_atten_weight.data.view(batch_size, 7, 7)

            opt_ans_emb = netW(opt_ans_input, format = 'index')
            opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
            opt_feat = netD(opt_ans_emb, opt_ans_input, opt_hidden, n_words)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = score.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(score, 1, descending=True)

            count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())

        i += 1

        result_all += save_tmp

        if i % 50 == 0:
            R1 = np.sum(np.array(rank_all_tmp)==1) / float(len(rank_all_tmp))
            R5 =  np.sum(np.array(rank_all_tmp)<=5) / float(len(rank_all_tmp))
            R10 = np.sum(np.array(rank_all_tmp)<=10) / float(len(rank_all_tmp))
            ave = np.sum(np.array(rank_all_tmp)) / float(len(rank_all_tmp))
            mrr = np.sum(1/(np.array(rank_all_tmp, dtype='float'))) / float(len(rank_all_tmp))
            print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(1, len(dataloader_val), mrr, R1, R5, R10, ave))

    return img_atten

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)
fake_ans_input = torch.FloatTensor(ques_length, opt.batchSize, n_words)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

# answer index location.
opt_index = torch.LongTensor( opt.batchSize)
fake_index = torch.LongTensor(opt.batchSize)

batch_sample_idx = torch.LongTensor(opt.batchSize)
# answer len
fake_len = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    fake_ans_input, sample_ans_input = fake_ans_input.cuda(), sample_ans_input.cuda()
    opt_index, fake_index =  opt_index.cuda(), fake_index.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    gt_index = gt_index.cuda()


ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

opt_ans_input = Variable(opt_ans_input)
fake_ans_input = Variable(fake_ans_input)
sample_ans_input = Variable(sample_ans_input)

opt_index = Variable(opt_index)
fake_index = Variable(fake_index)

fake_len = Variable(fake_len)
noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
gt_index = Variable(gt_index)

atten = eval()

R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
ave = np.sum(np.array(rank_all)) / float(len(rank_all))
mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(1, len(dataloader_val), mrr, R1, R5, R10, ave))
