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
from misc.netG import _netG
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='data/vdl_img_vgg.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='data/visdial_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='data/visdial_params.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output images and model checkpoints')
parser.add_argument('--encoder', default='G_QIH_VGG', help='what encoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=0, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--save_iter', type=int, default=1, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')

parser.add_argument('--log_interval', type=int, default=50, help='how many iterations show the log info')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    opt = checkpoint['opt']
    opt.start_epoch = checkpoint['epoch']
    opt.model_path = model_path
    opt.batchSize = 128
    opt.niter = 100
else:
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' %(t.day, t.month, t.hour)
    save_path = os.path.join(opt.outf, opt.encoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

####################################################################################
# Data Loader
####################################################################################
dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'train')

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=5,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################

vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ques_length + dataset.ans_length
itow = dataset.itow
img_feat_size = opt.conv_feat_size

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)

netW = model._netW(vocab_size, opt.ninp, opt.dropout)
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)
critG = model.LMCriterion()
sampler = model.gumbel_sampler()

if opt.cuda:
    netW.cuda()
    netE.cuda()
    netG.cuda()
    critG.cuda()
    sampler.cuda()

if opt.model_path != '':
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netG.load_state_dict(checkpoint['netG'])

# training function
def train(epoch):
    netW.train()
    netE.train()
    netG.train()

    lr = adjust_learning_rate(optimizer, epoch, opt.lr)
    data_iter = iter(dataloader)

    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)
    average_loss = 0
    count = 0
    i = 0

    total_loss = 0
    while i < len(dataloader):
        data = data_iter.next()
        image, history, question, answer, answerT, answerLen, answerIdx, \
        questionL, negAnswer, negAnswerLen, negAnswerIdx = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            ans, tans = answer[:,rnd,:].t(), answerT[:,rnd,:].t()

            his_input.data.resize_(his.size()).copy_(his)
            ques_input.data.resize_(ques.size()).copy_(ques)
            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)

            _, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)

            ans_emb = netW(ans_input)
            logprob, ques_hidden = netG(ans_emb, ques_hidden)
            loss = critG(logprob, ans_target.view(-1, 1))

            loss = loss / torch.sum(ans_target.data.gt(0))
            average_loss += loss.data[0]
            total_loss += loss.data[0]
            # do backward.
            netW.zero_grad()
            netE.zero_grad()
            netG.zero_grad()
            loss.backward()

            optimizer.step()
            count += 1
        i += 1
        if i % opt.log_interval == 0:
            average_loss /= count
            print("step {} / {} (epoch {}), g_loss {:.3f}, lr = {:.6f}"\
                .format(i, len(dataloader), epoch, average_loss, lr))
            average_loss = 0
            count = 0

    return total_loss / (10 * i), lr


def val():
    netE.eval()
    netW.eval()
    netG.eval()

    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    i = 0
    average_loss = 0
    rank_all_tmp = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                    opt_answerT, answer_ids, answerLen, opt_answerLen, img_id = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques, tans = question[:,rnd,:].t(), opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            ans = opt_answer[:,rnd,:,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            his_input.data.resize_(his.size()).copy_(his)
            ques_input.data.resize_(ques.size()).copy_(ques)
            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)

            gt_index.data.resize_(gt_id.size()).copy_(gt_id)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                ques_hidden, hist_hidden, rnd+1)

            _, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)

            hidden_replicated = []
            for hid in ques_hidden:
                hidden_replicated.append(hid.view(opt.nlayers, batch_size, 1, \
                    opt.nhid).expand(opt.nlayers, batch_size, 100, opt.nhid).clone().view(opt.nlayers, -1, opt.nhid))
            hidden_replicated = tuple(hidden_replicated)

            ans_emb = netW(ans_input, format = 'index')

            output, _ = netG(ans_emb, hidden_replicated)
            logprob = - output
            logprob_select = torch.gather(logprob, 1, ans_target.view(-1,1))

            mask = ans_target.data.eq(0)  # generate the mask
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
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())

        i += 1

    return rank_all_tmp, average_loss



####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize, 49, 512)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

ans_input = torch.LongTensor(ans_length, opt.batchSize)
ans_target = torch.LongTensor(ans_length, opt.batchSize)

ans_sample = torch.LongTensor(1, opt.batchSize)
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)

if opt.cuda:
    img_input, his_input = img_input.cuda(), his_input.cuda()
    ques_input, ans_input = ques_input.cuda(), ans_input.cuda()
    ans_target, ans_sample = ans_target.cuda(), ans_sample.cuda()
    noise_input = noise_input.cuda()
    gt_index = gt_index.cuda()

ques_input = Variable(ques_input)
ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
ans_sample = Variable(ans_sample)
noise_input = Variable(noise_input)
img_input = Variable(img_input)
his_input = Variable(his_input)
gt_index = Variable(gt_index)

optimizer = optim.Adam([{'params': netW.parameters()},
                        {'params': netG.parameters()},
                        {'params': netE.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

history = []

for epoch in range(opt.start_epoch+1, opt.niter):
    t = time.time()
    train_loss, lr = train(epoch)

    print ('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time()-t))

    print('Evaluating ... ')
    rank_all, val_loss = val()
    R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
    R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
    print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))

    train_his = {'loss': train_loss}
    val_his = {'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}

    history.append({'epoch':epoch, 'train': train_his, 'val': val_his})

    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW': netW.state_dict(),
                    'netG': netG.state_dict(),
                    'netE': netE.state_dict()},
                    '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' %(save_path), 'w'))
