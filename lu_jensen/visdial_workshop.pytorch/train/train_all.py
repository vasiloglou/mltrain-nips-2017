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
parser.add_argument('--encoder', default='scrach', help='what encoder to use.')
parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')
parser.add_argument('--update_D', action='store_true', help='whether train use the GAN loss.')
parser.add_argument('--update_LM', action='store_true', help='whether train use the GAN loss.')

parser.add_argument('--model_path_D', default='save/HCIAE-D-MLE.pth', help='folder to output images and model checkpoints')
parser.add_argument('--model_path_G', default='save/HCIAE-G-MLE.pth', help='folder to output images and model checkpoints')

parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=5)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--eval_iter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--save_iter', type=int, default=1, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

parser.add_argument('--D_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('--G_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('--LM_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.8')

parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--gumble_weight', type=int, default=0.5, help='folder to output images and model checkpoints')

opt = parser.parse_args()

print(opt)
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# create new folder.
t = datetime.datetime.now()
cur_time = '%s-%s-%s' %(t.day, t.month, t.hour)
save_path = os.path.join(opt.outf, opt.encoder + '.' + cur_time)
try:
    os.makedirs(save_path)
except OSError:
    pass

if opt.model_path_D != '' :
    print("=> loading checkpoint '{}'".format(opt.model_path_D))
    checkpoint_D = torch.load(opt.model_path_D)

if opt.model_path_G != '':
    print("=> loading checkpoint '{}'".format(opt.model_path_G))
    checkpoint_G = torch.load(opt.model_path_G)

####################################################################################
# Data Loader
####################################################################################

dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'train')

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'val')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=10,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ans_length + dataset.ques_length
itow = dataset.itow
img_feat_size = 512

print('init Discriminator model...')
netE_d = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW_d = model._netW(vocab_size, opt.ninp, opt.dropout)
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout)
critD =model.nPairLoss(opt.ninp, opt.margin)

if opt.model_path_D != '' :
    print('Loading Discriminator model...')
    netW_d.load_state_dict(checkpoint_D['netW'])
    netE_d.load_state_dict(checkpoint_D['netE'])
    netD.load_state_dict(checkpoint_D['netD'])

print('init Generative model...')
netE_g = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW_g = model._netW(vocab_size, opt.ninp, opt.dropout)
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)
sampler = model.gumbel_sampler()
critG = model.G_loss(opt.ninp)
critLM = model.LMCriterion()


if  opt.model_path_G != '':
    print('Loading Generative model...')
    netW_g.load_state_dict(checkpoint_G['netW'])
    netE_g.load_state_dict(checkpoint_G['netE'])
    netG.load_state_dict(checkpoint_G['netG'])


if opt.cuda: # ship to cuda, if has GPU
    netW_d.cuda(), netW_g.cuda()
    netE_d.cuda(), netE_g.cuda()
    netD.cuda(), netG.cuda()
    critD.cuda(), critG.cuda()
    sampler.cuda(), critLM.cuda()

####################################################################################
# training model
####################################################################################
def train(epoch):
    netW_d.train(), netE_d.train(), netE_g.train()
    netD.train(), netG.train(), netW_g.train()

    fake_len = torch.LongTensor(opt.batchSize)
    fake_len = fake_len.cuda()

    n_neg = opt.negative_sample
    ques_hidden1 = netE_d.init_hidden(opt.batchSize)
    ques_hidden2 = netE_g.init_hidden(opt.batchSize)

    hist_hidden1 = netE_d.init_hidden(opt.batchSize)
    hist_hidden2 = netE_g.init_hidden(opt.batchSize)

    real_hidden = netD.init_hidden(opt.batchSize)
    wrong_hidden = netD.init_hidden(opt.batchSize)
    fake_hidden = netD.init_hidden(opt.batchSize)

    data_iter = iter(dataloader)

    err_d = 0
    err_g = 0
    err_lm = 0
    average_loss = 0
    count = 0
    i = 0
    loss_store = []
    while i < len(dataloader):
        t1 = time.time()
        data = data_iter.next()
        image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                    opt_answerT, opt_answerLen, opt_answerIdx = data
        batch_size = question.size(0)
        image = image.view(-1, 512)
        img_input.data.resize_(image.size()).copy_(image)

        err_d_tmp = 0
        err_g_tmp = 0
        err_lm_tmp = 0
        err_d_fake_tmp = 0
        err_g_fake_tmp = 0
        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            ans = answer[:,rnd,:].t()
            tans = answerT[:,rnd,:].t()
            wrong_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()

            real_len = answerLen[:,rnd].long()
            wrong_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)
            wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)

            batch_sample_idx.data.resize_(batch_size, opt.neg_batch_sample).zero_()
            sample_batch_neg(answerIdx[:,rnd], opt_answerIdx[:,rnd,:], batch_sample_idx, opt.neg_batch_sample)

            # -----------------------------------------
            # update the Generator using MLE loss.
            # -----------------------------------------
            if opt.update_LM:
                ques_emb_g = netW_g(ques_input, format = 'index')
                his_emb_g = netW_g(his_input, format = 'index')

                ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
                hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))

                featG, ques_hidden1 = netE_g(ques_emb_g, his_emb_g, img_input, \
                                                    ques_hidden1, hist_hidden1, rnd+1)

                _, ques_hidden1 = netG(featG.view(1, -1, opt.ninp), ques_hidden1)
                # MLE loss for generator
                ans_emb = netW_g(ans_input)
                logprob, _ = netG(ans_emb, ques_hidden1)
                lm_loss = critLM(logprob, ans_target.view(-1, 1))
                lm_loss = lm_loss / torch.sum(ans_target.data.gt(0))

                netW_g.zero_grad()
                netG.zero_grad()
                netE_g.zero_grad()

                lm_loss.backward()
                optimizerLM.step()
                err_lm += lm_loss.data[0]
                err_lm_tmp += lm_loss.data[0]

            # sample the answer using gumble softmax sampler.
            ques_emb_g = netW_g(ques_input, format = 'index')
            his_emb_g = netW_g(his_input, format = 'index')

            ques_hidden1 = repackage_hidden(ques_hidden1, batch_size)
            hist_hidden1 = repackage_hidden(hist_hidden1, his_emb_g.size(1))

            featG, ques_hidden1 = netE_g(ques_emb_g, his_emb_g, img_input, \
                                                ques_hidden1, hist_hidden1, rnd+1)

            _, ques_hidden1 = netG(featG.view(1, -1, opt.ninp), ques_hidden1)

            # Gumble softmax to sample the output.
            fake_onehot = []
            fake_idx = []
            noise_input.data.resize_(ans_length, batch_size, vocab_size+1)
            noise_input.data.uniform_(0,1)

            ans_sample = ans_input[0]
            for di in range(ans_length):
                ans_emb = netW_g(ans_sample, format = 'index')
                logprob, ques_hidden1 = netG(ans_emb.view(1,-1,opt.ninp), ques_hidden1)
                one_hot, idx = sampler(logprob, noise_input[di], opt.gumble_weight)
                fake_onehot.append(one_hot.view(1, -1, vocab_size+1))
                fake_idx.append(idx)
                if di+1 < ans_length:
                    ans_sample = idx

            # convert the list into the tensor variable.
            fake_onehot = torch.cat(fake_onehot, 0)
            fake_idx = torch.cat(fake_idx,0)

            fake_len.resize_(batch_size).fill_(ans_length-1)
            for di in range(ans_length-1, 0, -1):
                fake_len.masked_fill_(fake_idx.data[di].eq(vocab_size), di)

            # generate fake mask.
            fake_mask.data.resize_(fake_idx.size()).fill_(1)
            # get the real, wrong and fake index.
            for b in range(batch_size):
                fake_mask.data[:fake_len[b]+1, b] = 0

            # apply the mask on the fake_idx.
            fake_idx.masked_fill_(fake_mask, 0)

            # get the fake diff mask.
            #fake_diff_mask = torch.sum(fake_idx == ans_target, 0) != 0
            fake_onehot = fake_onehot.view(-1, vocab_size+1)
            
            ######################################
            # Discriminative trained generative model.
            ######################################
            # forward the discriminator again.
            ques_emb_d = netW_d(ques_input, format = 'index')
            his_emb_d = netW_d(his_input, format = 'index')

            ques_hidden2 = repackage_hidden(ques_hidden2, batch_size)
            hist_hidden2 = repackage_hidden(hist_hidden2, his_emb_d.size(1))

            featD, _ = netE_d(ques_emb_d, his_emb_d, img_input, \
                                        ques_hidden2, hist_hidden2, rnd+1)

            ans_real_emb = netW_d(ans_target, format='index')
            #ans_wrong_emb = netW_d(wrong_ans_input, format='index')
            ans_fake_emb = netW_d(fake_onehot, format='onehot')
            ans_fake_emb = ans_fake_emb.view(ans_length, -1, opt.ninp)

            real_hidden = repackage_hidden(real_hidden, batch_size)
            #wrong_hidden = repackage_hidden(wrong_hidden, ans_wrong_emb.size(1))
            fake_hidden = repackage_hidden(fake_hidden, batch_size)

            fake_feat = netD(ans_fake_emb, fake_idx, fake_hidden, vocab_size)
            real_feat = netD(ans_real_emb, ans_target, real_hidden, vocab_size)

            d_g_loss, g_fake = critG(featD, real_feat, fake_feat)#, fake_diff_mask.detach())

            netW_g.zero_grad()
            netG.zero_grad()
            netE_g.zero_grad()

            d_g_loss.backward()
            optimizerG.step()

            err_g += d_g_loss.data[0]
            err_g_tmp += d_g_loss.data[0]
            err_g_fake_tmp += g_fake

            count += 1

        i += 1
        loss_store.append({'iter':i, 'err_lm':err_lm_tmp/10, 'err_d':err_d_tmp/10, 'err_g':err_g_tmp/10, \
                            'd_fake': err_d_fake_tmp/10, 'g_fake':err_g_fake_tmp/10})

        if i % 20 == 0:
            print ('Epoch:%d %d/%d, err_lm %4f, err_d %4f, err_g %4f, d_fake %4f, g_fake %4f' \
                % (epoch, i, len(dataloader), err_lm_tmp/10, err_d_tmp/10, err_g_tmp/10, err_d_fake_tmp/10, \
                    err_g_fake_tmp/10))


    #average_loss = average_loss / count
    err_g = err_g / count
    err_d = err_d / count
    err_lm = err_lm / count

    return err_lm, err_d, err_g, loss_store

def val():
    netE_g.eval()
    netE_d.eval()
    netW_g.eval()
    netW_d.eval()

    netG.eval()
    netD.eval()

    n_neg = 100
    ques_hidden1 = netE_g.init_hidden(opt.batchSize)
    ques_hidden2 = netE_d.init_hidden(opt.batchSize)

    hist_hidden1 = netE_d.init_hidden(opt.batchSize)
    hist_hidden2 = netE_g.init_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    data_iter_val = iter(dataloader_val)

    count = 0
    i = 0
    rank_G = []
    rank_D = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                    opt_answerT, answer_ids, answerLen, opt_answerLen, img_id = data

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
            # Evaluate the Generator:
            _, ques_hidden1 = netG(featG.view(1,-1,opt.ninp), ques_hidden1)
            #_, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)
            # extend the hidden
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
for epoch in range(opt.start_epoch+1, opt.niter):

    t = time.time()
    train_loss_lm, train_loss_d, train_loss_g, loss_store = train(epoch)
    print ('Epoch: %d LM loss %4f Discriminator loss %4f Generator loss %4f Time: %3f' \
                            % (epoch, train_loss_lm, train_loss_d, train_loss_g, time.time()-t))

    train_his = {'lossLM': train_loss_lm, 'loss_G':train_loss_g, 'loss_D':train_loss_d, 'loss_store':loss_store}
    if epoch % opt.eval_iter == 0:
        print('Evaluating ... ')
        rank_G, rank_D = val()
        R1 = np.sum(np.array(rank_G)==1) / float(len(rank_G))
        R5 =  np.sum(np.array(rank_G)<=5) / float(len(rank_G))
        R10 = np.sum(np.array(rank_G)<=10) / float(len(rank_G))
        ave = np.sum(np.array(rank_G)) / float(len(rank_G))
        mrr = np.sum(1/(np.array(rank_G, dtype='float'))) / float(len(rank_G))
        print ('Generator: %d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
        val_his_G = {'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}

        R1 = np.sum(np.array(rank_D)==1) / float(len(rank_D))
        R5 =  np.sum(np.array(rank_D)<=5) / float(len(rank_D))
        R10 = np.sum(np.array(rank_D)<=10) / float(len(rank_D))
        ave = np.sum(np.array(rank_D)) / float(len(rank_D))
        mrr = np.sum(1/(np.array(rank_D, dtype='float'))) / float(len(rank_D))
        print ('Discriminator: %d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
        val_his_D = {'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}

        history.append({'epoch':epoch, 'train': train_his, 'val_G': val_his_G, 'val_D':val_his_D})

    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW_d': netW_d.state_dict(),
                    'netW_g': netW_g.state_dict(),
                    'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                    'netE_d': netE_d.state_dict(),
                    'netE_g': netE_g.state_dict()},
                    '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' %(save_path), 'w'))
