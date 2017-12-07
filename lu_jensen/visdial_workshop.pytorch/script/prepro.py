import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from unidecode import unidecode
import os
import re
import pdb

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def tokenize_data(data):
    '''
    Tokenize captions, questions and answers
    Also maintain word count if required
    '''
    ques_toks, ans_toks, caption_toks = [], [], []

    print data['split']
    print 'Tokenizing captions...'
    for i in data['data']['dialogs']:
        caption = word_tokenize(i['caption'])
        caption_toks.append(caption)

    print 'Tokenizing questions...'
    for i in data['data']['questions']:
        ques_tok = word_tokenize(i + '?')
        ques_toks.append(ques_tok)

    print 'Tokenizing answers...'
    for i in data['data']['answers']:
        ans_tok = word_tokenize(i)
        ans_toks.append(ans_tok)

    return ques_toks, ans_toks, caption_toks

def build_vocab(data, ques_toks, ans_toks, caption_toks, params):
    count_thr = args.word_count_threshold
    
    i = 0
    counts = {}
    for imgs in data['data']['dialogs']:
        caption = caption_toks[i]
        i += 1

        for w in caption:
            counts[w] = counts.get(w, 0) + 1

        for dialog in imgs['dialog']:
            question = ques_toks[dialog['question']]
            answer = ans_toks[dialog['answer']]
            
            for w in question + answer:
                counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

    print 'inserting the special UNK token'
    vocab.append('UNK')

    return vocab


def encode_vocab(questions, answers, captions, wtoi):

    ques_idx, ans_idx, cap_idx = [], [], []

    for txt in captions:
        ind = [wtoi.get(w, wtoi['UNK']) for w in txt]
        cap_idx.append(ind)

    for txt in questions:
        ind = [wtoi.get(w, wtoi['UNK']) for w in txt]
        ques_idx.append(ind)

    for txt in answers:
        ind = [wtoi.get(w, wtoi['UNK']) for w in txt]
        ans_idx.append(ind)

    return ques_idx, ans_idx, cap_idx


def create_mats(data, ques_idx, ans_idx, cap_idx, params):
    N = len(data['data']['dialogs'])
    num_round = 10
    max_ques_len = params.max_ques_len
    max_cap_len = params.max_cap_len
    max_ans_len = params.max_ans_len

    captions = np.zeros([N, max_cap_len], dtype='uint32')
    questions = np.zeros([N, num_round, max_ques_len], dtype='uint32')
    answers = np.zeros([N, num_round, max_ans_len], dtype='uint32')
   
    answer_index = np.zeros([N, num_round],  dtype='uint32')

    caption_len = np.zeros(N, dtype='uint32')
    question_len = np.zeros([N, num_round],  dtype='uint32')
    answer_len = np.zeros([N, num_round], dtype='uint32')
    options = np.zeros([N, num_round, 100], dtype='uint32')

    image_list = []
    for i, img in enumerate(data['data']['dialogs']):
        image_id = img['image_id']
        image_list.append(image_id)

        cap_len = len(cap_idx[i][0:max_cap_len])
        caption_len[i] = cap_len
        captions[i][0:cap_len] = cap_idx[i][0:cap_len]


        for j, dialog in enumerate(img['dialog']):
            
            ques_len = len(ques_idx[dialog['question']][0:max_ques_len])
            question_len[i,j] = ques_len
            questions[i,j,:ques_len] = ques_idx[dialog['question']][0:ques_len]

            ans_len = len(ans_idx[dialog['answer']][0:max_ans_len])
            answer_len[i,j] = ans_len
            answers[i,j,:ans_len] = ans_idx[dialog['answer']][0:ans_len]

            options[i,j] = dialog['answer_options']
            answer_index[i,j] = dialog['gt_index']

    options_list = np.zeros([len(ans_idx), max_ans_len], dtype='uint32')
    options_len = np.zeros(len(ans_idx), dtype='uint32')

    for i, ans in enumerate(ans_idx):
        options_len[i] = len(ans[0:max_ans_len])
        options_list[i][0:options_len[i]] = ans[0:max_ans_len]

    return captions, caption_len, questions, question_len, answers, answer_len, \
            options, options_list, options_len, answer_index, image_list

def img_info(image_list, split):
    out = []
    for i,imgId in enumerate(image_list):
        jimg = {}
        jimg['imgId'] = imgId
        file_name = 'COCO_%s_%012d.jpg' %(split, imgId)
        jimg['path'] = os.path.join(split, file_name)
        out.append(jimg)

    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input files
    parser.add_argument('-input_json_train', default = '../data/visdial_0.9_train.json', help='Input `train` json file')
    parser.add_argument('-input_json_val', default='../data/visdial_0.9_val.json', help='Input `val` json file')

    # Output files
    parser.add_argument('-output_json', default='visdial_params.json', help='Output json file')
    parser.add_argument('-output_h5', default='visdial_data.h5', help='Output hdf5 file')

    # Options
    parser.add_argument('-max_ques_len', default=16, type=int, help='Max length of questions')
    parser.add_argument('-max_ans_len', default=8, type=int, help='Max length of answers')
    parser.add_argument('-max_cap_len', default=24, type=int, help='Max length of captions')
    parser.add_argument('-word_count_threshold', default=5, type=int, help='Min threshold of word count to include in vocabulary')

    args = parser.parse_args()

    print 'Reading json...'
    data_train = json.load(open(args.input_json_train, 'r'))
    data_val = json.load(open(args.input_json_val, 'r'))

    ques_tok_train, ans_tok_train, cap_tok_train  = tokenize_data(data_train)
    ques_tok_val, ans_tok_val, cap_tok_val  = tokenize_data(data_val)


    vocab = build_vocab(data_train, ques_tok_train, ans_tok_train, cap_tok_train, args)

    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    print 'Encoding based on vocabulary...'
    ques_idx_train, ans_idx_train, cap_idx_train = encode_vocab(ques_tok_train, ans_tok_train, cap_tok_train, wtoi)
    ques_idx_val, ans_idx_val, cap_idx_val = encode_vocab(ques_tok_val, ans_tok_val, cap_tok_val, wtoi)
    

    cap_train, cap_len_train, ques_train, ques_len_train, ans_train, ans_len_train, \
            opt_train, opt_list_train, opt_len_train, ans_index_train, image_list_train = \
                create_mats(data_train, ques_idx_train, ans_idx_train, cap_idx_train, args)

    cap_val, cap_len_val, ques_val, ques_len_val, ans_val, ans_len_val, \
            opt_val, opt_list_val, opt_len_val, ans_index_val, image_list_val = \
                create_mats(data_val, ques_idx_val, ans_idx_val, cap_idx_val, args)

    print 'Saving hdf5...'
    f = h5py.File(args.output_h5, 'w')
    f.create_dataset('ques_train', dtype='uint32', data=ques_train)
    f.create_dataset('ques_len_train', dtype='uint32', data=ques_len_train)
    f.create_dataset('ans_train', dtype='uint32', data=ans_train)
    f.create_dataset('ans_len_train', dtype='uint32', data=ans_len_train)
    f.create_dataset('ans_index_train', dtype='uint32', data=ans_index_train)
    f.create_dataset('cap_train', dtype='uint32', data=cap_train)
    f.create_dataset('cap_len_train', dtype='uint32', data=cap_len_train)
    f.create_dataset('opt_train', dtype='uint32', data=opt_train)
    f.create_dataset('opt_len_train', dtype='uint32', data=opt_len_train)
    f.create_dataset('opt_list_train', dtype='uint32', data=opt_list_train)

    f.create_dataset('ques_val', dtype='uint32', data=ques_val)
    f.create_dataset('ques_len_val', dtype='uint32', data=ques_len_val)
    f.create_dataset('ans_val', dtype='uint32', data=ans_val)
    f.create_dataset('ans_len_val', dtype='uint32', data=ans_len_val)
    f.create_dataset('ans_index_val', dtype='uint32', data=ans_index_val)
    f.create_dataset('cap_val', dtype='uint32', data=cap_val)
    f.create_dataset('cap_len_val', dtype='uint32', data=cap_len_val)
    f.create_dataset('opt_val', dtype='uint32', data=opt_val)
    f.create_dataset('opt_len_val', dtype='uint32', data=opt_len_val)
    f.create_dataset('opt_list_val', dtype='uint32', data=opt_list_val)
    f.close()

    out = {}
    out['itow'] = itow # encode the (1-indexed) vocab
    out['img_train'] = img_info(image_list_train, 'train2014')
    out['img_val'] = img_info(image_list_val, 'val2014')

    json.dump(out, open(args.output_json, 'w'))


    #data_val_toks, ques_val_inds, ans_val_inds = encode_vocab(data_val_toks, ques_val_toks, ans_val_toks, word2ind)
