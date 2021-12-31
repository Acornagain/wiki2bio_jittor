#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import jittor as jt
import argparse
import time
import random
from SeqUnit import *
from AdamClip import AdamClip
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import *

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_size", type=int, default=500, help="Size of each layer.")
parser.add_argument("--emb_size", type=int, default=400, help="Size of embedding.")
parser.add_argument("--field_size", type=int, default=50, help="Size of embedding.")
parser.add_argument("--pos_size", type=int, default=5, help="Size of embedding.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size of train set.")
parser.add_argument("--epoch", type=int, default=50, help="Number of training epoch.")
parser.add_argument("--source_vocab", type=int, default=20003, help="vocabulary size.")
parser.add_argument("--field_vocab", type=int, default=1480, help="vocabulary size.")
parser.add_argument("--position_vocab", type=int, default=31, help="vocabulary size.")
parser.add_argument("--target_vocab", type=int, default=20003, help="vocabulary size.")
parser.add_argument("--report", type=int, default=10000, help="report valid results after some steps.")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="learning rate.")

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--load', type=str, default='0', help='load directory')
parser.add_argument('--dir', type=str, default='processed_data_disorder', help='data set directory')
parser.add_argument('--limits', type=int, default=0, help='max data set size')

parser.add_argument("--dual_attention", type=bool, default=True, help='dual attention layer or normal attention')
parser.add_argument("--fgate_encoder", type=bool, default=True, help='add field gate in encoder lstm')

parser.add_argument("--field", type=bool, default=False, help='concat field information to word embedding')
parser.add_argument("--position", type=bool, default=False, help='concat position information to word embedding')
parser.add_argument("--encoder_pos", type=bool, default=True, help='position information in field-gated encoder')
parser.add_argument("--decoder_pos", type=bool, default=True, help='position information in dual attention decoder')

parser.add_argument("--plot_attention", type=bool, default=False, help='just plots attention')
parser.add_argument("--test_disorder_fields", type=bool, default=False, help='disorder fields')
parser.add_argument("--train_disorder_fields", type=bool, default=False, help='disorder fields')
parser.add_argument("--load_dir", type=str, default=None)

args = parser.parse_args()

last_best = 0.0

gold_path_test = 'processed_data_disorder/test/test_split_for_rouge/gold_summary_'
gold_path_valid = 'processed_data_disorder/valid/valid_split_for_rouge/gold_summary_'

# test phase
if args.plot_attention is False:
    if args.load != "0":
        save_dir = 'results/res/' + args.load + '/'
        save_file_dir = save_dir + 'files/'
        pred_dir = 'results/evaluation/' + args.load + '/'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
        pred_path = pred_dir + 'pred_summary_'
        pred_beam_path = pred_dir + 'beam_summary_'
    # train phase
    else:
        prefix = str(int(time.time() * 1000))
        save_dir = 'results/res/' + prefix + '/'
        save_file_dir = save_dir + 'files/'
        pred_dir = 'results/evaluation/' + prefix + '/'
        os.mkdir(save_dir)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
        pred_path = pred_dir + 'pred_summary_'
        pred_beam_path = pred_dir + 'beam_summary_'

    log_file = save_dir + 'log.txt'

def init_debug(model):
    model.enc_lstm.W = jt.array(np.load('/root/root/Test/tf_data/enc_lstm_w.npy'))
    model.dec_lstm.W = jt.array(np.load('/root/root/Test/tf_data/dec_lstm_w.npy'))
    model.dec_out.W = jt.array(np.load('/root/root/Test/tf_data/dec_out_w.npy'))
    model.word_embedding.weight = jt.array(np.load('/root/root/Test/tf_data/embed.npy'))
    model.att_layer.Wh = jt.array(np.load('/root/root/Test/tf_data/att_wh.npy'))
    model.att_layer.bh = jt.array(np.load('/root/root/Test/tf_data/att_bh.npy'))
    model.att_layer.Ws = jt.array(np.load('/root/root/Test/tf_data/att_ws.npy'))
    model.att_layer.bs = jt.array(np.load('/root/root/Test/tf_data/att_bs.npy'))
    model.att_layer.Wo = jt.array(np.load('/root/root/Test/tf_data/att_wo.npy'))
    model.att_layer.bo = jt.array(np.load('/root/root/Test/tf_data/att_bo.npy'))

def train(dataloader, model):
    write_log("#######################################################")
    for k in list(vars(args).keys()):
        write_log('%s: %s' % (k, vars(args)[k]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    k = 0
    loss, start_time = 0.0, time.time()
    optimizer = jt.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    first = True
    for _ in range(args.epoch):
        for x in dataloader.batch_iter(trainset, args.batch_size, True, disorder_fields=args.train_disorder_fields):
            model.train()
            lloss = model(x)
            loss += jt.to_float(lloss)
            optimizer.zero_grad()
            optimizer.backward(lloss)
            optimizer.clip_grad_norm(5.0, 2)
            optimizer.step()
            # for pg in optimizer.param_groups:
            #     if "grads" in pg:
            #         for g in pg["grads"]:
            #             print(jt.mean(g))
            #         print()
            
            k += 1
            # jt.display_memory_info()
            progress_bar(k%args.report, args.report)
            if (k % args.report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // args.report, loss, cost_time))
                if k // args.report >= 1:
                    ksave_dir = save_model(model, save_dir, k // args.report)
                    write_log(evaluate(dataloader, model, ksave_dir, 'test', disorder=args.train_disorder_fields))
                    items = [('name', 'frederic jackson'), ('birthdate', '9 january 1968'), ('birthplace', 'uccle , belgium'), 
                    ('nationality', 'belgium'), ('occupation', 'film director'), ('article_title', 'frederic jackson')]
                    # random.shuffle(items)
                    x = format(items)
                    if args.dual_attention:
                        generate_dual_attention_plot(ksave_dir + 'model', x, save_dir=ksave_dir)
                    else:
                        generate_attention_plot(ksave_dir + 'model', x, save_dir=ksave_dir)
                loss, start_time = 0.0, time.time()
                
                    


def test(dataloader, model):
    write_log("#######################################################")
    for k in list(vars(args).keys()):
        write_log('%s: %s' % (k, vars(args)[k]))
    write_log("#######################################################")
    write_log(evaluate(dataloader, model, save_dir, 'test'))
    
def disorder_test(load_dir):
    model = SeqUnit(hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, lr=args.learning_rate)
    model.load(load_dir)
    model.eval()
    dataloader = DataLoader(args.dir, args.limits)
    write_log(evaluate(dataloader, model, save_dir, 'test', True))

def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    save_file = nnew_dir + 'model'
    model.save(save_file)
    return nnew_dir

def evaluate(dataloader, model, ksave_dir, mode='valid', disorder=False):
    model.eval()
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = args.dir + "/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = args.dir + "/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set

    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []

    k = 0
    for x in dataloader.batch_iter(evalset, args.batch_size, False, disorder):
        predictions, atts = model(x)
        # prediction: (batch_size, max_len, num_vocab)
        atts = np.squeeze(atts.data)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        # if it is an unknown token
                        sub = texts[k][np.argmax(atts[tk, : len(texts[k]), idx])]
                        # why  choose this one?
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")

    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                  (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result
    # print result
    if mode == 'valid':
        print(result)

    return result


def write_log(s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():

    copy_file(save_file_dir)
    dataloader = DataLoader(args.dir, args.limits)
    print("start initializing SeqUnit")
    model = SeqUnit(hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, lr=args.learning_rate)
    # copy_file(save_file_dir)
    # init_debug(model)
    if args.load_dir is not None:
        model.load(args.load_dir)
    if args.mode == 'train':
        train(dataloader, model)
    else:
        test(dataloader, model)

def generate_attention_plot(load_dir, x, normalize_on_decout_axis=False, save_dir=None):

    from matplotlib import pyplot as plt

    model = SeqUnit(hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, lr=args.learning_rate)
    model.load(load_dir)
    model.eval()

    v = Vocab()

    tokens, atts = model(x)
    enc_in = []
    enc_fd = []
    out = []

    for id in x['enc_in'][0]:
        enc_in.append(v.id2word(id))
    for id in x['enc_fd'][0]:
        enc_fd.append(v.id2key(id))
        
    for word in tokens.reshape(-1):
        tmp = v.id2word(word.item())
        if (tmp == '-lrb-'):
            out.append('(')
        elif (tmp == '-rrb-'):
            out.append(')')
        elif (tmp == 'END_TOKEN'):
            out.append('<eos>')
        else:
            out.append(v.id2word(word.item()))

    fig, ax = plt.subplots(figsize=(10,10))
    atts = atts.detach().numpy().reshape(atts.shape[0], atts.shape[1])
    ax.imshow(atts, cmap=plt.cm.Greens)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(enc_in)))
    ax.set_yticks(np.arange(len(out)))
    ax.set_xticklabels(enc_in, fontsize=22)
    ax.set_yticklabels(out, fontsize=22)
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    a = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    a = a.reshape(w, h, 4)
    if save_dir is not None:
        plt.savefig(save_dir + '/alpha.png', dpi=400)
    return a



def generate_dual_attention_plot(load_dir, x, normalize_on_decout_axis=False, save_dir=None):

    from matplotlib import pyplot as plt
    
    model = SeqUnit(hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, lr=args.learning_rate)
    model.load(load_dir)
    model.eval()
    
    v = Vocab()
    alpha, beta, gamma = [], [], [] 
    tokens, atts = model(x, alpha, beta, gamma)
    alpha = np.array(alpha)
    beta = np.array(beta)
    gamma = np.array(gamma)
    alpha = alpha.reshape(alpha.shape[0], alpha.shape[1])
    beta = beta.reshape(beta.shape[0], beta.shape[1])
    gamma = gamma.reshape(gamma.shape[0], gamma.shape[1])
    if not normalize_on_decout_axis:
        print(np.mean(gamma), np.mean(beta), np.mean(alpha))
        print(np.sum(alpha * beta, axis=1))
        g_s = np.sum(gamma, axis=1)
        gamma = gamma / g_s.reshape(-1, 1)
        a_s = np.sum(alpha, axis=1)
        alpha = alpha / a_s.reshape(-1, 1)
        b_s = np.sum(beta, axis=1)
        beta = beta / b_s.reshape(-1, 1)
        print(g_s)
        print(a_s)
        print(b_s)
        
    enc_in = []
    enc_fd = []
    out = []

    for id in x['enc_in'][0]:
        enc_in.append(v.id2word(id))
    for id in x['enc_fd'][0]:
        enc_fd.append(v.id2key(id))
        
    for word in tokens.reshape(-1):
        tmp = v.id2word(word.item())
        if (tmp == '-lrb-'):
            out.append('(')
        elif (tmp == '-rrb-'):
            out.append(')')
        elif (tmp == 'END_TOKEN'):
            out.append('<eos>')
        else:
            out.append(v.id2word(word.item()))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(gamma, cmap=plt.cm.Greens)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(enc_in)))
    ax.set_yticks(np.arange(len(out)))
    ax.set_xticklabels(enc_in, fontsize=22)
    ax.set_yticklabels(out, fontsize=22)
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    g = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    g = g.reshape(w, h, 4)
    if save_dir is not None:
        plt.savefig(save_dir + '/gamma.png', dpi=400)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(alpha, cmap=plt.cm.Greens)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(enc_in)))
    ax.set_yticks(np.arange(len(out)))
    ax.set_xticklabels(enc_in, fontsize=22)
    ax.set_yticklabels(out, fontsize=22)
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    a = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    a = a.reshape(w, h, 4)
    if save_dir is not None:
        plt.savefig(save_dir + '/alpha.png', dpi=400)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(beta, cmap=plt.cm.Greens)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(enc_fd)))
    ax.set_yticks(np.arange(len(out)))
    ax.set_xticklabels(enc_fd, fontsize=22)
    ax.set_yticklabels(out, fontsize=22)
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    b = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    b = b.reshape(w, h, 4)
    if save_dir is not None:
        plt.savefig(save_dir + '/beta.png', dpi=400)

    return a, b, g

def format(items):
    v = Vocab()

    enc_in = []
    enc_fd = []
    enc_pos = []
    enc_rpos = []
    length = 0
    
    for item in items:
        words = item[1].split(' ')
        for i in range(len(words)):
            enc_fd.append(v.key2id(item[0]))
            enc_in.append(v.word2id(words[i]))
            enc_pos.append(i + 1)
            enc_rpos.append(len(words) - i)
            length += 1
    
    x = dict()
    x['enc_in'] = [enc_in]
    x['enc_fd'] = [enc_fd]
    x['enc_pos'] = [enc_pos]
    x['enc_rpos'] = [enc_rpos]
    x['enc_len'] = [length]
    x['dec_in'] = [0]
    x['dec_len'] = [0]
    x['dec_out'] = [0]

    return x

if __name__=='__main__':
    if args.plot_attention:
        jt.flags.use_cuda = 0
        items = [('name', 'frederic jackson'), ('birthdate', '9 january 1968'), ('birthplace', 'uccle , belgium'), 
        ('nationality', 'belgium'), ('occupation', 'film director'), ('article_title', 'frederic jackson')]
        # random.shuffle(items)
        x = format(items)
        if args.dual_attention:
            generate_dual_attention_plot('results/res/1640447125705/loads/8/model', x, save_dir='att_plots')
        else:
            generate_attention_plot('results/res/seq2seq/loads/25/model', x, save_dir='seq2seq_att_plots')
    elif args.test_disorder_fields:
        print('disorder test')
        jt.flags.use_cuda = 1
        disorder_test('results/res/seq2seq+field+pos/loads/29/model')
    else:
        jt.flags.use_cuda = 1
        main()
