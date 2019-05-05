# this script is to train an LSTM encoder such that VQA answer is predictable from Question-Explain pair.
# We used Billinear Attention while training Q-E-A pair..
# Experiment seems to fail

import os
import sys


import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from build_vocab import Vocabulary
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC
from model_explain import GuideVfeat, CaptionEncoder, Relev_Check_by_IDX, UNCorrXAI
from nltk.tokenize import word_tokenize
import utils_hsc as utils_save

sys.path.append('D:\\VQA\\BAN')

import codes.base_model as base_model

from codes.dataset import Dictionary, VQAFeatureDataset
from model_explain import CaptionEncoder
from data_loader import vqaE_CapEnc_Loader
from model_VQAE import EnsembleVQAE
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
from data_loader import VQAFeatureLoaderAdapter


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='data/VQA_E_vocab.pkl',
                        help='path for VQA_E vocabulary wrapper')
    parser.add_argument('--vqa_E_train_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json',
                        help='path for train annotation json file')
    parser.add_argument('--vqa_E_val_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json',
                        help='path for validation annotation json file')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--hidden_size_BAN', type=int, default=1280, help='dimension of GRU hidden states in BAN')
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--num_epoch',type=int,default=500)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--t_method', type=str, default='uncorr')
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--hsc_epoch', type=int, default=13)
    parser.add_argument('--log_step', type=int, default=50, help='step size for prining log info')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--model_num_CE', type=int, default=1, help='model number of caption encoder')
    args = parser.parse_args()
    return args

def requires_grad_Switch(module,isTrain=True):

    for p in module.parameters():
        p.requires_grad=isTrain

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    #pdb.set_trace()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    name = 'train'

    question_path = os.path.join(
        'codes\\tools\\data', 'v2_OpenEnded_mscoco_%s_questions.json' % \
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    dictionary_vqa = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    with open(args.vocab_path, 'rb') as f:
        vocab_VQAE = pickle.load(f)


    vqaE_loader = vqaE_CapEnc_Loader('train+val', dictionary_vqa, vocab_VQAE, args.batch_size, shuffle=True, num_workers=0)


    caption_encoder = CaptionEncoder(args.embed_size, args.hidden_size_BAN, args.hidden_size,
                                     vqaE_loader.dataset.datasets[0].num_ans_candidates,bidirectional_=True).to(device)

    params1=caption_encoder.parameters()
    optimizer1 = torch.optim.Adamax(params1, lr=args.learning_rate)

    if not os.path.exists(os.path.join('model_xai', 'caption_encoder','model{}'.format(args.model_num_CE))):
        os.makedirs(os.path.join('model_xai',  'caption_encoder','model{}'.format(args.model_num_CE)))

    save_path = os.path.join('model_xai', 'caption_encoder')

    constructor = 'build_%s' % args.model
    vqa_dset = VQAFeatureDataset(args.split, dictionary_vqa, adaptive=True)
    model = getattr(base_model, constructor)(vqa_dset, args.num_hid, args.op, args.gamma).cuda()

    model_path = args.input + '/model%s.pth' % \
                 ('' if 0 > args.epoch else '_epoch%d' % args.epoch)
    print('loading vqa %s' % model_path)
    model_data = torch.load(model_path)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)

    ban_vqaE = EnsembleVQAE(model, decoder).to(device)

    model_ban_vqaE_path = os.path.join(
            'model_xai', 'vqaE', 'vqaE-{}-Final.pth'.format(20))

    print('loading ban_vqaE model %s' % model_ban_vqaE_path)
    model_ban_vqaE_data = torch.load(model_ban_vqaE_path)

    ban_vqaE.load_state_dict(model_ban_vqaE_data['model_state']) # ban_vqaE의 decoder 로드하기

    criterion = nn.CrossEntropyLoss()

    requires_grad_Switch(model,isTrain=False)
    requires_grad_Switch(decoder, isTrain=False)
    captionMaxSeqLength=40
    total_step = len(vqaE_loader)

    for epoch in range(args.num_epoch):
        for i,(_,_,Wq,answer,Wc,lengths, num_objs) in enumerate(vqaE_loader):

            caption_encoder.zero_grad()

            Wq=Wq.squeeze(1)
            Wc=Wc.squeeze(1)
            Wc=Wc.type(torch.LongTensor)
            Wc=Wc.cuda()
            q_emb=model.module.extractQEmb(Wq)
            #states=[None]
            states=None
            input_list=[]
            num_objs=num_objs.cuda()

            # for k in range(min(captionMaxSeqLength,Wc.size(0))):
            #     input_list.append(decoder.embed(Wc[:,k]))
            #     #hiddens, states_tmp = caption_encoder(input_list[k], states[k])
            #     #states.append(states_tmp)
            # inputs=torch.stack(input_list,1)

            inputs = pack_padded_sequence(decoder.embed(Wc), lengths, batch_first=True)

            hiddens_packed, states = caption_encoder(inputs, states)
            hiddens, input_sizes = pad_packed_sequence(hiddens_packed, batch_first=True)
            preds=caption_encoder.forward_CL_ATT(q_emb,hiddens)
            #outs=caption_encoder.forward_DoubleLSTM_out(hiddens)
            answer=answer.cuda()
            Loss = instance_bce_with_logits(preds, answer)
            Loss.backward()
            optimizer1.step()

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, i, total_step, Loss.item(), np.exp(Loss.item())))

        # Save the model checkpoints
        save_path = os.path.join(
            'model_xai', 'caption_encoder', 'model{}'.format(args.model_num_CE), 'ce-{}-Final.pth'.format(epoch + 1))
        ####################################################################################################
        # implement utils.save_xai_model

        utils_save.save_ce_module(save_path, caption_encoder, epoch, optimizer1)

        ####################################################################################################





