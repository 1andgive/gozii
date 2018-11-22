
import os
import sys


import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pickle
from build_vocab import Vocabulary
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC
from model_explain import GuideVfeat, CaptionEncoder, Relev_Check_by_IDX, UNCorrXAI
from nltk.tokenize import word_tokenize

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
import utils_hsc as utils_save

import pdb


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='XAI')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--hsc_epoch', type=int, default=13)
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--hidden_size_BAN', type=int, default=1280, help='dimension of GRU hidden states in BAN')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--xai_model_loc', type=str, default='model_XAI/')
    parser.add_argument('--x_method', type=str, default='weight_only')
    parser.add_argument('--t_method', type=str, default='uncorr')
    parser.add_argument('--s_method', type=str, default='BestOne')
    parser.add_argument('--HSC_model', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=400, help='step size for saving trained models')
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=13)
    parser.add_argument('--RelScoreThres', type=float, default=0.6)
    parser.add_argument('--CheckConverges', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = torch.max(p,1)
    label_from_vqa=torch.cuda.LongTensor(p.size()).fill_(0)
    for i in range(p.size(0)):
        label_from_vqa[i,idx[i]]=1
    return idx, label_from_vqa






def train_XAI(uncorr_xai, vqa_loader, vocab_Caption, optimizer, args, Dict_AC_2_Q):
    N = len(vqa_loader.dataset)
    M = vqa_loader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N * args.num_epoch).start()
    total_step = len(vqa_loader)
    criterion=nn.CrossEntropyLoss()



    if not os.path.exists(os.path.join('model_xai', args.t_method, args.x_method, args.s_method)):
        os.makedirs(os.path.join('model_xai',  args.t_method, args.x_method, args.s_method))

    save_path = os.path.join('model_xai',  args.t_method, args.x_method, args.s_method)
    is_Init = True


    for epoch in range(args.num_epoch):

        for i, (v, b, q, a_)in enumerate(vqa_loader):
            bar.update(idx)
            batch_size = v.size(0)
            q = q.type(torch.LongTensor)

            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()

            if(epoch % 2 == 0):
                logits, outputs, captions=uncorr_xai(v, b, q, t_method=args.t_method, x_method=args.x_method, s_method=args.s_method, model_num=args.model_num, flag='fix_guide', is_Init=is_Init)
            else:
                logits, outputs=uncorr_xai(v, b, q, t_method=args.t_method, x_method=args.x_method, s_method=args.s_method, model_num=args.model_num, flag='fix_cap_enc', is_Init=is_Init)

            idx += batch_size
            answer_idx, label_from_vqa = get_answer(logits.data, vqa_loader)

            ############################################################################################################
            # Relevance Score 부분은 문제 없으나, network 점검을 위해 잠시 주석처리함
            # if (epoch % 2 == 0):
            #     RelScoreTensor=Relev_Check_by_IDX(captions, q, answer_idx, uncorr_xai.BAN.module.w_emb,Dict_AC_2_Q)
            #     RelevantIDX=torch.ge(RelScoreTensor,args.RelScoreThres)
            #     outputs=outputs[RelevantIDX]
            #     answer_idx = answer_idx[RelevantIDX]

            ############################################################################################################

            Loss = criterion(outputs,answer_idx)

            uncorr_xai.CaptionEncoder.zero_grad()
            uncorr_xai.Guide.zero_grad()

            Loss.backward()

            if (epoch % 2 == 0):
                optimizer[0].step()
            else:
                optimizer[1].step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, i, total_step, Loss.item(), np.exp(Loss.item())))

            if(args.CheckConverges):
                if i > 100:
                    break






        bar.update(idx)
        is_Init = False

        # Save the model checkpoints
        save_path = os.path.join(
            args.xai_model_loc, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim), args.x_method,
            args.s_method, 'xai-model-{}-Final.pth'.format(epoch + 1))
        ####################################################################################################
        # implement utils.save_xai_model

        utils_save.save_xai_module(save_path, uncorr_xai.CaptionEncoder, uncorr_xai.Guide, epoch, optimizer)

        ####################################################################################################

    ####################################################################################################
    # implement utils.save_xai_model

    utils_save.save_xai_module(save_path, uncorr_xai.CaptionEncoder, uncorr_xai.Guide, epoch, optimizer)



if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    name = 'train'

    question_path = os.path.join(
        'codes\\tools\\data', 'v2_OpenEnded_mscoco_%s_questions.json' % \
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])


    Dict_qid2ques={}
    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Load COO-Caption vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # train_dataset

    vqa_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(vqa_dset, args.num_hid, args.op, args.gamma).cuda()

    vqa_loader = DataLoader(vqa_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)



    Dict_IdxA_2_IdxQ={}
    for Wa_set in vqa_dset.ans2label.keys():
        for Wa in word_tokenize(Wa_set):
            if Wa in vqa_dset.dictionary.word2idx.keys():
                if vqa_dset.ans2label[Wa_set] in Dict_IdxA_2_IdxQ.keys():
                    Dict_IdxA_2_IdxQ[vqa_dset.ans2label[Wa_set]].append(vqa_dset.dictionary.word2idx[Wa])
                else:
                    Dict_IdxA_2_IdxQ[vqa_dset.ans2label[Wa_set]]=[vqa_dset.dictionary.word2idx[Wa]]

    Dict_IdxC_2_IdxQ = {}
    for Wc in vocab.word2idx.keys():
        if Wc in vqa_dset.dictionary.word2idx.keys():
            Dict_IdxC_2_IdxQ[vocab.word2idx[Wc]] = vqa_dset.dictionary.word2idx[Wc]


    Dict_AC_2_Q=(Dict_IdxA_2_IdxQ, Dict_IdxC_2_IdxQ)

    #train the encoder/decoder pair this time (fine-tuning stage for dual-loss)
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048,LRdim=args.LRdim).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)



    ################################################################################################################
    # implement these in model.py

    caption_encoder = CaptionEncoder(args.embed_size, args.hidden_size_BAN, args.hidden_size, vqa_dset.num_ans_candidates).to(device)
    guide=GuideVfeat(args.hidden_size_BAN, 2048).to(device)


    ################################################################################################################

    params1 = list(caption_encoder.parameters())
    params2 = list(guide.parameters())

    # params = list(decoder.parameters())
    optimizer1 = torch.optim.Adam(params1, lr=args.learning_rate)
    optimizer2 = torch.optim.Adam(params2, lr=args.learning_rate)

    optimizer=(optimizer1, optimizer2)

    def process(args, model, vqa_loader, optimizer, Dict_AC_2_Q):
        model_path = args.input + '/model%s.pth' % \
                     ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

        print('loading vqa %s' % model_path)
        model_data = torch.load(model_path)

        if(args.t_method == 'mean'):
            model_hsc_path = os.path.join(
                args.hsc_path, args.t_method, 'model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}-400.pth'.format(args.hsc_epoch))
        else:
            model_hsc_path = os.path.join(
                args.hsc_path,args.t_method, 'model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}-400.pth'.format(args.hsc_epoch))

        print('loading hsc(attention caption) %s' % model_hsc_path)
        model_hsc_data=torch.load(model_hsc_path)

        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        #pdb.set_trace()

        # model.train(False) # Do not train VQA part
        # encoder.train(False)
        # decoder.train(False)



        uncorr_xai = UNCorrXAI(model,encoder,decoder,caption_encoder,guide)

        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues

        if not os.path.exists(os.path.join(args.xai_model_loc,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim), args.x_method,args.s_method)):
            os.makedirs(os.path.join(args.xai_model_loc,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim),args.x_method,args.s_method))


        train_XAI(uncorr_xai, vqa_loader, vocab, optimizer, args, Dict_AC_2_Q)

        ################################################################################################################





    process(args, model, vqa_loader, optimizer, Dict_AC_2_Q)
