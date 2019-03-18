
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
from model_explain import Sen_Sim
from model_explain import Relev_Check, CaptionVocabCandidate

from address_server_XAI import *

sys.path.append(addr_BAN)

from codes.dataset import Dictionary, VQAFeatureDataset
from data_loader import BottomUp_get_loader, VQAE_FineTunning_loader
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from address_server_XAI import *
import codes.base_model as base_model

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
    parser.add_argument('--split', type=str, default='test2014')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--hsc_epoch', type=int, default=16)
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--BUTD_path', type=str, default='models_BUTD/',
                        help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_threshold_0.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='saved_figs_with_caption/')
    parser.add_argument('--x_method', type=str, default='weight_only') # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='uncorr') # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne') # BestOne, BeamSearch
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--isBUTD', type=bool, default=False)
    parser.add_argument('--isUnion', type=bool, default=False)
    parser.add_argument('--isFeeding', type=bool, default=False)
    parser.add_argument('--paramH', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--result_json_path', type=str, default='test_json/', help='saving path for captioning json')
    parser.add_argument('--test_target', type=str, default='COCO', help='COCO or KARPATHY')
    parser.add_argument('--method', type=str, default='VQAE&LRCN', help='applied method')
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]

def get_image(q_id, Dict_qid2vid, data_root=addr_test_imgs): #q_id => single scalar
    img_id=Dict_qid2vid[q_id]
    img_name='COCO_test2015_%012d.jpg' % img_id
    img_path=os.path.join(data_root,img_name)
    img=plt.imread(img_path)

    return img


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def check_captions(caption_generator, dataloader,Dict_qid2vid, vocab,save_fig_loc,x_method_, t_method_, s_method_, args_):
    N = len(dataloader.dataset)
    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()
    print('t_method : {}, x_method : {}, isBUTD : {}, isUnion : {}'.format(t_method_,x_method_,args.isBUTD,args.isUnion))
    captions_list = []
    img_id_list=[]
    for i,(v, b, q, ans, img_ids, _) in enumerate(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        with torch.no_grad():
            v = Variable(v).cuda()
            q=q.squeeze(1)


            #print(args.isUnion)
            generated_captions, logits, _, encoded_feats, Vmat = caption_generator.generate_caption_n_context(v, b, q,t_method=t_method_, x_method=args.x_method, s_method=s_method_,
                                                                                                                isBUTD=args.isBUTD , isUnion=args.isUnion, useVQA=True)
            beam_list = []
            idx += batch_size
            question_list=[]
            answer_list=[]
            for idx2 in range(batch_size):
                question_list.append(get_question(q.data[idx2], dataloader))
                answer_list.append(get_answer(logits.data[idx2], dataloader))



        for b_ in range(batch_size):
            caption = [vocab.idx2word[generated_captions[b_][w_idx].item()] for w_idx in
                       range(generated_captions.size(1))]

            # ##################################################################### EXPLAIN ##################################################################################
            # word_candidate_idx_in_coco_vocab = CaptionVocabCandidate(question_list[b_], answer_list[b_], vocab)
            # if (args.isBUTD):
            #     generated_captions = caption_generator.generate_explain(Vmat[b_].unsqueeze(0), encoded_feats[b_].unsqueeze(0),
            #                                                             word_candidate_idx_in_coco_vocab,
            #                                                             t_method='uncorr',
            #                                                             x_method='weight_only',
            #                                                             s_method=s_method_,
            #                                                             isBUTD=True,
            #                                                             isUnion=True,
            #                                                             model_num=args_.model_num)
            # else:
            #     generated_captions = caption_generator.generate_explain(None, encoded_feats[b_].unsqueeze(0),
            #                                                             word_candidate_idx_in_coco_vocab,
            #                                                             t_method='uncorr',
            #                                                             x_method='weight_only',
            #                                                             s_method=s_method_,
            #                                                             model_num=args_.model_num)
            # caption = [vocab.idx2word[generated_captions[0][w_idx].item()] for w_idx in
            #            range(generated_captions.size(1))]


            # print(captions_list[0])

            ##################################################################### EXPLAIN ##################################################################################
            caption = caption_refine(caption, model_num=args_.model_num)

            captions_list.append(caption)




        img_id_list.extend(img_ids)

    bar.update(idx)
    result_json=make_json(captions_list, img_id_list)

    with open(args.result_json_path + '/captions_%s_%s_results.json' \
              % (args.split, args.method), 'w') as f:

        json.dump(result_json, f)


def make_json(captions, ImgIds):
    utils.assert_eq(len(captions), len(ImgIds))
    results = []
    for i in range(len(captions)):
        result = {}
        result['image_id'] = ImgIds[i]
        result['caption'] = captions[i]
        results.append(result)
    return results

#######################################################################################

def caption_refine(explains, NumBeams=1, model_num=1):

    x_caption = ''
    for num_sen in range(NumBeams):
        if(NumBeams > 1):
            explain=explains[num_sen]
        else:
            explain=explains
        for word_ in explain:
            if word_ == '<start>':
                if model_num <7:
                    x_caption = ''
            elif word_ == '<end>':
                break
            else:
                x_caption = x_caption + ' ' + word_

        if (NumBeams > 1):
            x_caption=x_caption+'\n'

    return x_caption


#######################################################################################`

if __name__ == '__main__':
    args = parse_args()
    if (args.t_method == 'uncorr'):
        args.isUnion = True

    torch.backends.cudnn.benchmark = True

    name = args.split
    Dict_qid2vid=[]

    if not os.path.exists(
            args.result_json_path):
        os.makedirs(args.result_json_path)

    dictionary_vqa = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Visualization through eval_dataset

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    vqaE_loader = VQAE_FineTunning_loader('eval', dictionary_vqa, vocab, args.batch_size, shuffle=False, num_workers=0)



    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    if(args.isBUTD):

        decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers,
                                 paramH=args.paramH).to(device)
        args.save_fig_loc=args.save_fig_loc+args.hsc_path
        #args.hsc_path='models_BUTD/'
        model_file='model-{}.pth'.format(20)
    else:

        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
        model_file = 'model-{}.pth'.format(16)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(vqaE_loader.dataset, args.num_hid, args.op,
                                             args.gamma).cuda()

    model_path = args.input + '/model%s.pth' % \
                 ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

    print('loading %s' % model_path)
    model_data = torch.load(model_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    def process(args, model, eval_loader,Dict_qid2vid,vocab):

        if (args.isBUTD):
            model_hsc_path = os.path.join(
                args.BUTD_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim), model_file)
        else:
            model_hsc_path = os.path.join(
                args.hsc_path, args.t_method, 'model{}_LR{}'.format(args.model_num,args.LRdim), model_file)

        print('loading hsc datas %s' % model_hsc_path)
        model_hsc_data=torch.load(model_hsc_path)

        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])



        #pdb.set_trace()

        encoder.train(False)
        decoder.train(False)

        caption_generator=BAN_HSC(model,encoder,decoder,vocab)

        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues

        if not os.path.exists(os.path.join(args.save_fig_loc,'model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method)):
            os.makedirs(os.path.join(args.save_fig_loc,'model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method))
        check_captions(caption_generator, eval_loader, Dict_qid2vid,vocab,args.save_fig_loc,args.x_method, args.t_method, args.s_method, args)

        ################################################################################################################





    process(args, model, vqaE_loader,Dict_qid2vid,vocab)
