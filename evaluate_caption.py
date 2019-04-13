## EVALUATE ON COCO DATASET

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
from data_loader import BottomUp_get_loader
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from address_server_XAI import *

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
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--hsc_epoch', type=int, default=0)
    parser.add_argument('--hsc_path', type=str, default='models_BUTD_36/standard_vocab/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=1000, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1000, help='dimension of lstm hidden states')
    #parser.add_argument('--vocab_path', type=str, default='data/vocab_threshold_0.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='saved_figs_with_caption/')
    parser.add_argument('--x_method', type=str, default='weight_only') # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='mean') # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne') # BestOne, BeamSearch
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--isBUTD', type=bool, default=False)
    parser.add_argument('--isUnion', type=bool, default=False)
    parser.add_argument('--isFeeding', type=bool, default=False)
    parser.add_argument('--paramH', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--result_json_path', type=str, default='test_json/', help='saving path for captioning json')
    parser.add_argument('--test_target', type=str, default='COCO', help='COCO or KARPATHY')
    parser.add_argument('--method', type=str, default='mean_butd_fix36_cider0', help='applied method')
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
    for v, b, img_ids in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        with torch.no_grad():
            v = Variable(v).cuda()


            #print(args.isUnion)
            generated_captions, _, _, encoded_feats, Vmat = caption_generator.generate_caption_n_context(v, None, None,t_method=t_method_, x_method=args.x_method, s_method=s_method_,
                                                                                                                isBUTD=args.isBUTD , isUnion=args.isUnion, useVQA=False)
            beam_list = []
            idx += batch_size

            for idx2 in range(len(img_ids)):

                if (s_method_ == 'BestOne'):
                    caption = [vocab.idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                               range(generated_captions.size(1))]
                    beam_list.append(caption)
                elif (s_method_ == 'BeamSearch'):
                    for tmp_idx in range(generated_captions.size(0)):
                        #pdb.set_trace()
                        caption = [vocab.idx2word[generated_captions[tmp_idx][w_idx].item()] for w_idx in
                                   range(generated_captions.size(1))]
                        beam_list.append(caption)

        if (s_method_ == 'BestOne'):
            for b_ in range(batch_size):
                beam_list[b_] = caption_refine(beam_list[b_], model_num=args_.model_num)

        elif (s_method_ == 'BeamSearch'):
            for b_ in range(batch_size):
                beam_list[b_] = caption_refine(beam_list[b_], NumBeams=5)


        captions_list.extend(beam_list)
        img_id_list.extend(img_ids)

    bar.update(idx)
    result_json=make_json(captions_list, img_id_list)

    with open(args.result_json_path + '/captions_%s_%s_results.json' \
              % (args.split, args.method + str(args.hsc_epoch)), 'w') as f:

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

    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Visualization through eval_dataset

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    # Build data loader
    val_json = json.load(open(addr_coco_cap_val_path))
    test_json = json.load(open(addr_coco_cap_test2014_path))
    if (args.split=='test2014'):
        test_loader = BottomUp_get_loader('test2014', addr_coco_cap_test2014_path, vocab,
                                 None, args.batch_size,
                                 shuffle=False, num_workers=0)
    elif(args.split=='val'):
        name='val2014'
        test_loader = BottomUp_get_loader(name, addr_coco_cap_val_path, vocab,
                                          None, args.batch_size,
                                          shuffle=False, num_workers=0)

    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    if(args.isBUTD):

        decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers,
                                 paramH=args.paramH).to(device)
        args.save_fig_loc=args.save_fig_loc+args.hsc_path
        #args.hsc_path='models_BUTD/'
        model_file='model-{}-CiderOpt-GradClip-0.25.pth'.format(args.hsc_epoch)
    else:

        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
        if (args.t_method == 'mean'):
            model_file = 'model-{}'.format(args.hsc_epoch)
        else:
            #model_file = 'model-{}-400.pth'.format(args.hsc_epoch)
            model_file = 'model-{}-400'.format(args.hsc_epoch)

    model = None

    def process(args, model, eval_loader,Dict_qid2vid):


        if(args.t_method == 'mean'):
            model_hsc_path = os.path.join(
                args.hsc_path, args.t_method, 'model{}_LR{}'.format(args.model_num,args.LRdim), model_file)
        else:
            model_hsc_path = os.path.join(
                args.hsc_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim), model_file)

        print('loading hsc datas %s' % model_hsc_path)
        model_hsc_data=torch.load(model_hsc_path)

        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])

        model = None

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





    process(args, model, test_loader,Dict_qid2vid)
