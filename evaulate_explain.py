
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
from model_explain import Relev_Check, CaptionVocabCandidate, CaptionVocabCandidateBox


from address_server_XAI import *

sys.path.append(addr_BAN)

from codes.base_model import BiSU

from codes.dataset import Dictionary, VQAFeatureDataset
from data_loader import BottomUp_get_loader, VQAE_FineTunning_loader
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from address_server_XAI import *
import codes.base_model as base_model
from model_VQAE import EnsembleVQAE
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
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
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
    parser.add_argument('--vocab_vqae_path', type=str, default='data/VQA_E_vocab.pkl',
                        help='path for VQA_E vocabulary wrapper')
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
    parser.add_argument('--model_vqaE', type=str, default='model_XAI/vqaE')
    parser.add_argument('--model_vqaE_BUTD', type=str, default='model_XAI/vqaE_BUTD')
    parser.add_argument('--selfAtt', type=bool, default=False, help='self Attention?')
    parser.add_argument('--input_selfAtt', type=str, default='selfAtt-30-Final.pth')
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
    return idx

def get_answer2(p, dataloader):
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

def check_captions(caption_generator, dataloader,Dict_qid2vid, vocab,save_fig_loc,x_method_, t_method_, s_method_, args_, method, selfAtt=None):
    N = len(dataloader.dataset)
    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()
    print('t_method : {}, x_method : {}, isBUTD : {}, isUnion : {}'.format(t_method_,x_method_,args.isBUTD,args.isUnion))
    captions_list_all = []
    img_id_list=[]

    print('processing-{}-'.format(method))

    for i,(v, b, q, ans, img_ids, _) in enumerate(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        with torch.no_grad():
            v = Variable(v).cuda()
            q=q.squeeze(1)

            if (method == 'MEAN_LRCN'):
                generated_captions, logits, att, encoded_feats, _ = caption_generator.generate_caption_n_context(v, b, q, t_method='mean', x_method='weight_only',
                                                      s_method=s_method_, useVQA=True)
            elif (method == 'VQAE_LRCN'):
                if(selfAtt):
                    v=selfAtt(v)
                generated_captions, logits, att = caption_generator.generate_caption(v, b, q,
                                                                                                 t_method='mean',
                                                                                                 x_method='weight_only',
                                                                                                 s_method='BestOne',
                                                                                                 model_num=1)
            elif (method == 'MEAN_BUTD'):
                generated_captions, logits, att, encoded_feats, Vmat = caption_generator.generate_caption_n_context(v, b, q, t_method='mean', x_method='weight_only',
                                                      s_method=s_method_,
                                                      isBUTD=True, useVQA=True)
            elif (method == 'VQAE_BUTD'):
                if (selfAtt):
                    v = selfAtt(v)
                generated_captions, logits, att = caption_generator.generate_caption(v, b, q,
                                                                                                 t_method='mean',
                                                                                                 x_method='weight_only',
                                                                                                 s_method='BestOne',
                                                                                                 model_num=1,
                                                                                                 isBUTD=True)
            elif (method == 'OURS_LRCN' or method == 'OURS_LRCN_Feeding'):

                generated_captions, logits, att, encoded_feats, _ = caption_generator.generate_caption_n_context(v, b, q, t_method='uncorr', x_method='weight_only',
                                                      s_method=s_method_, useVQA=True)
            else:
                generated_captions, logits, att, encoded_feats, Vmat = caption_generator.generate_caption_n_context(
                    v, b, q, t_method='uncorr', x_method='weight_only', s_method=s_method_,
                    isBUTD=True, isUnion=True, useVQA=True)

            idx += batch_size
            bar.update(idx)
            question_list = []
            answer_list = []
            captions_list = []
            for idx2 in range(len(img_ids)):
                question_list.append(get_question(q.data[idx2], dataloader))
                if (s_method_ == 'BestOne'):
                    caption = [vocab.idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                               range(generated_captions.size(1))]
                    captions_list.append(caption)
                elif (s_method_ == 'BeamSearch'):
                    for tmp_idx in range(generated_captions.size(0)):
                        # pdb.set_trace()
                        caption = [vocab.idx2word[generated_captions[tmp_idx][w_idx].item()] for w_idx in
                                   range(generated_captions.size(1))]
                        captions_list.append(caption)
                answer_list.append(get_answer2(logits.data[idx2], dataloader))

            if (method == 'OURS_LRCN_Feeding' or method == 'OURS_BUTD_Feeding'):
                ##################################################################### EXPLAIN ##################################################################################

                word_candidate_idx_in_coco_vocab = CaptionVocabCandidateBox(question_list, answer_list, vocab)
                if (method == 'OURS_BUTD_Feeding'):
                    generated_captions = caption_generator.generate_explain(Vmat, encoded_feats,
                                                                                        word_candidate_idx_in_coco_vocab,
                                                                                        t_method='uncorr',
                                                                                        x_method='weight_only',
                                                                                        s_method=s_method_,
                                                                                        isBUTD=True,
                                                                                        isUnion=True,
                                                                                        model_num=args_.model_num)
                else:
                    generated_captions = caption_generator.generate_explain(None, encoded_feats,
                                                                                        word_candidate_idx_in_coco_vocab,
                                                                                        t_method='uncorr',
                                                                                        x_method='weight_only',
                                                                                        s_method=s_method_,
                                                                                        model_num=args_.model_num)
                captions_list = []
                for idx2 in range(len(img_ids)):
                    if (s_method_ == 'BestOne'):
                        caption = [vocab.idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                                   range(generated_captions.size(1))]
                        captions_list.append(caption)

                # print(captions_list[0])

                ##################################################################### EXPLAIN ##################################################################################
            for caption in captions_list:
                caption = caption_refine(caption, model_num=args_.model_num)
                #pdb.set_trace()
                captions_list_all.append(caption)




        img_id_list.extend(img_ids)

    bar.update(idx)
    result_json=make_json(captions_list_all, img_id_list)

<<<<<<< HEAD
    with open(args.result_json_path + '/VQAEexplains_%s_%s_results.json' \
              % (args.split, args.method), 'w') as f:
=======
    with open(args.result_json_path + '/VQAEexplain_%s_results.json' \
              % (method), 'w') as f:
>>>>>>> b76e7a47c98e4814eb59cf0f0a63b0347f983398

        json.dump(result_json, f)


def make_json(captions, ImgIds):
    utils.assert_eq(len(captions), len(ImgIds))
    results = []
    for i in range(len(captions)):
        result = {}
        result['image_id'] = ImgIds[i]
        result['ann_id'] = ImgIds[i]
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

    with open(args.vocab_vqae_path, 'rb') as f:
        vocab_VQAE = pickle.load(f)


    vqaE_loader = VQAE_FineTunning_loader('eval', dictionary_vqa, vocab, args.batch_size, shuffle=False, num_workers=0)

    # Build the models
    encoder_LRCN_mean = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    encoder_LRCN_uni = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    decoder_LRCN_mean = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder_LRCN_uni = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder_vqae_LRCN = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)

    encoder_BUTD_mean = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    encoder_BUTD_uni = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    decoder_BUTD_mean = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab),
                                       args.num_layers, paramH=args.paramH)
    decoder_BUTD_uni = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab),
                                      args.num_layers, paramH=args.paramH)
    decoder_vqae_BUTD = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab_VQAE),
                                       args.num_layers, paramH=args.paramH).to(device)

    if (args.selfAtt):
        selfAtt = BiSU(256, 2048, 1)
        selfAtt.to(device)
        print('selfAttention applied')
    else:
        selfAtt=None

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
    model_BUTD = getattr(base_model, constructor)(vqaE_loader.dataset, args.num_hid, args.op, args.gamma).cuda()

    def process(args, model, model_BUTD, eval_loader,Dict_qid2vid,vocab):

        model_hsc_path_LRCN_mean = os.path.join(
            args.hsc_path, 'mean', 'model{}_LR{}'.format(args.model_num, args.LRdim), 'model-{}.pth'.format(16))
        model_hsc_path_BUTD_mean = os.path.join(
            args.BUTD_path, 'mean', 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(20))
        model_hsc_path_LRCN_uni = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(args.model_num, args.LRdim), 'model-{}.pth'.format(16))
        model_hsc_path_BUTD_uni = os.path.join(
            args.BUTD_path, 'uncorr', 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(20))

        print('loading hsc_LRCN mean datas %s' % model_hsc_path_LRCN_mean)
        model_hsc_LRCN_data_mean = torch.load(model_hsc_path_LRCN_mean)
        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni)
        model_hsc_LRCN_data_uni = torch.load(model_hsc_path_LRCN_uni)
        print('loading hsc_BUTD mean datas %s' % model_hsc_path_BUTD_mean)
        model_hsc_BUTD_data_mean = torch.load(model_hsc_path_BUTD_mean)
        print('loading hsc_BUTD uni datas %s' % model_hsc_path_BUTD_uni)
        model_hsc_BUTD_data_uni = torch.load(model_hsc_path_BUTD_uni)

        encoder_LRCN_mean.load_state_dict(model_hsc_LRCN_data_mean['encoder_state'])
        encoder_LRCN_uni.load_state_dict(model_hsc_LRCN_data_uni['encoder_state'])
        decoder_LRCN_mean.load_state_dict(model_hsc_LRCN_data_mean['decoder_state'])
        decoder_LRCN_uni.load_state_dict(model_hsc_LRCN_data_uni['decoder_state'])

        encoder_BUTD_mean.load_state_dict(model_hsc_BUTD_data_mean['encoder_state'])
        encoder_BUTD_uni.load_state_dict(model_hsc_BUTD_data_uni['encoder_state'])
        decoder_BUTD_mean.load_state_dict(model_hsc_BUTD_data_mean['decoder_state'])
        decoder_BUTD_uni.load_state_dict(model_hsc_BUTD_data_uni['decoder_state'])

        model_vqae_path = args.model_vqaE + '/vqaE-20-Final.pth'

        model_vqae_BUTD_path = args.model_vqaE_BUTD + '/vqaE-20-Final.pth'

        print('loading %s' % model_vqae_path)
        model_vqae_LRCN_data = torch.load(model_vqae_path)

        print('loading %s' % model_vqae_BUTD_path)
        model_vqae_BUTD_data = torch.load(model_vqae_BUTD_path)

        if (args.selfAtt):
            print('loading %s' % args.input_selfAtt)
            selfAtt_data = torch.load(os.path.join('model_XAI', 'vqaE', args.input_selfAtt))
            selfAtt.load_state_dict(selfAtt_data.get('model_state', selfAtt_data))


            model_vqae_path = args.model_vqaE + '/vqaE-39-Final.pth'
            model_vqae_LRCN_data = torch.load(model_vqae_path)
            print('reloading %s' % model_vqae_path)
            selfAtt.train(False)


        model = nn.DataParallel(model).cuda()
        model_BUTD = nn.DataParallel(model_BUTD).cuda()
        ensemble_LRCN = EnsembleVQAE(model, decoder_vqae_LRCN).to(device)
        ensemble_LRCN.load_state_dict(model_vqae_LRCN_data['model_state'])

        ensemble_BUTD = EnsembleVQAE(model_BUTD, decoder_vqae_BUTD).to(device)
        ensemble_BUTD.load_state_dict(model_vqae_BUTD_data['model_state'])



        # pdb.set_trace()

        model.train(False)
        model_BUTD.train(False)
        ensemble_LRCN.train(False)
        encoder_LRCN_mean.train(False)
        encoder_LRCN_uni.train(False)
        decoder_LRCN_mean.train(False)
        decoder_LRCN_uni.train(False)

        ensemble_BUTD.train(False)
        encoder_BUTD_mean.train(False)
        encoder_BUTD_uni.train(False)
        decoder_BUTD_mean.train(False)
        decoder_BUTD_uni.train(False)

        caption_generator_LRCN_mean = BAN_HSC(model, encoder_LRCN_mean, decoder_LRCN_mean, vocab).to(device)
        caption_generator_LRCN_uni = BAN_HSC(model, encoder_LRCN_uni, decoder_LRCN_uni, vocab).to(device)
        caption_generator_BUTD_mean = BAN_HSC(model_BUTD, encoder_BUTD_mean, decoder_BUTD_mean, vocab).to(device)
        caption_generator_BUTD_uni = BAN_HSC(model_BUTD, encoder_BUTD_uni, decoder_BUTD_uni, vocab).to(device)

        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues
        methods = ['VQAE_LRCN', 'OURS_LRCN', 'MEAN_LRCN', 'OURS_LRCN_Feeding', 'VQAE_BUTD', 'OURS_BUTD', 'MEAN_BUTD', 'OURS_BUTD_Feeding']
        vocabs=[vocab_VQAE, vocab, vocab, vocab, vocab_VQAE, vocab, vocab, vocab]
        caption_generators=[ensemble_LRCN, caption_generator_LRCN_uni, caption_generator_LRCN_mean, caption_generator_LRCN_uni, ensemble_BUTD,
         caption_generator_BUTD_uni, caption_generator_BUTD_mean, caption_generator_BUTD_uni]

        for idx in range(0,len(methods)):
            check_captions(caption_generators[idx], eval_loader, Dict_qid2vid,vocabs[idx],args.save_fig_loc,args.x_method, args.t_method, args.s_method, args, methods[idx],selfAtt=selfAtt)

        ################################################################################################################





    process(args, model, model_BUTD, vqaE_loader,Dict_qid2vid,vocab)
