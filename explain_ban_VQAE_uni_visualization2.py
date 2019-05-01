
import os
import sys
import csv

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

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from model_VQAE import EnsembleVQAE
import pdb
from data_loader import VQAFeatureLoaderAdapter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='XAI')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--model_vqaE', type=str, default='model_XAI/vqaE')
    parser.add_argument('--model_vqaE_BUTD', type=str, default='model_XAI/vqaE_BUTD')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_images', type=int, default=5000) # 실제 원하는 이미지 갯수 입력
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--hsc_epoch', type=int, default=16)
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--BUTD_path', type=str, default='models_BUTD/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_threshold_0.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='Result_comparison/')
    parser.add_argument('--x_method', type=str, default='weight_only') # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='uncorr') # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne') # BestOne, BeamSearch
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--vocab_vqae_path', type=str, default='data/VQA_E_vocab.pkl',
                        help='path for VQA_E vocabulary wrapper')
    parser.add_argument('--paramH', type=int, default=256, help='dimension of lstm hidden states')
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

def get_image(q_id, Dict_qid2vid, data_root='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\test2015'): #q_id => single scalar
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

def check_captions(caption_generator_list, dataloader,Dict_qid2vid, vocab_list, save_fig_loc,x_method_, t_method_, s_method_, args_, save_folder):
    N = args_.max_images * len(save_folder)
    M = dataloader.dataset.num_ans_candidates

    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()

    csv_file = [os.path.join(save_fig_loc, save_name, 'model{}'.format(args.model_num),
                            '{}_reasonings.csv'.format(s_method_)) for save_name in save_folder]

    for v, b, q, i, num_objs in iter(dataloader):
        if(idx > N):
            break
        for i_cap in range(len(caption_generator_list)):
            bar.update(idx)
            batch_size = v.size(0)
            q = q.type(torch.LongTensor)
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                isNoAtt_=False
                num_objs=num_objs.cuda()

                if (save_folder[i_cap] == 'CAPTION_LRCN'):
                    generated_captions, logits, att, encoded_feats,_ = caption_generator_list[i_cap].generate_caption_n_context(v, b, q,t_method='mean', x_method='NoAtt', s_method=s_method_, useVQA=True)
                elif(save_folder[i_cap] == 'VQAE_LRCN'):
                    generated_captions, logits, att= caption_generator_list[i_cap].generate_caption(v, b, q,t_method='mean', x_method='weight_only', s_method='BestOne', model_num=1)
                elif(save_folder[i_cap]=='CAPTION_BUTD'):
                    generated_captions, logits, att, encoded_feats, Vmat = caption_generator_list[i_cap].generate_caption_n_context(v, b, q,t_method='mean', x_method='NoAtt', s_method=s_method_,
                                                                                                                isBUTD=True, useVQA=True)
                elif(save_folder[i_cap]=='VQAE_BUTD'):
                    generated_captions, logits, att = caption_generator_list[i_cap].generate_caption(v, b, q,
                                                                                                     t_method='mean',
                                                                                                     x_method='weight_only',
                                                                                                     s_method='BestOne',
                                                                                                     model_num=1, isBUTD=True)
                elif(save_folder[i_cap]=='OURS_LRCN' or save_folder[i_cap]=='OURS_LRCN_Feeding'):
                    generated_captions, logits, att, encoded_feats, _, unifications = caption_generator_list[
                        i_cap].generate_caption_n_context(v, b, q, t_method='uncorr', x_method='weight_only',
                                                          s_method=s_method_, useVQA=True, checkBeta=True)

                elif (save_folder[i_cap] == 'OURS_LRCN_NOATT'):
                    generated_captions, logits, att, encoded_feats, _, unifications = caption_generator_list[
                        i_cap].generate_caption_n_context(v, b, q, t_method='uncorr', x_method='NoAtt',
                                                          s_method=s_method_, useVQA=True, checkBeta=True)
                    isNoAtt_=True
                else:
                    generated_captions, logits, att, encoded_feats, Vmat = caption_generator_list[
                        i_cap].generate_caption_n_context(
                        v, b, q, t_method='uncorr', x_method='weight_only', s_method=s_method_,
                        isBUTD=True, isUnion=True, useVQA=True)

                idx += batch_size
                bar.update(idx)
                img_list=[]
                question_list=[]
                answer_list=[]
                captions_list=[]
                for idx2 in range(len(i)):
                    img_list.append(get_image(i[idx2].item(),Dict_qid2vid))
                    question_list.append(get_question(q.data[idx2],dataloader))
                    if (s_method_ == 'BestOne'):
                        caption = [vocab_list[i_cap].idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                                   range(generated_captions.size(1))]
                        captions_list.append(caption)
                    elif (s_method_ == 'BeamSearch'):
                        for tmp_idx in range(generated_captions.size(0)):
                            #pdb.set_trace()
                            caption = [vocab_list[i_cap].idx2word[generated_captions[tmp_idx][w_idx].item()] for w_idx in
                                       range(generated_captions.size(1))]
                            captions_list.append(caption)
                    answer_list.append(get_answer(logits.data[idx2], dataloader))

                if(save_folder[i_cap]=='OURS_LRCN_Feeding' or save_folder[i_cap]=='OURS_BUTD_Feeding'):
                    ##################################################################### EXPLAIN ##################################################################################
                    word_candidate_idx_in_coco_vocab = CaptionVocabCandidate(question_list[0], answer_list[0], vocab)
                    if(save_folder[i_cap]=='OURS_BUTD_Feeding'):
                        generated_captions = caption_generator_list[i_cap].generate_explain(Vmat, encoded_feats,
                                                                                word_candidate_idx_in_coco_vocab,
                                                                                t_method='uncorr',
                                                                                x_method='weight_only',
                                                                                s_method=s_method_,
                                                                                isBUTD=True,
                                                                                isUnion=True,
                                                                                model_num=args_.model_num)
                    else:
                        generated_captions = caption_generator_list[i_cap].generate_explain(None, encoded_feats,
                                                                                word_candidate_idx_in_coco_vocab,
                                                                                t_method='uncorr', x_method='weight_only',
                                                                                s_method=s_method_,
                                                                                model_num=args_.model_num)
                    captions_list = []
                    for idx2 in range(len(i)):
                        if (s_method_ == 'BestOne'):
                            caption = [vocab.idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                                       range(generated_captions.size(1))]
                            captions_list.append(caption)


                    # print(captions_list[0])

                    ##################################################################### EXPLAIN ##################################################################################

            tmp_fig = showAttention(question_list[0], img_list[0], answer_list[0], att[0, :, :, :], unifications,
                                    b[0, :, :4],
                                    captions_list[0], None, display=False, isNoAtt=isNoAtt_)
            plt.savefig(os.path.join(save_fig_loc, save_folder[i_cap], 'model{}'.format(args.model_num), 'imgs',
                                     '{}_att_uni_colormap.png'.format(i.item())))
            plt.close(tmp_fig)

            caption_=captions_list[0]






    bar.update(idx)


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

#######################################################################################

def showAttention(input_question, image, output_answer,attentions, unification,bbox, explains, RelScore, display=True, NumBeams=1, isNoAtt=False):
    fig = plt.figure(figsize=(36, 6))

    x_caption = ''
    for num_sen in range(NumBeams):
        if (NumBeams > 1):
            explain = explains[num_sen]
        else:
            explain = explains
        for word_ in explain:
            if word_ == '<start>':
                x_caption = ''
            elif word_ == '<end>':
                break
            else:
                x_caption = x_caption + ' ' + word_

        if (NumBeams > 1):
            x_caption = x_caption + '\n'
    Explain = 'Explain : {}'.format(x_caption)
    unification = unification.cpu()

    unification = unification.squeeze()

    _, idx_uni = torch.sort(unification, descending=True)

    idx_uni = idx_uni[:3]

    attentions = attentions.cpu()

    att2 = attentions[1, :, :]
    att2 = torch.t(att2)

    # pdb.set_trace()

    att2_sum = torch.sum(att2, 0)

    _, idx_att2 = torch.sort(att2_sum, descending=True)

    idx_att2 = idx_att2[:3]

    im_height = image.shape[0]
    im_width = image.shape[1]
    lis = range(attentions.size(1))

    bbox_uni = bbox[idx_uni, :]

    mask_1 = np.zeros(image.shape[0:2])
    mask_2 = np.zeros(image.shape[0:2])
    mask_ = np.zeros(image.shape[0:2])
    num_objects = att2_sum.size(0)

    factor_1 = 1
    factor_2 = unification

    int_pt = torch.IntTensor(4)
    for i in range(num_objects):
        pt = bbox[i, :].clone()
        pt[0] = pt[0] * im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        for ii in range(4):
            int_pt[ii] = round(pt[ii].item())
        mask_1[int_pt[1]:int_pt[3], int_pt[0]:int_pt[2]] += factor_1
        mask_2[int_pt[1]:int_pt[3], int_pt[0]:int_pt[2]] += factor_2[i]
        mask_[int_pt[1]:int_pt[3], int_pt[0]:int_pt[2]] += att2_sum[i]

    if(isNoAtt):
        mask_=np.ones(image.shape[0:2])

    gs1 = gridspec.GridSpec(1, 3)

    plt.rcParams.update({'font.size': 22})

    im1 = fig.add_subplot(131)
    im1.imshow(image)
    #
    # att_ax = fig.add_subplot(233)
    # cax2 = att_ax.matshow(mask_, cmap='afmhot')
    # fig.colorbar(cax2)

    uni_max=mask_1.max()
    im1.axes.get_xaxis().set_visible(False)
    im1.axes.get_yaxis().set_visible(False)

    uni_ax = fig.add_subplot(132)
    mask_1=np.multiply(mask_1, mask_)
    cax3 = uni_ax.matshow(mask_1, cmap='coolwarm')
    fig.colorbar(cax3)
    uni_ax.axes.get_xaxis().set_visible(False)
    uni_ax.axes.get_yaxis().set_visible(False)



    uni_ax3 = fig.add_subplot(133)
    mask_2 = np.multiply(mask_2, mask_)
    cax5 = uni_ax3.matshow(mask_2, cmap='coolwarm')
    fig.colorbar(cax5)

    uni_ax3.axes.get_xaxis().set_visible(False)
    uni_ax3.axes.get_yaxis().set_visible(False)

    plt.tight_layout()


    if display:
        plt.show()


    return fig

#######################################################################################`

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    name = 'test2015'

    question_path = os.path.join(
        'codes\\tools\\data', 'v2_OpenEnded_mscoco_%s_questions.json' % \
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])



    Dict_qid2vid = {}
    Dict_qid2ques={}
    for i in range(len(questions)):
        #pdb.set_trace()
        Dict_qid2vid[questions[i]['question_id']]=questions[i]['image_id']
    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Visualization through eval_dataset

    eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()

    eval_loader = VQAFeatureLoaderAdapter(eval_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.vocab_vqae_path, 'rb') as f:
        vocab_VQAE = pickle.load(f)

    # Build the models
    encoder_LRCN_mean = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    encoder_LRCN_uni = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    decoder_LRCN_mean = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder_LRCN_uni = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder_vqae_LRCN = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)

    encoder_BUTD_mean = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    encoder_BUTD_uni = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(
        device)
    decoder_BUTD_mean = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers, paramH=args.paramH)
    decoder_BUTD_uni = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab),
                                  args.num_layers, paramH=args.paramH)
    decoder_vqae_BUTD = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab_VQAE), args.num_layers, paramH=args.paramH).to(device)


    def process(args, model, eval_loader,Dict_qid2vid):




        model_hsc_path_LRCN_mean = os.path.join(
            args.hsc_path, 'mean', 'model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}.pth'.format(16))
        model_hsc_path_BUTD_mean = os.path.join(
            args.BUTD_path, 'mean', 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(20))
        model_hsc_path_LRCN_uni = os.path.join(
            args.hsc_path,'uncorr','model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}.pth'.format(16))
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

        model = nn.DataParallel(model).cuda()
        ensemble_LRCN = EnsembleVQAE(model, decoder_vqae_LRCN).to(device)
        ensemble_LRCN.load_state_dict(model_vqae_LRCN_data['model_state'])

        ensemble_BUTD = EnsembleVQAE(model, decoder_vqae_BUTD).to(device)
        ensemble_BUTD.load_state_dict(model_vqae_BUTD_data['model_state'])

        #pdb.set_trace()

        model.train(False)
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

        caption_generator_LRCN_mean=BAN_HSC(model,encoder_LRCN_mean,decoder_LRCN_mean,vocab).to(device)
        caption_generator_LRCN_uni = BAN_HSC(model, encoder_LRCN_uni, decoder_LRCN_uni, vocab).to(device)
        caption_generator_BUTD_mean = BAN_HSC(model, encoder_BUTD_mean, decoder_BUTD_mean, vocab).to(device)
        caption_generator_BUTD_uni = BAN_HSC(model, encoder_BUTD_uni, decoder_BUTD_uni, vocab).to(device)


        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues
        #save_folders=['VQAE_LRCN', 'OURS_LRCN', 'OURS_LRCN_Feeding', 'CAPTION_LRCN', 'VQAE_BUTD', 'OURS_BUTD', 'OURS_BUTD_Feeding', 'CAPTION_BUTD']
        save_folders = ['OURS_LRCN','OURS_LRCN_NOATT']
        if not os.path.exists(os.path.join(args.save_fig_loc,'VQAE_LRCN','model{}'.format(args.model_num),'imgs')): # # use ensemble_, vocab_VQAE
            os.makedirs(os.path.join(args.save_fig_loc,'VQAE_LRCN','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS_LRCN','model{}'.format(args.model_num),'imgs')): # x_method == weight_only
            os.makedirs(os.path.join(args.save_fig_loc,'OURS_LRCN','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS_LRCN_Feeding','model{}'.format(args.model_num),'imgs')): # x_method == weight_only
            os.makedirs(os.path.join(args.save_fig_loc,'OURS_LRCN_Feeding','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'CAPTION_LRCN','model{}'.format(args.model_num),'imgs')): # x_method == NoAtt
            os.makedirs(os.path.join(args.save_fig_loc,'CAPTION_LRCN','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'VQAE_BUTD','model{}'.format(args.model_num),'imgs')): # # use ensemble_, vocab_VQAE
            os.makedirs(os.path.join(args.save_fig_loc,'VQAE_BUTD','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS_BUTD','model{}'.format(args.model_num),'imgs')): # x_method == weight_only
            os.makedirs(os.path.join(args.save_fig_loc,'OURS_BUTD','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS_BUTD_Feeding','model{}'.format(args.model_num),'imgs')): # x_method == weight_only
            os.makedirs(os.path.join(args.save_fig_loc,'OURS_BUTD_Feeding','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'CAPTION_BUTD','model{}'.format(args.model_num),'imgs')): # x_method == NoAtt
            os.makedirs(os.path.join(args.save_fig_loc,'CAPTION_BUTD','model{}'.format(args.model_num),'imgs'))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS_LRCN_NOATT','model{}'.format(args.model_num),'imgs')): # x_method == NoAtt
            os.makedirs(os.path.join(args.save_fig_loc,'OURS_LRCN_NOATT','model{}'.format(args.model_num),'imgs'))
        #check_captions([caption_generator, ensemble_], eval_loader, Dict_qid2vid, [vocab, vocab_VQAE],
         #              args.save_fig_loc, args.x_method, args.t_method, args.s_method, args)
        # check_captions([ensemble_LRCN, caption_generator_LRCN_uni, caption_generator_LRCN_uni, caption_generator_LRCN_mean, ensemble_BUTD, caption_generator_BUTD_uni, caption_generator_BUTD_uni, caption_generator_BUTD_mean],
        #                eval_loader, Dict_qid2vid,
        #                [vocab_VQAE, vocab, vocab, vocab, vocab_VQAE, vocab, vocab, vocab],
        #                args.save_fig_loc, args.x_method, args.t_method, args.s_method, args, save_folders)
        check_captions(
            [caption_generator_LRCN_uni,caption_generator_LRCN_uni],
            eval_loader, Dict_qid2vid,
            [vocab,vocab],
            args.save_fig_loc, args.x_method, args.t_method, args.s_method, args, save_folders)
        ################################################################################################################





    process(args, model, eval_loader,Dict_qid2vid)