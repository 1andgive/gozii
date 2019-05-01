
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
from model_explain import Relev_Check, CaptionVocabCandidate, Relev_Check_by_IDX

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from model_VQAE import EnsembleVQAE
from nltk.tokenize import word_tokenize
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
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--model_vqaE', type=str, default='model_XAI/vqaE')
    parser.add_argument('--model_vqaE_BUTD', type=str, default='model_XAI/vqaE_BUTD')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
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
    parser.add_argument('--vocab_path', type=str, default='data/vocab_standard.pkl', help='path for vocabulary wrapper')
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
    _m, idx = torch.max(p,1)
    label_from_vqa=torch.cuda.LongTensor(p.size()).fill_(0)
    for i in range(p.size(0)):
        label_from_vqa[i,idx[i]]=1
    return idx

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

def check_captions(caption_generator_list, dataloader,Dict_qid2vid, vocab_list, model_list, save_fig_loc,x_method_, t_method_, s_method_, args_, save_folder, Dict_AC_2_Q, Dict_AC_2_Q_VQAE):
    N = args_.max_images * len(save_folder)
    M = dataloader.dataset.num_ans_candidates

    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()


    rel_score_list=[0 for x in range(len(caption_generator_list))]
    rel_idx=0
    temp=rel_score_list.copy()

    DICT_RELEV=Dict_AC_2_Q

    for v, b, q, i in iter(dataloader):
        batch_size = v.size(0)
        if(rel_idx > 0):
            temp='['
            for i_string in range(len(caption_generator_list)):
                temp+=' {}:{:.4} ||'.format(save_folder[i_string],rel_score_list[i_string]/rel_idx)
            temp+=']'
            print(temp)
        rel_idx += batch_size
        for i_cap in range(len(caption_generator_list)):
            bar.update(idx)
            model_num_=model_list[i_cap]
            q = q.type(torch.LongTensor)
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()

                num_objs = torch.sum(v, 2)
                num_objs = torch.sum(num_objs != 0.0, 1)
                num_objs = num_objs.float()

                generated_captions, logits, att, encoded_feats, _ = caption_generator_list[
                    i_cap].generate_caption_n_context(v, b, q, t_method='uncorr', x_method='weight_only',
                                                      s_method=s_method_, model_num=model_num_, useVQA=True, obj_nums=num_objs)
                DICT_RELEV = Dict_AC_2_Q

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
                answer_list.append(get_answer(logits.data, dataloader))


            RelScoreTensor=Relev_Check_by_IDX(generated_captions, q, answer_list[0], caption_generator_list[i_cap].BAN.module.w_emb, DICT_RELEV)

            rel_score_list[i_cap] += RelScoreTensor.sum(0)



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

def showAttention(input_question, image, output_answer, attentions,bbox, explains, RelScore, display=True, NumBeams=1):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(25,19))
    #pdb.set_trace()
    fig.text(0.35, 0.95, input_question, ha='center', va='center', fontsize=30)

    Answer='Answer : {}'.format(output_answer)
    fig.text(0.75, 0.95, Answer, ha='center', va='center',fontsize=30)



    x_caption = ''
    for num_sen in range(NumBeams):
        if(NumBeams > 1):
            explain=explains[num_sen]
        else:
            explain=explains
        for word_ in explain:
            if word_ == '<start>':
                x_caption = ''
            elif word_ == '<end>':
                break
            else:
                x_caption = x_caption + ' ' + word_

        if (NumBeams > 1):
            x_caption=x_caption+'\n'
    Explain = 'Explain : {}'.format(x_caption)
    fig.text(0.5, 0.90, Explain, ha='center', va='center',fontsize=30)
    attentions=attentions.cpu()



    att2 = attentions[1, :, :]
    att2=torch.t(att2)

    #pdb.set_trace()

    att2_sum = torch.sum(att2, 0)

    _, idx_att2 = torch.sort(att2_sum, descending=True)

    idx_att2 = idx_att2[:3]

    bbox_att2 = bbox[idx_att2, :]

    im_height = image.shape[0]
    im_width = image.shape[1]
    color_zero = [0, 0, 0]
    lis = range(attentions.size(1))
    x_axis = ["{:02d}".format(x) for x in lis]

    #
    #
    # im2 = fig.add_subplot(121)
    # im2.imshow(image)
    #
    # for i in range(3):
    #     pt=bbox_att2[i,:].clone()
    #     pt[0]=pt[0]*im_width
    #     pt[1] = pt[1] * im_height
    #     pt[2] = pt[2] * im_width
    #     pt[3] = pt[3] * im_height
    #     coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
    #     color=color_zero.copy()
    #     color[i]=1
    #     im2.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #     im2.text(pt[0], pt[1], str(idx_att2[i].item()+1), bbox={'facecolor':color, 'alpha':0.5})


    im3=fig.add_subplot(111)

    mask_=np.zeros(image.shape[0:2])
    num_objects=att2_sum.size(0)
    int_pt=torch.IntTensor(4)
    for i in range(num_objects):
        pt=bbox[i,:].clone()
        pt[0] = pt[0] * im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        for ii in range(4):
            int_pt[ii]=round(pt[ii].item())
        mask_[int_pt[1]:int_pt[3],int_pt[0]:int_pt[2]] += att2_sum[i]

    img_total = np.zeros([im_height, 2 * im_width, image.shape[2]])
    img_total=img_total.astype(int)
    img_total[:, :im_width, :] =image.copy()

    image=image.copy()
    img_r=np.multiply(image[:,:,0],mask_)
    img_g = np.multiply(image[:, :, 1], mask_)
    img_b = np.multiply(image[:, :, 2], mask_)
    image[:,:,0]=np.round_(img_r)
    image[:, :, 1] = np.round_(img_g)
    image[:, :, 2] = np.round_(img_b)
    image=image.astype(int)
    img_total[:, im_width:, :]=image



    im3.imshow(image)


    if display:
        plt.show()


    return fig, img_total

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

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.vocab_vqae_path, 'rb') as f:
        vocab_VQAE = pickle.load(f)


    Dict_IdxA_2_IdxQ = {}
    for Wa_set in eval_dset.ans2label.keys():
        for Wa in word_tokenize(Wa_set):
            if Wa in eval_dset.dictionary.word2idx.keys():
                if eval_dset.ans2label[Wa_set] in Dict_IdxA_2_IdxQ.keys():
                    Dict_IdxA_2_IdxQ[eval_dset.ans2label[Wa_set]].append(eval_dset.dictionary.word2idx[Wa])
                else:
                    Dict_IdxA_2_IdxQ[eval_dset.ans2label[Wa_set]] = [eval_dset.dictionary.word2idx[Wa]]

    Dict_IdxC_2_IdxQ = {}
    for Wc in vocab.word2idx.keys():
        if Wc in eval_dset.dictionary.word2idx.keys():
            Dict_IdxC_2_IdxQ[vocab.word2idx[Wc]] = eval_dset.dictionary.word2idx[Wc]

    Dict_IdxC_2_IdxQ_VQAE = {}
    for Wc in vocab_VQAE.word2idx.keys():
        if Wc in eval_dset.dictionary.word2idx.keys():
            Dict_IdxC_2_IdxQ_VQAE[vocab_VQAE.word2idx[Wc]] = eval_dset.dictionary.word2idx[Wc]

    Dict_AC_2_Q = (Dict_IdxA_2_IdxQ, Dict_IdxC_2_IdxQ)
    Dict_AC_2_Q_VQAE = (Dict_IdxA_2_IdxQ, Dict_IdxC_2_IdxQ_VQAE)



    # Build the models

    encoder_LRCN_uni1 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=1, LRdim=32).to(
        device)
    decoder_LRCN_uni1 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni2 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=1, LRdim=128).to(
        device)
    decoder_LRCN_uni2 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)


    encoder_LRCN_uni4 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=3, LRdim=64).to(
        device)
    decoder_LRCN_uni4 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni5 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=4, LRdim=64).to(
        device)
    decoder_LRCN_uni5 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni6 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=5, LRdim=64).to(
        device)
    decoder_LRCN_uni6 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni7 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=6, LRdim=64).to(
        device)
    decoder_LRCN_uni7 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni8 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=7, LRdim=64).to(
        device)
    decoder_LRCN_uni8 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    encoder_LRCN_uni0 = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=1, LRdim=64).to(
        device)
    decoder_LRCN_uni0 = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)


    def process(args, model, eval_loader,Dict_qid2vid, Dict_AC_2_Q, Dict_AC_2_Q_VQAE):





        model_hsc_path_LRCN_uni1 = os.path.join(
            args.hsc_path,'uncorr','model{}_LR{}'.format(1,32), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni2 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(1, 128), 'model-{}.pth'.format(16))


        model_hsc_path_LRCN_uni4 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(3, 64), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni5 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(4, 64), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni6 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(5, 64), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni7 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(6, 64), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni8 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(7, 64), 'model-{}.pth'.format(16))

        model_hsc_path_LRCN_uni0 = os.path.join(
            args.hsc_path, 'uncorr', 'model{}_LR{}'.format(1, 64), 'model-{}-small_vocab.pth'.format(16))


        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni1)
        model_hsc_LRCN_data_uni1 = torch.load(model_hsc_path_LRCN_uni1)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni2)
        model_hsc_LRCN_data_uni2 = torch.load(model_hsc_path_LRCN_uni2)


        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni4)
        model_hsc_LRCN_data_uni4 = torch.load(model_hsc_path_LRCN_uni4)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni5)
        model_hsc_LRCN_data_uni5 = torch.load(model_hsc_path_LRCN_uni5)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni6)
        model_hsc_LRCN_data_uni6 = torch.load(model_hsc_path_LRCN_uni6)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni7)
        model_hsc_LRCN_data_uni7 = torch.load(model_hsc_path_LRCN_uni7)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni8)
        model_hsc_LRCN_data_uni8 = torch.load(model_hsc_path_LRCN_uni8)

        print('loading hsc_LRCN uni datas %s' % model_hsc_path_LRCN_uni0)
        model_hsc_LRCN_data_uni0 = torch.load(model_hsc_path_LRCN_uni0)

        encoder_LRCN_uni1.load_state_dict(model_hsc_LRCN_data_uni1['encoder_state'])
        decoder_LRCN_uni1.load_state_dict(model_hsc_LRCN_data_uni1['decoder_state'])

        encoder_LRCN_uni2.load_state_dict(model_hsc_LRCN_data_uni2['encoder_state'])
        decoder_LRCN_uni2.load_state_dict(model_hsc_LRCN_data_uni2['decoder_state'])

        encoder_LRCN_uni4.load_state_dict(model_hsc_LRCN_data_uni4['encoder_state'])
        decoder_LRCN_uni4.load_state_dict(model_hsc_LRCN_data_uni4['decoder_state'])

        encoder_LRCN_uni5.load_state_dict(model_hsc_LRCN_data_uni5['encoder_state'])
        decoder_LRCN_uni5.load_state_dict(model_hsc_LRCN_data_uni5['decoder_state'])

        encoder_LRCN_uni6.load_state_dict(model_hsc_LRCN_data_uni6['encoder_state'])
        decoder_LRCN_uni6.load_state_dict(model_hsc_LRCN_data_uni6['decoder_state'])

        encoder_LRCN_uni7.load_state_dict(model_hsc_LRCN_data_uni7['encoder_state'])
        decoder_LRCN_uni7.load_state_dict(model_hsc_LRCN_data_uni7['decoder_state'])

        encoder_LRCN_uni8.load_state_dict(model_hsc_LRCN_data_uni8['encoder_state'])
        decoder_LRCN_uni8.load_state_dict(model_hsc_LRCN_data_uni8['decoder_state'])

        encoder_LRCN_uni0.load_state_dict(model_hsc_LRCN_data_uni0['encoder_state'])
        decoder_LRCN_uni0.load_state_dict(model_hsc_LRCN_data_uni0['decoder_state'])


        model_vqae_path = args.model_vqaE + '/vqaE-20-Final.pth'

        print('loading %s' % model_vqae_path)
        model_vqae_LRCN_data = torch.load(model_vqae_path)

        model = nn.DataParallel(model).cuda()


        #pdb.set_trace()

        model.train(False)
        encoder_LRCN_uni1.train(False)
        decoder_LRCN_uni1.train(False)

        encoder_LRCN_uni2.train(False)
        decoder_LRCN_uni2.train(False)


        encoder_LRCN_uni4.train(False)
        decoder_LRCN_uni4.train(False)

        encoder_LRCN_uni5.train(False)
        decoder_LRCN_uni5.train(False)

        encoder_LRCN_uni6.train(False)
        decoder_LRCN_uni6.train(False)

        encoder_LRCN_uni7.train(False)
        decoder_LRCN_uni7.train(False)

        encoder_LRCN_uni8.train(False)
        decoder_LRCN_uni8.train(False)

        encoder_LRCN_uni0.train(False)
        decoder_LRCN_uni0.train(False)


        caption_generator_LRCN_uni1 = BAN_HSC(model, encoder_LRCN_uni1, decoder_LRCN_uni1, vocab).to(device)
        caption_generator_LRCN_uni2 = BAN_HSC(model, encoder_LRCN_uni2, decoder_LRCN_uni2, vocab).to(device)
        caption_generator_LRCN_uni4 = BAN_HSC(model, encoder_LRCN_uni4, decoder_LRCN_uni4, vocab).to(device)
        caption_generator_LRCN_uni5 = BAN_HSC(model, encoder_LRCN_uni5, decoder_LRCN_uni5, vocab).to(device)
        caption_generator_LRCN_uni6 = BAN_HSC(model, encoder_LRCN_uni6, decoder_LRCN_uni6, vocab).to(device)
        caption_generator_LRCN_uni7 = BAN_HSC(model, encoder_LRCN_uni7, decoder_LRCN_uni7, vocab).to(device)
        caption_generator_LRCN_uni8 = BAN_HSC(model, encoder_LRCN_uni8, decoder_LRCN_uni8, vocab).to(device)
        caption_generator_LRCN_uni0 = BAN_HSC(model, encoder_LRCN_uni0, decoder_LRCN_uni0, vocab).to(device)



        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues
        # save_folders=['Relu_32', 'Relu_128', 'Tanh_64', 'LRelu_64', 'MathUni', 'DetailUni']
        save_folders=['Relu_64']
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
        #check_captions([caption_generator, ensemble_], eval_loader, Dict_qid2vid, [vocab, vocab_VQAE],
         #              args.save_fig_loc, args.x_method, args.t_method, args.s_method, args)
        # check_captions([caption_generator_LRCN_uni1, caption_generator_LRCN_uni2, caption_generator_LRCN_uni4,
        #                 caption_generator_LRCN_uni5, caption_generator_LRCN_uni6, caption_generator_LRCN_uni7],
        #                eval_loader, Dict_qid2vid,
        #                [vocab, vocab, vocab, vocab, vocab, vocab],
        #                [1,1,3,4,5,6],
        #                args.save_fig_loc, args.x_method, args.t_method, args.s_method, args, save_folders, Dict_AC_2_Q, Dict_AC_2_Q_VQAE)

        check_captions([caption_generator_LRCN_uni0],
                       eval_loader, Dict_qid2vid,
                       [vocab],
                       [1],
                       args.save_fig_loc, args.x_method, args.t_method, args.s_method, args, save_folders, Dict_AC_2_Q,
                       Dict_AC_2_Q_VQAE)

        ################################################################################################################





    process(args, model, eval_loader,Dict_qid2vid,Dict_AC_2_Q,Dict_AC_2_Q_VQAE)