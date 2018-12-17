
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
from model_explain import Relev_Check

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC

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
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--hsc_epoch', type=int, default=13)
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='saved_figs_with_caption/')
    parser.add_argument('--x_method', type=str, default='weight_only') # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='uncorr') # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne') # BestOne, BeamSearch
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
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

def check_captions(caption_generator, dataloader,Dict_qid2vid, vocab,save_fig_loc,x_method_, t_method_, s_method_):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()
    for v, b, q, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        q = q.type(torch.LongTensor)
        with torch.no_grad():
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()

            generated_captions, logits, att = caption_generator.generate_caption(v, b, q,t_method=t_method_, x_method=x_method_, s_method=s_method_)

            idx += batch_size
            img_list=[]
            question_list=[]
            answer_list=[]
            captions_list=[]
            for idx2 in range(len(i)):
                img_list.append(get_image(i[idx2].item(),Dict_qid2vid))
                question_list.append(get_question(q.data[idx2],dataloader))
                if (s_method_ == 'BestOne'):
                    caption = [vocab.idx2word[generated_captions[idx2][w_idx].item()] for w_idx in
                               range(generated_captions.size(1))]
                    captions_list.append(caption)
                elif (s_method_ == 'BeamSearch'):
                    for tmp_idx in range(generated_captions.size(0)):
                        #pdb.set_trace()
                        caption = [vocab.idx2word[generated_captions[tmp_idx][w_idx].item()] for w_idx in
                                   range(generated_captions.size(1))]
                        captions_list.append(caption)

                answer_list.append(get_answer(logits.data[idx2], dataloader))

        caption_=captions_list[0]
        RelScore=Relev_Check(captions_list[0], q, answer_list[0], caption_generator.BAN.module.w_emb, dataloader.dataset.dictionary)

        if RelScore is None:
            continue

        if (s_method_ == 'BestOne'):
            tmp_fig=showAttention(question_list[0],img_list[0],answer_list[0],att[0,:,:,:],b[0,:,:4], captions_list[0], RelScore,display=False)
        elif (s_method_ == 'BeamSearch'):
            tmp_fig = showAttention(question_list[0], img_list[0], answer_list[0], att[0, :, :, :], b[0, :, :4],
                                    captions_list[:len(generated_captions)], RelScore, display=False, NumBeams=len(generated_captions))
        plt.savefig(os.path.join(save_fig_loc,'model{}'.format(args.model_num),t_method_, x_method_, s_method_, '{}.png'.format(i.item())))
        plt.close(tmp_fig)

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
    fig.text(0.2, 0.95, input_question, ha='center', va='center', fontsize=20)

    Answer='Answer : {}'.format(output_answer)
    fig.text(0.2, 0.90, Answer, ha='center', va='center',fontsize=20)

    Answer = 'RelScore : {}'.format(round(RelScore,4))
    fig.text(0.8, 0.95, Answer, ha='center', va='center', fontsize=20)


    x_caption = ''
    for num_sen in range(NumBeams):
        if(NumBeams > 1):
            explain=explains[num_sen]
        else:
            explain=explains
        if explain[0] == '<start>':
            for i in explain[1:]:
                if i != '<end>':
                    x_caption=x_caption+' '+i
                else:
                    break
        if (NumBeams > 1):
            x_caption=x_caption+'\n'
    Explain = 'Explain : {}'.format(x_caption)
    fig.text(0.5, 0.90, Explain, ha='center', va='center',fontsize=20)
    attentions=attentions.cpu()



    att1 = attentions[0, :, :]
    att2 = attentions[1, :, :]
    #pdb.set_trace()
    att1=torch.t(att1)
    att2=torch.t(att2)

    #pdb.set_trace()

    att1_sum=torch.sum(att1,0)
    att2_sum = torch.sum(att2, 0)

    _,idx_att1=torch.sort(att1_sum,descending=True)
    _, idx_att2 = torch.sort(att2_sum, descending=True)

    idx_att1=idx_att1[:3]
    idx_att2 = idx_att2[:3]

    bbox_att1=bbox[idx_att1,:]
    bbox_att2 = bbox[idx_att2, :]

    im1=fig.add_subplot(221)
    im1.imshow(image)
    im_height=image.shape[0]
    im_width = image.shape[1]
    color_zero=[0,0,0]
    for i in range(3):
        pt=bbox_att1[i,:]
        pt[0]=pt[0]*im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color=color_zero.copy()
        color[i]=1
        im1.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        im1.text(pt[0], pt[1], str(idx_att1[i].item()+1), bbox={'facecolor':color, 'alpha':0.5})


    ax = fig.add_subplot(222)

    cax = ax.matshow(att1.numpy(), cmap='bone')
    fig.colorbar(cax)

    lis = range(attentions.size(1))
    x_axis = ["{:02d}".format(x) for x in lis]

    # Set up axes
    ax.set_xticklabels(x_axis, rotation=90)
    ax.set_yticklabels([''] + input_question.split(' ') +
                       ['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    im2 = fig.add_subplot(223)
    im2.imshow(image)

    for i in range(3):
        pt=bbox_att2[i,:]
        pt[0]=pt[0]*im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color=color_zero.copy()
        color[i]=1
        im2.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        im2.text(pt[0], pt[1], str(idx_att2[i].item()+1), bbox={'facecolor':color, 'alpha':0.5})

    ax2 = fig.add_subplot(224)

    cax2 = ax2.matshow(att2.numpy(), cmap='bone')
    fig.colorbar(cax2)

    # Set up axes
    ax2.set_xticklabels(x_axis, rotation=90)
    second_yaxis= ["f{:02d}".format(y) for y in range(len(input_question))]

    ax2.set_yticklabels(second_yaxis)

    # Show label at every tick
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
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

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    encoder = Encoder_HieStackedCorr(args.embed_size, 2048,LRdim=args.LRdim).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    def process(args, model, eval_loader,Dict_qid2vid):
        model_path = args.input + '/model%s.pth' % \
                     ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        if(args.t_method == 'mean'):
            model_hsc_path = os.path.join(
                args.hsc_path, args.t_method, 'model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}-400.pth'.format(args.hsc_epoch))
        else:
            model_hsc_path = os.path.join(
                args.hsc_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim), 'model-{}-400.pth'.format(args.hsc_epoch))

        print('loading hsc datas %s' % model_hsc_path)
        model_hsc_data=torch.load(model_hsc_path)

        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        #pdb.set_trace()

        model.train(False)
        encoder.train(False)
        decoder.train(False)

        caption_generator=BAN_HSC(model,encoder,decoder)

        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues

        if not os.path.exists(os.path.join(args.save_fig_loc,'model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method)):
            os.makedirs(os.path.join(args.save_fig_loc,'model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method))
        check_captions(caption_generator, eval_loader, Dict_qid2vid,vocab,args.save_fig_loc,args.x_method, args.t_method, args.s_method)

        ################################################################################################################





    process(args, model, eval_loader,Dict_qid2vid)
