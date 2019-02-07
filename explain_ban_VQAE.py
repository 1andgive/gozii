
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

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC
from model_VQAE import EnsembleVQAE
import pdb
from colors import rgb2ycrcb, ycrcb2rgb, rgb2hls, hls2rgb


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
    parser.add_argument('--input', type=str, default='model_XAI/vqaE')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_images', type=int, default=10000) # 실제 원하는 이미지 갯수 x 2 입력
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--hsc_epoch', type=int, default=13)
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='Result_comparison/')
    parser.add_argument('--x_method', type=str, default='weight_only') # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='uncorr') # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne') # BestOne, BeamSearch
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--vocab_vqae_path', type=str, default='data/VQA_E_vocab.pkl',
                        help='path for VQA_E vocabulary wrapper')
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

def check_captions(caption_generator_list, dataloader,Dict_qid2vid, vocab_list, save_fig_loc,x_method_, t_method_, s_method_, args_):
    N = args_.max_images
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()
    #save_folder=['OURS', 'VQAE']
    # save_folder = ['CAPTION']
    save_folder = ['OURS']
    for v, b, q, i in iter(dataloader):
        if(idx > args_.max_images):
            break
        for i_cap in range(len(caption_generator_list)):
            bar.update(idx)
            batch_size = v.size(0)
            q = q.type(torch.LongTensor)
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()


                generated_captions, logits, att, encoded_feats = caption_generator_list[i_cap].generate_caption_n_context(v, b, q,t_method=t_method_, x_method=args.x_method, s_method=s_method_)

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

                if(save_folder[i_cap]=='OURS'):
                    ##################################################################### EXPLAIN ##################################################################################
                    word_candidate_idx_in_coco_vocab = CaptionVocabCandidate(question_list[0], answer_list[0], vocab)
                    generated_captions = caption_generator_list[i_cap].generate_explain(encoded_feats,
                                                                            word_candidate_idx_in_coco_vocab,
                                                                            t_method=t_method_, x_method=args.x_method,
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

            caption_=captions_list[0]
            RelScore=Relev_Check(captions_list[0], q, answer_list[0], caption_generator_list[i_cap].BAN.module.w_emb, dataloader.dataset.dictionary)

            if RelScore is None:
                continue

            if (s_method_ == 'BestOne'):
                tmp_fig, img_=showAttention(question_list[0],img_list[0],answer_list[0],att[0,:,:,:],b[0,:,:4], captions_list[0], RelScore,display=False)
            elif (s_method_ == 'BeamSearch'):
                tmp_fig, img_ = showAttention(question_list[0], img_list[0], answer_list[0], att[0, :, :, :], b[0, :, :4],
                                        captions_list[:len(generated_captions)], RelScore, display=False, NumBeams=len(generated_captions))
            plt.savefig(os.path.join(save_fig_loc,save_folder[i_cap],'model{}'.format(args.model_num),t_method_, x_method_, s_method_, '{}_figure.png'.format(i.item())))
            plt.imsave(os.path.join(save_fig_loc,save_folder[i_cap],'model{}'.format(args.model_num),t_method_, x_method_, s_method_, '{}_image.png'.format(i.item())), img_)
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

    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder_vqae = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)



    def process(args, model, eval_loader,Dict_qid2vid):
        model_vqae_path = args.input + '/vqaE-%s-Final.pth' % \
                     ('' if 0 > args.epoch else '%d' % args.epoch)

        print('loading %s' % model_vqae_path)
        model_vqae_data = torch.load(model_vqae_path)



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
        ensemble_ = EnsembleVQAE(model, decoder_vqae).to(device)
        ensemble_.load_state_dict(model_vqae_data['model_state'])

        #pdb.set_trace()

        model.train(False)
        ensemble_.train(False)
        encoder.train(False)
        decoder.train(False)

        caption_generator=BAN_HSC(model,encoder,decoder,vocab_VQAE).to(device)


        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues

        if not os.path.exists(os.path.join(args.save_fig_loc,'VQAE','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method)): # # use ensemble_, vocab_VQAE
            os.makedirs(os.path.join(args.save_fig_loc,'VQAE','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method))
        if not os.path.exists(os.path.join(args.save_fig_loc,'OURS','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method)): # x_method == weight_only
            os.makedirs(os.path.join(args.save_fig_loc,'OURS','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method))
        if not os.path.exists(os.path.join(args.save_fig_loc,'CAPTION','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method)): # x_method == NoAtt
            os.makedirs(os.path.join(args.save_fig_loc,'CAPTION','model{}'.format(args.model_num),args.t_method,args.x_method,args.s_method))
        #check_captions([caption_generator, ensemble_], eval_loader, Dict_qid2vid, [vocab, vocab_VQAE],
         #              args.save_fig_loc, args.x_method, args.t_method, args.s_method, args)
        check_captions([caption_generator], eval_loader, Dict_qid2vid, [vocab],
                       args.save_fig_loc, args.x_method, args.t_method, args.s_method, args)

        ################################################################################################################





    process(args, model, eval_loader,Dict_qid2vid)
