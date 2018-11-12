
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


sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils


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
    parser.add_argument('--batch_size', type=int, default=1)
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
    parser.add_argument('--save_fig_loc', type=str, default='saved_figs/')
    parser.add_argument('--x_method', type=str, default='sum')
    parser.add_argument('--t_method', type=str, default='mean')
    parser.add_argument('--s_method', type=str, default='BestOne')
    parser.add_argument('--HSC_model', type=int, default=1)
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--log_step', type=int, default=20, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=400, help='step size for saving trained models')
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
    label_from_vqa=torch.zeros(p.size())
    label_from_vqa[idx]=1
    return dataloader.dataset.label2ans[idx.item()], label_from_vqa



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train_XAI(caption_generator, caption_encoder, BilinearTransformer_, Discretize_, Loss_Rel, vqa_loader, x_method_, t_method_, s_method_, num_epoch, optimizer_, log_step, save_step):
    N = len(vqa_loader.dataset)
    M = vqa_loader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N * num_epoch).start()
    total_step = len(vqa_loader)

    if not os.path.exists(os.path.join('model_xai', t_method_, x_method_, s_method_)):
        os.makedirs(os.path.join('model_xai',  t_method_, x_method_, s_method_))

    save_path = os.path.join('model_xai',  t_method_, x_method_, s_method_)

    for epoch in range(num_epoch):
        for v, b, q, i in iter(vqa_loader):
            bar.update(idx)
            batch_size = v.size(0)
            q = q.type(torch.LongTensor)
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()

                generated_captions, logits, att, context_vector = caption_generator.generate_caption_n_context(v, b, q,t_method=t_method_, x_method=x_method_, s_method=s_method_)

                pred[idx:idx + batch_size, :].copy_(logits.data)
                qIds[idx:idx + batch_size].copy_(i)
                idx += batch_size
                question_list=[]
                answer_list=[]
                captions_list=[]
                label_list=[]
                for idx in range(len(i)):
                    question_list.append(get_question(q.data[idx],vqa_loader))
                    if (s_method_ == 'BestOne'):
                        caption = [generated_captions[idx][w_idx].item() for w_idx in
                                   range(generated_captions.size(1))]
                        captions_list.append(caption)
                    answer_word,label_from_vqa=get_answer(logits.data[idx], vqa_loader)
                    answer_list.append(answer_word)
                    label_list.append(label_from_vqa)
                label_from_vqa=torch.stack(label_list,0)
                input_X=torch.stack(captions_list,0)

                input_Xnew=Discretize_(BilinearTransformer_(input_X,context_vector))

                logits = caption_encoder(input_Xnew)
                Loss_Discriminative = nn.functional.binary_cross_entropy_with_logits(logits,label_from_vqa)
                Loss_Relevance =Loss_Rel(input_X,input_Xnew)
                Loss=Loss_Discriminative+Loss_Relevance

                caption_encoder.zero_grad()
                BilinearTransformer_.zero_grad()
                Discretize_.zero_grad()

                Loss.backward()
                optimizer_.step()

                # Print log info
                if i % log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch, num_epoch, i, total_step, Loss.item(), np.exp(Loss.item())))

                    # Save the model checkpoints
                if (i + 1) % save_step == 0:
                    # pdb.set_trace()
                        model_path = os.path.join(
                            save_path, 'model-{}-{}.pth'.format(epoch + 1, i + 1))


                    ####################################################################################################
                    # implement utils.save_xai_model

                    utils.save_xai_model(save_path,  , epoch, optimizer_)

                    ####################################################################################################

        bar.update(idx)



if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    name = 'train'

    question_path = os.path.join(
        'codes\\tools\\data', 'v2_OpenEnded_mscoco_%s_questions.json' % \
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    if(args.HSC_model == 1):
        from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC,CaptionEncoder
    elif (args.HSC_model == 2):
        from model2 import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC,CaptionEncoder

        pdb.set_trace()
    elif (args.HSC_model == 3):
        from model3 import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC,CaptionEncoder

        pdb.set_trace()
    elif (args.HSC_model == 4):
        from model4 import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC,CaptionEncoder


    Dict_qid2ques={}
    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Visualization through eval_dataset

    vqa_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(vqa_dset, args.num_hid, args.op, args.gamma).cuda()

    vqa_loader = DataLoader(vqa_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    #train the encoder/decoder pair this time (fine-tuning stage for dual-loss)
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048,LRdim=args.LRdim).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    caption_encoder=CaptionEncoder(args.embed_size, args.hidden_size_BAN, decoder.embed, vqa_dset.num_ans_candidates)

    ################################################################################################################
    # implement these in model.py

    BilinearTransformer=
    Descretize=
    Loss_Rel=

    ################################################################################################################

    params = list(CaptionEncoder.parameters()) + list(BilinearTransformer.parameters()) + list(Descretize.parameters())
    # params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    def process(args, model, vqa_loader):
        model_path = args.input + '/model%s.pth' % \
                     ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

        print('loading vqa %s' % model_path)
        model_data = torch.load(model_path)

        if(args.t_method == 'mean'):
            model_hsc_path = os.path.join(
                args.hsc_path + 'model-{}-400.pth'.format(args.hsc_epoch))
        else:
            model_hsc_path = os.path.join(
                args.hsc_path,args.t_method, 'model-{}-400.pth'.format(args.hsc_epoch))

        print('loading hsc(attention caption) %s' % model_hsc_path)
        model_hsc_data=torch.load(model_hsc_path)

        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        #pdb.set_trace()

        model.train(False) # Do not train VQA part
        encoder.train(False)
        decoder.train(False)

        caption_generator=BAN_HSC(model,encoder,decoder)

        ################################################################################################################

        #Concatenate Encoder-Decoder to model and check whether the model generates correct captions based on visual cues

        if not os.path.exists(os.path.join(args.save_fig_loc,args.t_method,args.x_method,args.s_method)):
            os.makedirs(os.path.join(args.save_fig_loc,args.t_method,args.x_method,args.s_method))


        train_XAI(caption_generator, caption_encoder, BilinearTransformer, Descretize, Loss_Rel, vqa_loader, args.x_method, args.t_method, args.s_method, args.epoch, optimizer, args.log_step, args.save_step)

        ################################################################################################################





    process(args, model, vqa_loader)
