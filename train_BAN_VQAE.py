import os
import sys


import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

from build_vocab import Vocabulary
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC, DecoderTopDown
from model_VQAE import EnsembleVQAE
import utils_hsc as utils_save


sys.path.append('D:\\VQA\\BAN')

import codes.base_model as base_model
from codes.dataset import Dictionary, VQAFeatureDataset
from model_explain import CaptionEncoder
from data_loader import VQAE_FineTunning_loader
import pickle
import pdb


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='data/VQA_E_vocab.pkl', help='path for VQA_E vocabulary wrapper')
    parser.add_argument('--vqa_E_train_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json',
                        help='path for train annotation json file')
    parser.add_argument('--vqa_E_val_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json',
                        help='path for validation annotation json file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--paramH', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--hidden_size_BAN', type=int, default=1280, help='dimension of GRU hidden states in BAN')
    parser.add_argument('--hsc_path', type=str, default='models/', help='path for resuming hsc pre-trained models')
    parser.add_argument('--num_epoch',type=int,default=50)
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
    parser.add_argument('--checkpoint_dir', type=str, default='None', help='loading from this checkpoint')
    parser.add_argument('--isBUTD', type=bool, default=False)
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


    vqaE_loader = VQAE_FineTunning_loader('train+val', dictionary_vqa, vocab_VQAE, args.batch_size, shuffle=True, num_workers=0)
    if(args.isBUTD):
        decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab_VQAE),
                                 args.num_layers,
                                 paramH=args.paramH)
        vqaE_path = 'vqaE_BUTD'
    else:
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)
        vqaE_path = 'vqaE'

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(vqaE_loader.dataset.datasets[0], args.num_hid, args.op, args.gamma).cuda()

    model_path = args.input + '/model%s.pth' % \
                 ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

    print('loading %s' % model_path)
    model_data = torch.load(model_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    ban_vqaE = EnsembleVQAE(model,decoder).to(device)

    params=ban_vqaE.parameters()
    optimizer = torch.optim.Adamax(params, lr=args.learning_rate)



    if not os.path.exists(os.path.join('model_xai', vqaE_path)):
        os.makedirs(os.path.join('model_xai', vqaE_path))

    save_path = os.path.join('model_xai', vqaE_path)


    criterion_Explain = nn.CrossEntropyLoss()

    requires_grad_Switch(model,isTrain=True)
    requires_grad_Switch(decoder, isTrain=True)

    total_step = len(vqaE_loader)

    epoch_start = 0

    if (args.checkpoint_dir != 'None'):
        model_vqaE_path = os.path.join(
            'model_xai', vqaE_path,
            args.checkpoint_dir)
        model_vqaE_data = torch.load(model_vqaE_path)
        ban_vqaE.load_state_dict(model_vqaE_data['model_state'])
        optimizer.load_state_dict(model_vqaE_data['optimizer_state'])
        epoch_start = model_vqaE_data['epoch']+1

    for epoch in range(epoch_start, args.num_epoch):
        for i,(v, b, q, ans, captions, lengths) in enumerate(vqaE_loader):

            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            v = v.cuda()
            captions = captions.cuda()
            targets = targets.cuda()

            ban_vqaE.zero_grad()

            preds, att, outputs = ban_vqaE(v, b, q, ans, captions, lengths, isBUTD=args.isBUTD)
            #outs=caption_encoder.forward_DoubleLSTM_out(hiddens)
            Loss = criterion_Explain(outputs, targets) + instance_bce_with_logits(preds, ans)
            Loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, i, total_step, Loss.item(), np.exp(Loss.item())))

        # Save the model checkpoints
        save_path = os.path.join(
            'model_xai', vqaE_path, 'vqaE-{}-Final.pth'.format(epoch + 1))
        ####################################################################################################
        # implement utils.save_xai_model

        utils_save.save_all_model(save_path, ban_vqaE, epoch, optimizer)

        ####################################################################################################
