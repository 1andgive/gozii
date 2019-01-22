""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import sys
import pickle
from build_vocab import Vocabulary
import os

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils
import pdb
from model_VQAE import EnsembleVQAE
from model import DecoderRNN
from codes.train import compute_score_with_logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--input', type=str, default='model_XAI/vqaE')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--save_fig_loc', type=str, default='Result_comparison/')
    parser.add_argument('--x_method', type=str, default='weight_only')  # mean, NoAtt, sum, weight_only
    parser.add_argument('--t_method', type=str, default='uncorr')  # mean, uncorr
    parser.add_argument('--s_method', type=str, default='BestOne')  # BestOne, BeamSearch
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
    return dataloader.dataset.label2ans[idx[0]]


def get_logits(model, dataloader):
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
            logits, att = model(v, b, q, None)
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)
            idx += batch_size
            if args.debug:
                print(get_question(q.data[0], dataloader))
                print(get_answer(logits.data[0], dataloader))
    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p+eps).log()).sum(2).sum(0) # g

def evaluate(model, dataloader):
    N = len(dataloader.dataset)
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    if hasattr(model.module, 'glimpse'):
        entropy = torch.Tensor(model.module.glimpse).zero_().cuda()
    idx=0
    bar = progressbar.ProgressBar(maxval=N).start()
    for v, b, q, a in iter(dataloader):
        bar.update(idx)
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = q.type(torch.LongTensor)
        q = Variable(q, volatile=True).cuda()
        batch_size=v.size(0)
        pred, att = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        idx += batch_size
        if att is not None and 0 < model.module.glimpse:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
    bar.update(idx)
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)

    return score, upper_bound, entropy


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')
    eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    with open(args.vocab_vqae_path, 'rb') as f:
        vocab_VQAE = pickle.load(f)

    # Build the models)
    decoder_vqae = DecoderRNN(args.embed_size, args.hidden_size, len(vocab_VQAE), args.num_layers).to(device)


    def process(args, model, eval_loader):

        if not os.path.exists(os.path.join('LogWrites','VQAE')):
            os.makedirs(os.path.join('LogWrites','VQAE'))

        logger = utils.Logger(os.path.join('LogWrites','VQAE', 'log.txt'))

        model_vqae_path = args.input + '/vqaE-%s-Final.pth' % \
                          ('' if 0 > args.epoch else '%d' % args.epoch)

        print('loading %s' % model_vqae_path)
        model_vqae_data = torch.load(model_vqae_path)

        model = nn.DataParallel(model).cuda()

        ensemble_=EnsembleVQAE(model,decoder_vqae)
        ensemble_.load_state_dict(model_vqae_data['model_state'])

        ensemble_.train(False)

        eval_score, bound, entropy = evaluate(model, eval_loader)
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)


        utils.create_dir(args.output)
        if 0 <= args.epoch:
            model_label += '_epoch%d' % args.epoch


    process(args, model, eval_loader)



