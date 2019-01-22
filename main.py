import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from codes.dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset
import codes.base_model as base_model
from codes.train import train
import codes.utils as utils
from codes.utils import trim_collate
from codes.dataset import tfidf_from_questions
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', type=bool, default=False, help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', type=bool, default=False, help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', type=bool, default=True, help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__': # 모듈의 이름이 저장되는 변수. https://dojang.io/mod/page/view.php?id=1148 참고.
    # 파이썬 인터프리터로 스크립트 파일을 직접 실행했을 때는 모듈의 이름이 아니라 '__main__'이 들어갑니다.

    args = parse_args()

    utils.create_dir(args.output) # parser.add_argument('--output', type=str, default='saved_models/ban')
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__()) # 로그 기록 파일

    torch.manual_seed(args.seed) # manual_seed sets the random seed from pytorch random number generators (랜덤 seed값이 같으면 랜덤 value 생성이 동일)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl') # Question to Word Embeeding dictionary
    train_dset = VQAFeatureDataset('train', dictionary, adaptive=True)
    val_dset = VQAFeatureDataset('val', dictionary, adaptive=True)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.op, args.gamma).cuda()

    tfidf = None
    weights = None

    if args.tfidf:
        dict = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')
        tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dict)
    model.w_emb.init_embedding('codes/tools/data/glove6b_init_300d.npy', tfidf, weights)

    model = nn.DataParallel(model).cuda()

    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

#=========================================== need to modify this part, too.================================================================
    if args.use_both: # use train & val splits to optimize
        if args.use_vg: # use a portion of Visual Genome dataset
            vg_dsets = [
                VisualGenomeFeatureDataset('train', \
                    train_dset.h5_path, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes),
                VisualGenomeFeatureDataset('val', \
                    val_dset.h5_path, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes)]
            trainval_dset = ConcatDataset([train_dset, val_dset]+vg_dsets)
        else:
            trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    train(model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)