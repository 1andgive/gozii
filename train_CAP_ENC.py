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
from model import Encoder_HieStackedCorr, DecoderRNN, BAN_HSC
from model_explain import GuideVfeat, CaptionEncoder, Relev_Check_by_IDX, UNCorrXAI
from nltk.tokenize import word_tokenize
import utils_hsc as utils_save

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
from model_explain import CaptionEncoder
from data_loader import VQA_E_loader
import pickle
import pdb


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='data/vocab2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vqa_E_train_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json',
                        help='path for train annotation json file')
    parser.add_argument('--vqa_E_val_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json',
                        help='path for validation annotation json file')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args

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
        vocab_caption = pickle.load(f)

    vqa_E_train = json.load(open(args.vqa_E_train_path))
    vqa_E_val = json.load(open(args.vqa_E_val_path))

    vqaE_dset = VQA_E_loader(vqa_E_train,vqa_E_val,dictionary_vqa,vocab_caption)
    vqaE_loader = DataLoader(vqaE_dset, args.batch_size, shuffle=True, num_workers=0)

    pdb.set_trace()


