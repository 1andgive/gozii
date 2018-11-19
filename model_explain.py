import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
import numpy as np
from torch.nn.utils.weight_norm import weight_norm


def W_Sim(w1,w2):
    w1=w1.squeeze(0)
    w2=w2.squeeze(0)
    score=0.5*(1+torch.matmul(w1,w2)/(torch.norm(w1)*torch.norm(w2)+1e-9))
    return score

def Sen_Sim(Input,Caption):
    Input=Input.squeeze(0)
    Caption=Caption.squeeze(0)
    T_input=Input.size(0)
    T_caption=Caption.size(0)
    score=0
    for i in range(T_input):
        w_input=Input[i]
        scores=[W_Sim(w_input,Caption[j]) for j in range(T_caption)]
        score=score+max(scores)


    score=score/T_input
    return score


