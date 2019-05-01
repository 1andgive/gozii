import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
from model import DecoderRNN

class EnsembleVQAE(nn.Module):
    def __init__(self,BAN,decoder,embed_size=256 ,vdim=2048, qdim=1280):
        super(EnsembleVQAE,self).__init__()
        self.BAN=BAN
        self.decoder = decoder
        self.act_relu = nn.ReLU()
        self.embed_size=embed_size #embedding size =300,
        self.linearWq = nn.Linear(qdim, embed_size)
        self.linearWv = nn.Linear(vdim, embed_size)

    def forward(self, v, b, q, ans, captions, lengths, isBUTD=False):

        q=q.squeeze(1)
        logits, att = self.BAN(v, b, q, None)

        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                              2).unsqueeze(
            2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        atted_v_feats = att_for_v * v  # attended visual features
        atted_v_feats_sum = torch.sum(atted_v_feats, 1).unsqueeze(1)
        atted_v_feats_mean = torch.mean(atted_v_feats, 1).unsqueeze(1)

        q_emb = self.BAN.module.extractQEmb(q)
        q_emb = q_emb[:, -1, :]
        q_emb = q_emb.unsqueeze(1)

        if(isBUTD):
            outputs_caption=self.decoder(atted_v_feats, None, atted_v_feats_mean.squeeze(1), captions, lengths)
            outputs_caption = pack_padded_sequence(outputs_caption, lengths, batch_first=True)[0]
            return logits, att, outputs_caption
        else:
            context_vec=self.act_relu(self.linearWq(q_emb)) * self.act_relu(self.linearWv(atted_v_feats_sum))
            outputs_caption=self.decoder(context_vec.squeeze(1), captions, lengths)

            return logits, att, outputs_caption

    def generate_caption(self, v, b, q, t_method='mean',x_method='sum', s_method='BestOne', model_num=1, isBUTD=False, obj_nums=obj_nums):

        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        q = q.squeeze(1)
        logits, att = self.BAN(v, b, q, None)

        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                              2).unsqueeze(
            2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        atted_v_feats = att_for_v * v  # attended visual features
        atted_v_feats_sum = torch.sum(atted_v_feats, 1)
        atted_v_feats_mean= atted_v_feats_sum / obj_nums.unsqueeze(1)
        atted_v_feats_sum=atted_v_feats_sum.unsqueeze(1)

        q_emb = self.BAN.module.extractQEmb(q)
        q_emb = q_emb[:, -1, :]
        q_emb = q_emb.unsqueeze(1)

        if(isBUTD):
            Generated_Captions=self.decoder.sample(atted_v_feats, atted_v_feats_mean)
            return Generated_Captions, logits, att
        else:
            encoded_features = self.act_relu(self.linearWq(q_emb)) * self.act_relu(self.linearWv(atted_v_feats_sum))
            encoded_features=encoded_features.squeeze(1)

            if(s_method == 'BestOne'):
                if (model_num < 7):
                    Generated_Captions=self.decoder.sample(encoded_features)
                else:
                    Generated_Captions = self.decoder.sample2(encoded_features)
            elif(s_method == 'BeamSearch'):
                Generated_Captions = self.decoder.BeamSearch(encoded_features,NumBeams=3)

            return Generated_Captions, logits, att
