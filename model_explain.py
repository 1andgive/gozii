import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
from nltk.tokenize import word_tokenize
from model import max_k

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

# MAP 'Caption Word' and 'Answer Word' to 'Question Dictionary Index', find Word Embedding from this dictionary, and measure Sentence Similarity
def Relev_Check(Caption,Question,Answer,W_Emb,Dictionary):
    x_caption = []
    if Caption[0] == '<start>':
        for idx3 in Caption[1:]:
            if idx3 != '<end>':
                x_caption.append(idx3)
            else:
                break

    Wc_inQ_idx = []
    Wa_inQ_idx = []
    for Wc in x_caption:
        if Wc in Dictionary.word2idx.keys():
            Wc_inQ_idx.append(Dictionary.word2idx[Wc])

    for Wa in word_tokenize(Answer):
        if Wa in Dictionary.word2idx.keys():
            Wa_inQ_idx.append(Dictionary.word2idx[Wa])

    Wc = torch.Tensor(Wc_inQ_idx)
    Wa = torch.Tensor(Wa_inQ_idx)
    if (Wc.nelement() == 0):
        return 0.0
    elif (Wa.nelement() == 0):
        # RelScore=0.0
        return None
    else:
        Wc = Wc.unsqueeze(0)
        # Wq=q

        Wa = Wa.unsqueeze(0)
        Wc_Emb = W_Emb(Wc.type(torch.cuda.LongTensor))
        Wq_Emb = W_Emb(Question)
        Wa_Emb = W_Emb(Wa.type(torch.cuda.LongTensor))
        RelScore = 0.5 * (Sen_Sim(Wq_Emb, Wc_Emb) + Sen_Sim(Wa_Emb, Wc_Emb))
        RelScore = RelScore.item()

        return RelScore


# MAP 'Caption Word' and 'Answer Word' to 'Question Dictionary Index', find Word Embedding from this dictionary, and measure Sentence Similarity
def Relev_Check_by_IDX(CaptionIDX,QuestionIDX,AnswerIDX,W_Emb,Dict_AC_2_Q):

    with torch.no_grad():
        RelScoreList=[]
        Dict_A2Q = Dict_AC_2_Q[0]
        Dict_C2Q = Dict_AC_2_Q[1]

        for idx in range(CaptionIDX.size(0)):

            Caption=CaptionIDX[idx]
            Wa_IDX=AnswerIDX[idx]
            Question=QuestionIDX[idx]

            x_caption = []
            if Caption[0] == 1:
                for idx3 in Caption[1:]:
                    if idx3 != 2:
                        x_caption.append(idx3)
                    else:
                        break

            Wc_inQ_idx = []
            for Wc_IDX in x_caption:
                if Wc_IDX.item() in Dict_C2Q.keys():
                    Wc_inQ_idx.append(Dict_C2Q[Wc_IDX.item()])



            if Wa_IDX.item() in Dict_A2Q.keys():
                Wa_inQ_idx=Dict_A2Q[Wa_IDX.item()]
            else:
                Wa_inQ_idx=[]

            Wc = torch.Tensor(Wc_inQ_idx)
            Wa = torch.Tensor(Wa_inQ_idx)


            if (Wc.nelement() == 0):
                RelScoreList.append(0.0)
            elif (Wa.nelement() == 0):
                # RelScore=0.0
                RelScoreList.append(0.0)
            else:
                Wc = Wc.unsqueeze(0)
                # Wq=q

                Wa = Wa.unsqueeze(0)
                Question=Question.unsqueeze(0)

                Wc_Emb = W_Emb(Wc.type(torch.cuda.LongTensor))
                Wq_Emb = W_Emb(Question)
                Wa_Emb = W_Emb(Wa.type(torch.cuda.LongTensor))
                RelScore = 0.5 * (Sen_Sim(Wq_Emb, Wc_Emb) + Sen_Sim(Wa_Emb, Wc_Emb))
                RelScore = RelScore.item()

                RelScoreList.append(RelScore)

    return torch.Tensor(RelScoreList)


class CaptionEncoder(nn.Module):
    def __init__(self,embed_size, hidden_size, c_hidden_size, num_class, low_rank=256, doubly_low_rank=256,num_layers=1, bidirectional_=False):
        super(CaptionEncoder,self).__init__()
        self.hidden_size=hidden_size
        self.embed_size=embed_size
        self.c_hidden_size=c_hidden_size
        self.lstm_=nn.LSTM(embed_size,c_hidden_size,num_layers,batch_first=True,bidirectional=bidirectional_) # word embedding to hidden size mapping, hidden size == q_embedding size
        self.P = weight_norm(nn.Linear(low_rank,num_class))
        self.U = weight_norm(nn.Linear(hidden_size, low_rank))
        if bidirectional_:
            self.V = weight_norm(nn.Linear(2*c_hidden_size, low_rank))
        else:
            self.V = weight_norm(nn.Linear(c_hidden_size, low_rank))
        self.U2 = weight_norm(nn.Linear(hidden_size, doubly_low_rank))
        if bidirectional_:
            self.V2 = weight_norm(nn.Linear(2*c_hidden_size, doubly_low_rank))
        else:
            self.V2 = weight_norm(nn.Linear(c_hidden_size, doubly_low_rank))
        self.W = weight_norm(nn.Linear(c_hidden_size, num_class))
        self.T=weight_norm(nn.Linear(hidden_size,2*c_hidden_size))
        self.act_relu = nn.ReLU()
        self.p_=nn.Parameter(torch.Tensor(1,1,1,low_rank).normal_())
        self.p_bias=nn.Parameter(torch.Tensor(1,1,1,1).normal_())
        self.p2c=weight_norm(nn.Linear(doubly_low_rank,num_class))
        self.softmax=nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)  # attention

    def forward(self, input_embeddings, states):

        outputs, hiddens = self.lstm_(input_embeddings, states) # final_hiddens = (h_n, c_n)

        return outputs, hiddens

    def forward_CL(self,Q_emb, C_emb):

        Emb =self.act_relu(self.V(C_emb)) * self.act_relu(self.U(Q_emb))

        Emb =self.act_relu(self.P(Emb))

        outputs = self.softmax(Emb.squeeze())
        return outputs


    def forward_CL_ATT(self,Q_emb, C_emb):
        right_ =self.act_relu(self.V(C_emb)).unsqueeze(1) # [b x 1 x phi x K] phi= number of words sequence in caption
        left_ = self.act_relu(self.U(Q_emb)).unsqueeze(1) # [b x 1 x psi x K] psi= number of words sequence in question
        left_=self.dropout(left_*self.p_) # [b x 1 x psi x K]
        Att_=torch.matmul(right_,torch.transpose(left_,2,3))+self.p_bias # [b x 1 x psi x phi]

        Att_=self.softmax(self.act_relu(Att_))
        left_=self.act_relu(self.U2(Q_emb))  # [b x phi x K2]
        right_=self.act_relu(self.V2(C_emb)) # [b x psi x K2]
        left_=torch.transpose(left_,1,2).unsqueeze(3) #[b x K2 x phi x 1]
        right_ = torch.transpose(right_, 1, 2).unsqueeze(2) #[b x K2 x 1 x psi]
        outputs=torch.matmul(torch.matmul(right_,Att_),left_)
        outputs=outputs.squeeze()
        outputs=outputs.unsqueeze(1)
        outputs=self.act_relu(self.p2c(outputs))

        outputs = self.softmax(outputs.squeeze())
        return outputs

    def forward_DoubleLSTM_initstate(self,Q_emb):

        Emb_input =self.act_relu(self.T(Q_emb))
        Emb_input=torch.transpose(Emb_input,0,1)

        return [Emb_input[:,:,:self.c_hidden_size].contiguous(), Emb_input[:,:,self.c_hidden_size:].contiguous()]

    def forward_DoubleLSTM_out(self,C_hidden):

        outs=self.act_relu(self.W(C_hidden))
        outs=self.softmax(outs.squeeze())

        return outs



    def forward_DirectCL(self,Q_emb,WcMat):

        left_ = self.act_relu(self.U(Q_emb))
        right_ = self.act_relu(self.V(WcMat))
        mu_ = torch.matmul(torch.transpose(right_,1,2),left_)

        y_=torch.matmul(WcMat,mu_)
        y_=self.act_relu(self.W(y_))
        y_=self.softmax(y_.squeeze())

        return y_



class GuideVfeat(nn.Module):
    def __init__(self,q_embed_size,vdim, decoder_hidden_dim):
        super(GuideVfeat,self).__init__()
        self.q_embed_size=q_embed_size
        self.vdim=vdim
        self.linear=nn.Linear(q_embed_size,vdim)
        self.act_relu = nn.ReLU()
        self.linear_hidden = nn.Linear(q_embed_size,decoder_hidden_dim)
        self.linear_hidden2 = nn.Linear(q_embed_size, decoder_hidden_dim)

        self.act_sig=nn.Sigmoid()

    def forward(self,q_emb,x):
        h_vec=self.act_relu(self.linear(q_emb))
        h_vec=h_vec.squeeze(1)
        x_new=(1+h_vec)*x # Guiding

        L0_approx=torch.log(torch.mean(torch.sum(self.act_sig(h_vec),1)))
        L2_approx=torch.log(torch.mean(torch.sum(h_vec,1)))

        return x_new, L0_approx, L2_approx

    def forward_hidden(self, q_emb):
        h_vec = torch.transpose(self.act_relu(self.linear_hidden(q_emb)),0,1)
        c_vec = torch.transpose(self.act_relu(self.linear_hidden2(q_emb)),0,1)
        return (h_vec, c_vec)


class UNCorrXAI(nn.Module):
    def __init__(self,BAN,encoder,decoder,CaptionEncoder,Guide):
        super(UNCorrXAI,self).__init__()
        self.BAN=BAN
        self.encoder=encoder # EncoderHieStackedCorr
        self.decoder=decoder # DecoderRNN
        self.CaptionEncoder=CaptionEncoder # CaptionEncoder
        self.Guide=Guide # GuideVfeat
        self.softmax = nn.Softmax(dim=1)

    def forward(self,v,b,q, t_method='mean',x_method='sum', s_method='BestOne', model_num= 1, flag='fix_guide', is_Init=False, is_Attention=False):
        assert flag in ['fix_guide', 'fix_cap_enc']
        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']
        assert model_num in [1, 2, 3, 4]
        assert t_method in ['mean', 'uncorr']

        logits, att = self.BAN(v, b, q, None)



        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                              2).unsqueeze(
            2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        num_objects = att_final.size(1)

        if x_method == 'sum':
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'mean':
            atted_v_feats = att_for_v * v  # attended visual features
        elif x_method == 'sat_cut':
            att_for_v = att_for_v.cpu().numpy()
            att_for_v = np.clip(att_for_v, 0, 1.0 / num_objects)
            att_for_v = torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3':
            att_for_v, _ = max_k(att_for_v, dim_=1, k=3)
            v, _ = max_k(v, dim_=1, k=3)
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3_sat':
            att_for_v, _ = max_k(att_for_v, dim_=1, k=3)
            v, _ = max_k(v, dim_=1, k=3)
            att_for_v = att_for_v.cpu().numpy()
            att_for_v = np.clip(att_for_v, 0, 1.0 / num_objects)
            att_for_v = torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'weight_only':
            atted_v_feats = (num_objects * att_for_v) * v  # attended visual features
        elif x_method == 'NoAtt':
            atted_v_feats = v


        Enc = self.encoder
        if (model_num == 1):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat(atted_v_feats))
        elif (model_num == 2):
            if (t_method == 'mean'):
                x_ = Enc.bn(Enc.linear(Enc.SumVmat(atted_v_feats)))
            elif (t_method == 'uncorr'):
                x_ = Enc.SumVmat(Enc.UnCorrVmat(atted_v_feats))
        elif (model_num == 3):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat_tanh(atted_v_feats))
        elif (model_num == 4):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat_Lrelu(atted_v_feats))

        q_embs=self.BAN.module.extractQEmb(q)
        q_emb = q_embs[:, -1, :]
        q_emb=q_emb.unsqueeze(1)

        if (is_Init):
            x_new=x_
        else:
            x_new, L0_guide, L2_guide=self.Guide(q_emb,x_)



        encoded_feats=Enc.bn(Enc.linear(x_new))

        # We don't train Decoder part!!

        if (is_Attention):
            # 2. Fixing Encoder-Class. Input to <Enc-Class> is Expectation of embedding
            states = None
            states2 = None
            Dec = self.decoder
            Cap_Enc = self.CaptionEncoder
            inputs = encoded_feats.unsqueeze(1)
            Embed_Table = Dec.embed(torch.cuda.LongTensor([ii for ii in range(Dec.vocab_size)]))
            Embed_Table = Embed_Table.unsqueeze(0)
            input_list = []
            for i in range(Dec.max_seg_length):
                hiddens, states = Dec.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)

                outputs = Dec.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
                ProbMF = self.softmax(outputs)  # Probability Mass Function of Words, (batch_size, vocab_size)
                Expected_Emb = torch.mean(ProbMF.unsqueeze(2) * Embed_Table, 1)
                input_list.append(Expected_Emb)  # inputs: (batch_size, 1, embed_size))

            inputs = torch.stack(input_list, 1)
            hiddens, _ = Cap_Enc(inputs, None)
            return logits, Cap_Enc.forward_CL_ATT(q_embs, hiddens), L0_guide, L2_guide

        else:

            ################################################################################################################
            # 1. Fixing Guide. Input to <Enc-Class> is embedding of discrete word

            if (flag == 'fix_guide'):
                states=None
                states2=None
                Dec=self.decoder
                Cap_Enc = self.CaptionEncoder
                inputs = encoded_feats.unsqueeze(1)
                sampled_ids = []
                for i in range(Dec.max_seg_length):
                    hiddens, states = Dec.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)

                    outputs = Dec.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
                    _, predicted = outputs.max(1)  # predicted: (batch_size)
                    inputs = Dec.embed(predicted)  # inputs: (batch_size, embed_size)
                    sampled_ids.append(predicted)
                    inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
                    hiddens2, states2 = Cap_Enc(inputs,states2)

                sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)


                return logits, Cap_Enc.forward_CL(q_emb,hiddens2), sampled_ids

            ################################################################################################################
            # 2. Fixing Encoder-Class. Input to <Enc-Class> is Expectation of embedding

            elif (flag == 'fix_cap_enc'):
                states = None
                states2 = None
                Dec = self.decoder
                Cap_Enc = self.CaptionEncoder
                inputs = encoded_feats.unsqueeze(1)
                Embed_Table=Dec.embed(torch.cuda.LongTensor([ii for ii in range(Dec.vocab_size)]))
                Embed_Table=Embed_Table.unsqueeze(0)
                for i in range(Dec.max_seg_length):
                    hiddens, states = Dec.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)

                    outputs = Dec.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
                    ProbMF = self.softmax(outputs)  # Probability Mass Function of Words, (batch_size, vocab_size)
                    Expected_Emb = torch.mean(ProbMF.unsqueeze(2)*Embed_Table,1)
                    inputs = Expected_Emb.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
                    hiddens2, states2 = Cap_Enc(inputs, states2)

                return logits, Cap_Enc.forward_CL(q_emb,hiddens2), L0_guide, L2_guide

    def Explain(self, v, b, q, t_method='mean', x_method='sum', s_method='BestOne', model_num=1):
        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']
        assert model_num in [1, 2, 3, 4]
        assert t_method in ['mean', 'uncorr']


        logits, att = self.BAN(v, b, q, None)



        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                              2).unsqueeze(
            2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        num_objects = att_final.size(1)

        if x_method == 'sum':
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'mean':
            atted_v_feats = att_for_v * v  # attended visual features
        elif x_method == 'sat_cut':
            att_for_v = att_for_v.cpu().numpy()
            att_for_v = np.clip(att_for_v, 0, 1.0 / num_objects)
            att_for_v = torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3':
            att_for_v, _ = max_k(att_for_v, dim_=1, k=3)
            v, _ = max_k(v, dim_=1, k=3)
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3_sat':
            att_for_v, _ = max_k(att_for_v, dim_=1, k=3)
            v, _ = max_k(v, dim_=1, k=3)
            att_for_v = att_for_v.cpu().numpy()
            att_for_v = np.clip(att_for_v, 0, 1.0 / num_objects)
            att_for_v = torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'weight_only':
            atted_v_feats = (num_objects * att_for_v) * v  # attended visual features
        elif x_method == 'NoAtt':
            atted_v_feats = v


        Enc = self.encoder
        if (model_num == 1):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat(atted_v_feats))
        elif (model_num == 2):
            if (t_method == 'mean'):
                x_ = Enc.bn(Enc.linear(Enc.SumVmat(atted_v_feats)))
            elif (t_method == 'uncorr'):
                x_ = Enc.SumVmat(Enc.UnCorrVmat(atted_v_feats))
        elif (model_num == 3):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat_tanh(atted_v_feats))
        elif (model_num == 4):
            if (t_method == 'mean'):
                x_ = Enc.MeanVmat(atted_v_feats)
            elif (t_method == 'uncorr'):
                x_ = Enc.MeanVmat(Enc.UnCorrVmat_Lrelu(atted_v_feats))

        q_emb=self.BAN.module.extractQEmb(q)
        q_emb=q_emb[:,-1,:]
        q_emb=q_emb.unsqueeze(1)


        x_, _, _ = self.Guide(q_emb, x_)

        encoded_feats = Enc.bn(Enc.linear(x_))

        if (s_method == 'BestOne'):
            Generated_Captions = self.decoder.sample(encoded_feats)
        elif (s_method == 'BeamSearch'):
            Generated_Captions = self.decoder.BeamSearch(encoded_feats, NumBeams=3)

        return Generated_Captions, logits, att

# MAP 'Caption Word' and 'Answer Word' to 'Question Dictionary Index', find Word Embedding from this dictionary, and measure Sentence Similarity
def CaptionVocabCandidate(Question,Answer, CocoVocab):
    CocoVocabList = []
    QuestionSentence=word_tokenize(Question)
    num_pad=QuestionSentence.count('_')
    for i in range(num_pad):
        QuestionSentence.remove('_')


    for Wa in word_tokenize(Answer):
        if Wa in CocoVocab.word2idx.keys():
            if (Wa=='yes') or (Wa=='no'):
                continue
            else:
                CocoVocabList.append(CocoVocab.word2idx[Wa])
            # CocoVocabList.append(CocoVocab.word2idx[Wa])

    if CocoVocabList==[]:
        for Wq in QuestionSentence:
            if Wq in CocoVocab.word2idx.keys():
                if (Wq == 'is') or (Wq == 'are') or (Wq == 'will') or (Wq == "'s") or (Wq == 'the') or (Wq == 'many') or (Wq == 'a'):
                    continue
                else:
                    CocoVocabList.append(CocoVocab.word2idx[Wq])


    return CocoVocabList