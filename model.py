import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
import numpy as np
from torch.nn.utils.weight_norm import weight_norm

# model = expectation, relu
# model2 = probability, relu
# model3 = expectation, tanh


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class Encoder_HieStackedCorr(nn.Module):
    def __init__(self, embed_size, vdim, model_num, num_stages=5,LRdim=64, hidden_size=512):
        super(Encoder_HieStackedCorr, self).__init__()
        if(model_num <= 6):
            self.linear=nn.Linear(vdim,embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        else:
            self.linear = nn.Linear(vdim, hidden_size)
            self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

        self.num_stages=num_stages
        self.lowRank_dim=LRdim
        self.linear_U1= weight_norm(nn.Linear(vdim,LRdim))
        self.linear_U2=weight_norm(nn.Linear(vdim,LRdim))
        self.act_relu=nn.ReLU()
        self.act_tanh=nn.Tanh()
        self.act_Lrelu = nn.LeakyReLU()
    def forward(self, Vmat, t_method='mean', model_num=1):
        assert t_method in ['mean', 'uncorr']
        if(model_num > 6):
            model_num=1
        #assert model_num in [1,2,3,4,5,6]

        if(model_num==1):
            if(t_method == 'mean'):
                features = self.bn(self.linear(self.MeanVmat(Vmat)))
            elif(t_method == 'uncorr'):
                features = self.bn(self.linear(self.MeanVmat(self.UnCorrVmat(Vmat))))
        elif(model_num==2):
            if (t_method == 'mean'):
                features = self.bn(self.linear(self.SumVmat(Vmat)))
            elif (t_method == 'uncorr'):
                features = self.bn(self.linear(self.SumVmat(self.UnCorrVmat(Vmat)))) 
        elif(model_num==3):
            if (t_method == 'mean'):
                features = self.bn(self.linear(self.MeanVmat(Vmat)))
            elif (t_method == 'uncorr'):
                features = self.bn(self.linear(self.MeanVmat(self.UnCorrVmat_tanh(Vmat))))
        elif(model_num==4):
            if (t_method == 'mean'):
                features = self.bn(self.linear(self.MeanVmat(Vmat)))
            elif (t_method == 'uncorr'):
                features = self.bn(self.linear(self.MeanVmat(self.UnCorrVmat_Lrelu(Vmat))))
        elif (model_num == 5):
            if (t_method == 'mean'):
                features = self.bn(self.linear(self.MeanVmat(Vmat)))
            elif (t_method == 'uncorr'):
                features = self.bn(self.linear(self.MeanVmat(self.UnCorrelatedResidualHierarchy(10,Vmat))))
        elif (model_num == 6):
            if (t_method == 'mean'):
                features = self.bn(self.linear(self.MeanVmat(Vmat)))
            elif (t_method == 'uncorr'):
                features = self.bn(self.linear(self.MeanVmat(self.UnCorrVmat_Detail(Vmat))))
        return features

    def UnCorrelatedResidualHierarchy(self, num_stages, Vmat):
        ##building Un-Correlated Residual Hierarchy for Visual Matrix
        num_batches = Vmat.size(0)

        for i in range(num_stages):

            psi = torch.matmul(Vmat, torch.transpose(Vmat, 1, 2))

            Diag_Tensor = torch.zeros(num_batches, psi.size(1)).cuda()
            for k in range(num_batches):
                Diag_Tensor[k] = torch.rsqrt(torch.diag(psi[k]) + 1e-6)
            Corr = Diag_Tensor.unsqueeze(1) * psi * Diag_Tensor.unsqueeze(2)
            Umat = 1 - Corr
            Vmat = Vmat + torch.matmul(Umat, Vmat)  # Vmat = V' (transposed version of V
        return Vmat

    def MeanVmat(self,Vmat):
        v_final=torch.mean(Vmat,1)
        return v_final

    def SumVmat(self,Vmat):
        v_final=torch.sum(Vmat,1)
        return v_final

    def UnCorrVmat(self,Vmat):
        #pdb.set_trace()
        RightUnCorr=self.act_relu(self.linear_U1(Vmat))
        LeftUnCorr = self.act_relu(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))

        return torch.matmul(UnCorr,Vmat)

    def UnCorrVmat_Detail(self,Vmat):
        #pdb.set_trace()
        num_batches = Vmat.size(0)
        RightUnCorr=self.act_relu(self.linear_U1(Vmat))
        LeftUnCorr = self.act_relu(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))
        Diag_Tensor = torch.zeros(num_batches, UnCorr.size(1)).cuda()
        for k in range(num_batches):
            Diag_Tensor[k] = torch.rsqrt(torch.diag(UnCorr[k]) + 1e-6)
        UnCorr = Diag_Tensor.unsqueeze(1) * UnCorr * Diag_Tensor.unsqueeze(2)
        UnCorr=1+torch.eye(Vmat.size(1)).cuda()-UnCorr

        return torch.matmul(UnCorr, Vmat)

    def UnCorrVmat_tanh(self,Vmat):
        #pdb.set_trace()
        RightUnCorr=self.act_tanh(self.linear_U1(Vmat))
        LeftUnCorr = self.act_tanh(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))

        return torch.matmul(UnCorr,Vmat)

    def UnCorrVmat_Lrelu(self,Vmat):
        #pdb.set_trace()
        RightUnCorr=self.act_Lrelu(self.linear_U1(Vmat))
        LeftUnCorr = self.act_Lrelu(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))

        return torch.matmul(UnCorr,Vmat)



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=40):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab_size=vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths, model_num=1):
        """Decode image feature vectors and generates captions."""


        if model_num < 7:
            embeddings = self.embed(captions)
            embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed)

        elif (model_num == 7):
            #embeddings = self.embed(captions[:,:-1]) # input에서는 <eos>가 제거됨
            embeddings = self.embed(captions)  # input에서는 <eos>가 제거됨
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            features=features.unsqueeze(0)
            init_memory=torch.cuda.FloatTensor(features.size()).fill_(0)
            hiddens, _ = self.lstm(packed, [features,init_memory])


        outputs = self.linear(hiddens[0]) #teacher forcing 방식
        return outputs

    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def sample2(self, features, input=1):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = features.unsqueeze(0)
        init_memory = torch.cuda.FloatTensor(features.size()).fill_(0)
        states=[features, init_memory]
        inputs=self.embed(torch.cuda.LongTensor([input]))
        inputs = inputs.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def sample_with_answer(self, features, vocab_candidates, states=None):
        """Generate captions for given image features using greedy search."""

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            if(i==0):
                predicted=torch.cuda.LongTensor([1]) # '<start>'
            # elif (i == 1):
            #     answer_word=vocab_candidates[0]
            #
            #     predicted=torch.cuda.LongTensor([answer_word]) # '<start>'
            #     vocab_candidates.remove(answer_word)
            elif (i == 1):
                outputs_candidate=outputs[0,vocab_candidates]
                _, output_best=outputs_candidate.max(0)
                predicted=vocab_candidates[output_best]
                predicted = torch.cuda.LongTensor([predicted])
            sampled_ids.append(predicted)

            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def BeamSearch(self, features, states=None, NumBeams=5):
        """Generate captions for given image features using greedy search."""
        num_batches=features.size(0)
        sample_ids=[]
        inputs = [features.unsqueeze(1) for x in range(NumBeams)]
        states = [states for x in range(NumBeams)]
        Probs = torch.zeros(num_batches,NumBeams).cuda()

        tmp_2step_samples_ids=[[[] for x in range(NumBeams)] for y in range(NumBeams)]
        tmp_2step_Probs = [[] for y in range(NumBeams)]

        for i in range(self.max_seg_length):
            for beam_idx in range(NumBeams):

                hiddens, states[beam_idx] = self.lstm(inputs[beam_idx], states[beam_idx])  # hiddens: (batch_size, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
                tmp_probs, predicted = max_k(outputs, dim_=1, k=NumBeams)  # predicted: (batch_size, NumBeams), tmp_probs: (batch_size, NumBeams)


                # if intial step, directly go to next step
                if( i == 0):
                    if 1 in predicted.reshape(-1,1):

                        sample_ids=[[torch.LongTensor([1]).cuda()] for idx in range(NumBeams)]
                        inputs = [self.embed(torch.LongTensor([1]).cuda()).unsqueeze(1) for tmp_idx in
                                  range(NumBeams)]  # inputs: (batch_size, embed_size)
                        Probs = Probs + torch.ones(num_batches,NumBeams).cuda()
                    else:
                        sample_ids=[[predicted[:,idx]] for idx in range(NumBeams)]
                        inputs = [self.embed(predicted[:,tmp_idx]).unsqueeze(1) for tmp_idx in range(NumBeams)]  # inputs: (batch_size, embed_size)
                        Probs = Probs+tmp_probs
                    break # go to next step

                # find best-k among k*k candidates
                else:
                    tmp_2step_samples_ids[beam_idx]=[sample_ids[beam_idx].copy() for tmp_idx in range(NumBeams)]
                    [tmp_2step_samples_ids[beam_idx][tmp_idx].append(predicted[:,tmp_idx]) for tmp_idx in range(NumBeams)]
                    tmp_2step_Probs[beam_idx] = (Probs* tmp_probs).unsqueeze(2) #Probs should be cumulative probability

                # sample_ids[beam_idx].append(predicted[beam_idx])
                # inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
                # inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)

            if (i>0):
                # first, select best new elements for sample_ids from tmp_2step_samples_ids
                # select inputs for the next step
                Probs,Top_k_idx0,Top_k_idx1=max2D_k(tmp_2step_Probs,k=NumBeams) # Top_k_idx2D => (rows,cols) and rows corresponds to beam_idx value of sample_ids and Probs

                for id in range(NumBeams):
                    sample_ids[id]=tmp_2step_samples_ids[int(Top_k_idx0[:,id].item())][int(Top_k_idx1[:,id].item())]

                    inputs[id] = self.embed(sample_ids[id][-1])
                    inputs[id] = inputs[id].unsqueeze(1)

        sampled_id_list = torch.Tensor(sample_ids)

        return sampled_id_list

class BAN_HSC(nn.Module):
    def __init__(self,BAN,encoder,decoder,vocab):
        super(BAN_HSC,self).__init__()
        self.BAN=BAN
        self.encoder=encoder
        self.decoder=decoder
        self.vocab=vocab
    def generate_caption(self, v, b, q, t_method='mean',x_method='sum', s_method='BestOne', model_num=1):

        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        encoded_features, logits, att=self.forward( v, b, q , t_method=t_method, x_method=x_method, s_method=s_method)


        if(s_method == 'BestOne'):
            if (model_num < 7):
                Generated_Captions=self.decoder.sample(encoded_features)
            else:
                input_=self.vocab('<start>')
                Generated_Captions = self.decoder.sample2(encoded_features,input=input_)
        elif(s_method == 'BeamSearch'):
            Generated_Captions = self.decoder.BeamSearch(encoded_features,NumBeams=3)

        return Generated_Captions, logits, att

    def generate_explain(self,encoded_features, vocab_candidates, t_method='mean',x_method='sum', s_method='BestOne', model_num=1):
        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        if (s_method == 'BestOne'):
            if (model_num < 7):
                Generated_Explains = self.decoder.sample_with_answer(encoded_features, vocab_candidates)
            else:
                input_ = self.vocab('<start>')
                _= self.decoder.sample2(encoded_features, input=input_)
        elif (s_method == 'BeamSearch'):
            _ = self.decoder.BeamSearch(encoded_features, NumBeams=3)

        return Generated_Explains

    def generate_caption_n_context(self, v, b, q, t_method='mean',x_method='sum', s_method='BestOne'):

        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        encoded_features, logits, att=self.forward( v, b, q , t_method=t_method, x_method=x_method, s_method=s_method)

        if(s_method == 'BestOne'):
            Generated_Captions=self.decoder.sample(encoded_features)
        elif(s_method == 'BeamSearch'):
            Generated_Captions = self.decoder.BeamSearch(encoded_features,NumBeams=3)

        return Generated_Captions, logits, att, encoded_features


    def forward(self, v, b, q, t_method='mean',x_method='sum', s_method='BestOne'):

        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        logits, att=self.BAN(v,b,q,None)

        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                               2).unsqueeze(2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        num_objects=att_final.size(1)


        if x_method=='sum':
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats=torch.sum(atted_v_feats,1).unsqueeze(1)
        elif x_method == 'mean':
            atted_v_feats = att_for_v * v  # attended visual features
        elif x_method == 'sat_cut':
            att_for_v=att_for_v.cpu().numpy()
            att_for_v=np.clip(att_for_v,0,1.0/num_objects)
            att_for_v=torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3':
            att_for_v,_ = max_k(att_for_v,dim_=1,k=3)
            v,_ = max_k(v,dim_=1,k=3)
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'top3_sat':
            att_for_v,_ = max_k(att_for_v,dim_=1,k=3)
            v,_ = max_k(v,dim_=1,k=3)
            att_for_v = att_for_v.cpu().numpy()
            att_for_v = np.clip(att_for_v, 0, 1.0 / num_objects)
            att_for_v = torch.from_numpy(att_for_v).cuda()
            atted_v_feats = att_for_v * v  # attended visual features
            atted_v_feats = torch.sum(atted_v_feats, 1).unsqueeze(1)
        elif x_method == 'weight_only':
            atted_v_feats = (num_objects*att_for_v) * v  # attended visual features
        elif x_method == 'NoAtt':
            atted_v_feats=v

        #pdb.set_trace()
        encoded_features=self.encoder(atted_v_feats,t_method)

        return encoded_features, logits, att





def max_k(inputTensor,dim_=0,k=1):
    _, idx_att = torch.sort(inputTensor, dim=dim_, descending=True)

    if(dim_ != 0):
        idx_att=torch.transpose(idx_att,0,dim_)
        inputTensor=torch.transpose(inputTensor,0,dim_)
    idx_att=idx_att[:k]
    outputTensor = inputTensor[idx_att.squeeze()]
    if (dim_ != 0):
        idx_att = torch.transpose(idx_att, 0, dim_)
        outputTensor = torch.transpose(outputTensor, 0, dim_)
    return outputTensor, idx_att

def append_elemwise(list1, list2):
    assert len(list1)==len(list2)

    for idx in range(len(list1)):
        list1[idx].append(list2[idx])

    return list1

def max2D_k(list2D,k=1):
    num_cols=list2D[0].size(1)
    tensor2D=torch.transpose(torch.cat(list2D,2),1,2)
    num_batches=tensor2D.size(0)
    value_list=[]
    idx0_list=[]
    idx1_list=[]
    for batch_idx in range(num_batches):
        tensor2D_tmp=tensor2D[batch_idx]
        values,idx=torch.sort(tensor2D_tmp.reshape(-1,1).squeeze(),descending=True)
        values=values[:k]
        idx=idx[:k]
        idx0=[]
        idx1=[]
        for i in range(k):
            tmp_idx=divmod(idx[i].item(),num_cols)
            idx0.append(tmp_idx[0])
            idx1.append(tmp_idx[1])
        value_list.append(values.unsqueeze(0))
        idx0_list.append(torch.Tensor(idx0).unsqueeze(0))
        idx1_list.append(torch.Tensor(idx1).unsqueeze(0))
    return torch.cat(value_list,0), torch.cat(idx0_list,0), torch.cat(idx1_list,0)


class DecoderTopDown(nn.Module):
    def __init__(self, embed_size, vdim, hidden_size1, hidden_size2, vocab_size, num_layers, max_seq_length=40, paramH=256):
        """Set the hyper-parameters and build the layers."""
        super(DecoderTopDown, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size2=hidden_size2
        self.hidden_size1 = hidden_size1
        self.TopDownAttentionLSTM = nn.LSTM(hidden_size2+vdim+embed_size, hidden_size1, num_layers, batch_first=True)
        self.LanguageLSTM = nn.LSTM(hidden_size1+vdim, hidden_size2, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size2, vocab_size)
        self.max_seg_length = max_seq_length
        self.softmax = nn.Softmax(dim=1)
        self.paramH=paramH
        self.linear_Wva=nn.Linear(vdim,self.paramH)
        self.linear_Wha = nn.Linear(hidden_size1,self.paramH)
        self.linear_wa = nn.Linear(self.paramH, 1)
        self.act_tanh=nn.Tanh()

    def forward(self, Vmat, enc_features, union_vfeats, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        #embeddings = torch.cat((enc_features.unsqueeze(1), embeddings), 1)


        ## OVERALL STRATEGY
        # loop with padded sequence length
        # eliminate data that meet up its original length

        batch_size=len(lengths)
        iter_batch_idx=list(range(batch_size))

        max_seq_length=lengths[0]

        hidden2=torch.cuda.FloatTensor(batch_size, self.hidden_size2).fill_(0)

        states1=None
        states2=[hidden2.unsqueeze(0), hidden2.unsqueeze(0)]
        iter_batch_idx_list=[]
        states1_list=[[],[]]
        states2_list=[[],[]]
        outputs = torch.cuda.FloatTensor(batch_size, max_seq_length, self.vocab_size).fill_(0)
        for i in range(max_seq_length):
            iter_batch_idx=[j for j in iter_batch_idx if lengths[j]> i]
            iter_batch_idx_list.append(iter_batch_idx)
            input1=torch.cat([hidden2[iter_batch_idx_list[i],:], union_vfeats[iter_batch_idx_list[i], :], embeddings[iter_batch_idx_list[i],i,:]],1)
            input1=input1.unsqueeze(1)

            if i==0:
                hidden1, states1=self.TopDownAttentionLSTM(input1,states1)
            else:
                hidden1, states1 = self.TopDownAttentionLSTM(input1, tuple([states1_list[0][:,iter_batch_idx_list[i],:].cuda(), states1_list[1][:,iter_batch_idx_list[i],:].cuda()]))

            atten_logit=self.linear_wa(self.act_tanh(self.linear_Wva(Vmat[iter_batch_idx_list[i], :])+self.linear_Wha(hidden1)))
            atten_logit=atten_logit.squeeze(2)
            atten=self.softmax(atten_logit)
            atten=atten.unsqueeze(1)
            atten_vfeats=torch.matmul(atten,Vmat[iter_batch_idx_list[i], :])
            input2=torch.cat([atten_vfeats, hidden1],2)
            if i == 0:
                hidden2, states2= self.LanguageLSTM(input2,states2)
            else:
                hidden2, states2 = self.LanguageLSTM(input2,tuple([states2_list[0][:,iter_batch_idx_list[i],:].cuda(), states2_list[1][:,iter_batch_idx_list[i],:].cuda()]))


            valid_outputs = self.linear(hidden2)  # teacher forcing 방식
            hidden2=hidden2.squeeze(1)
            outputs[iter_batch_idx_list[i],i,:]=valid_outputs.squeeze(1)
            if i == 0:
                states1_list[0]= states1[0].clone().cpu()
                states1_list[1]= states1[1].clone().cpu()
                states1 = None
                states2_list[0]= states2[0].clone().cpu()
                states2_list[1] =states2[1].clone().cpu()
                states2 = None
            else:
                states1_list[0][:, iter_batch_idx_list[i], :] = states1[0][:, iter_batch_idx_list[i], :].cpu()
                states1_list[1][:, iter_batch_idx_list[i], :] = states1[1][:, iter_batch_idx_list[i], :].cpu()
                states1 = None
                states2_list[0][:, iter_batch_idx_list[i], :] = states2[0][:, iter_batch_idx_list[i], :].cpu()
                states2_list[1][:, iter_batch_idx_list[i], :] = states2[1][:, iter_batch_idx_list[i], :].cpu()
                states2 = None
        return outputs
        #hidden2_out_prev=

        #hiddens, _ = self.lstm(packed)
