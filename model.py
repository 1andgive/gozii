import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
from copy import deepcopy
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
    def forward_BUTD(self, Vmat, t_method='mean', model_num=1,isUnion=False, checkBeta=False):
        assert t_method in ['mean', 'uncorr']
        if(model_num > 6):
            model_num=1
        #assert model_num in [1,2,3,4,5,6]

        if(model_num==1):
            if(t_method == 'mean'):
                features = self.MeanVmat(Vmat)
            elif(t_method == 'uncorr'):
                features,UMat=self.UnCorrVmat(Vmat)
                features = self.MeanVmat(features)
        elif(model_num==2):
            if (t_method == 'mean'):
                features = self.SumVmat(Vmat)
                Vmat= Vmat.size(1) * Vmat # BUTD uses this Vmat again
            elif (t_method == 'uncorr'):
                features, UMat = self.UnCorrVmat(Vmat)
                features = self.SumVmat(features)
                Vmat= Vmat.size(1) * Vmat
        elif(model_num==3):
            if (t_method == 'mean'):
                features = self.MeanVmat(Vmat)
            elif (t_method == 'uncorr'):
                features, UMat = self.UnCorrVmat_tanh(Vmat)
                features = self.MeanVmat(features)
        elif(model_num==4):
            if (t_method == 'mean'):
                features = self.MeanVmat(Vmat)
            elif (t_method == 'uncorr'):
                features, UMat = self.UnCorrVmat_Lrelu(Vmat)
                features = self.MeanVmat(features)
        elif (model_num == 5):
            if (t_method == 'mean'):
                features = self.MeanVmat(Vmat)
            elif (t_method == 'uncorr'):
                features, UMat = self.UnCorrelatedResidualHierarchy(10,Vmat)
                features = self.MeanVmat(features)
        elif (model_num == 6):
            if (t_method == 'mean'):
                features = self.MeanVmat(Vmat)
            elif (t_method == 'uncorr'):
                features, UMat = self.UnCorrVmat_Detail(Vmat)
                features = self.MeanVmat(features)

        if(t_method == 'uncorr'):
            # uncorr => isUnion : True
            enc_features = self.bn(self.linear(features))
            unified_features=features

            betas=torch.mean(UMat,1)
            betas=betas.unsqueeze(2)
            Vmat=betas*Vmat
            if(checkBeta):
                return enc_features, unified_features, Vmat, betas
            else:
                return enc_features, unified_features, Vmat, None
        elif (t_method == 'mean'):
            enc_features=self.bn(self.linear(features))
            unified_features=features
            return enc_features, unified_features, Vmat, None

    def forward(self, Vmat, t_method='mean', model_num=1,checkBeta_=False):
        if(checkBeta_):
            enc_features, _, _, betas = self.forward_BUTD(Vmat, t_method=t_method, model_num=model_num, checkBeta=checkBeta_)
            return enc_features, betas
        else:
            enc_features,_,_,_=self.forward_BUTD(Vmat,t_method=t_method,model_num=model_num, checkBeta=checkBeta_)
            return enc_features

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
        return Vmat, Umat

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

        return torch.matmul(UnCorr,Vmat), UnCorr

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

        return torch.matmul(UnCorr, Vmat), UnCorr

    def UnCorrVmat_tanh(self,Vmat):
        #pdb.set_trace()
        RightUnCorr=self.act_tanh(self.linear_U1(Vmat))
        LeftUnCorr = self.act_tanh(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))

        return torch.matmul(UnCorr,Vmat), UnCorr

    def UnCorrVmat_Lrelu(self,Vmat):
        #pdb.set_trace()
        RightUnCorr=self.act_Lrelu(self.linear_U1(Vmat))
        LeftUnCorr = self.act_Lrelu(self.linear_U2(Vmat))
        UnCorr=torch.matmul(LeftUnCorr,torch.transpose(RightUnCorr,1,2))

        return torch.matmul(UnCorr,Vmat), UnCorr



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab_size=vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.softmax=nn.Softmax(dim=1)
        
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

    
    def sample(self, features, states=None, model_num=1):
        """Generate captions for given image features using greedy search."""
        if(model_num < 7):
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
        else:
            return self.sample2(features)

    def sample2(self, features, input=1):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        #pdb.set_trace()
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
                predicted=torch.cuda.LongTensor(features.size(0)).fill_(1) # '<start>'
            # elif (i == 1):
            #     answer_word=vocab_candidates[0]
            #
            #     predicted=torch.cuda.LongTensor([answer_word]) # '<start>'
            #     vocab_candidates.remove(answer_word)
            elif (i == 1):
                predicted=[]
                for b_idx in range(outputs.size(0)):
                    if(vocab_candidates[b_idx]):
                        outputs_candidate=outputs[b_idx, vocab_candidates[b_idx]]
                        _, output_best = outputs_candidate.max(0)
                        predicted.append(vocab_candidates[b_idx][output_best])
                    else:
                        outputs_candidate = outputs[b_idx]
                        _,predicted_id=outputs_candidate.max(0)
                        predicted.append(predicted_id)
                predicted = torch.cuda.LongTensor(predicted)
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
                outputs=self.softmax(outputs)
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

        atted_v_feats, logits, att = self.forward(v, b, q, t_method=t_method, x_method=x_method,
                                                  s_method=s_method)
        encoded_features = self.encoder(atted_v_feats, t_method)


        if(s_method == 'BestOne'):
            if (model_num < 7):
                Generated_Captions=self.decoder.sample(encoded_features)
            else:
                input_=self.vocab('<start>')
                Generated_Captions = self.decoder.sample2(encoded_features,input=input_)
        elif(s_method == 'BeamSearch'):
            Generated_Captions = self.decoder.BeamSearch(encoded_features,NumBeams=3)

        return Generated_Captions, logits, att

    def generate_explain(self,Vmat, encoded_features, vocab_candidates, t_method='mean',x_method='sum', s_method='BestOne',isBUTD=False , isUnion=False, model_num=1):
        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']

        if (s_method == 'BestOne'):
            if (model_num < 7):
                if(isBUTD):
                    Generated_Explains = self.decoder.sample_with_answer_BUTD(Vmat, encoded_features, vocab_candidates, isUnion=isUnion)
                    # Vmat, union_vfeats, vocab_candidates, isUnion=False,
                else:
                    Generated_Explains = self.decoder.sample_with_answer(encoded_features, vocab_candidates)
            else:
                input_ = self.vocab('<start>')
                _= self.decoder.sample2(encoded_features, input=input_)
        elif (s_method == 'BeamSearch'):
            _ = self.decoder.BeamSearch(encoded_features, NumBeams=3)

        return Generated_Explains

    def generate_caption_n_context(self, v, b, q, t_method='mean',x_method='sum', s_method='BestOne',isBUTD=False , isUnion=False, model_num=1, useVQA=False, checkBeta=False):

        assert x_method in ['sum', 'mean', 'sat_cut', 'top3', 'top3_sat', 'weight_only', 'NoAtt']
        assert s_method in ['BestOne', 'BeamSearch']



        if(isBUTD):
            if(useVQA):
                atted_v_feats, logits, att = self.forward(v, b, q, t_method=t_method, x_method=x_method,
                                                             s_method=s_method)
            else:
                atted_v_feats=v
                logits=None
                att=None

            encoded_features, union_vfeats, atted_v_feats, betas = self.encoder.forward_BUTD(atted_v_feats, t_method=t_method,
                                                                        model_num=model_num, isUnion=isUnion, checkBeta=True)

            Generated_Captions = self.decoder.sample(atted_v_feats, union_vfeats, isUnion=False, isDuplicate=False)
            if(checkBeta):
                return Generated_Captions, logits, att, union_vfeats, atted_v_feats, betas
            else:
                return Generated_Captions, logits, att, union_vfeats, atted_v_feats # atted_v_feats = Vmat
        else:
            if (useVQA):
                atted_v_feats, logits, att = self.forward(v, b, q, t_method=t_method, x_method=x_method,
                                                            s_method=s_method)
            else:
                atted_v_feats=v
                logits=None
                att=None

            encoded_features, betas = self.encoder(atted_v_feats, t_method, checkBeta_=True)
            if(s_method == 'BestOne'):
                Generated_Captions=self.decoder.sample(encoded_features, model_num=model_num)
            elif(s_method == 'BeamSearch'):
                Generated_Captions = self.decoder.BeamSearch(encoded_features,NumBeams=3)

            if (checkBeta):
                return Generated_Captions, logits, att, encoded_features, None, betas
            else:
                return Generated_Captions, logits, att, encoded_features, None


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


        return atted_v_feats, logits, att





def max_k(inputTensor,dim_=0,k=1):
    _, idx_att = torch.sort(inputTensor, dim=dim_, descending=True)

    if(dim_ != 0):
        idx_att=torch.transpose(idx_att,0,dim_)
        inputTensor=torch.transpose(inputTensor,0,dim_)
    idx_att=idx_att[:k]
    #outputTensor = inputTensor[idx_att.squeeze()]
    outputTensor = torch.gather(inputTensor,0,idx_att)
    if (dim_ != 0):
        idx_att = torch.transpose(idx_att, 0, dim_)
        outputTensor = torch.transpose(outputTensor, 0, dim_)
    return outputTensor, idx_att

def max_k_NoDuplicate(inputTensor, sample_ids,dim_=0,k=1):
    if(len(sample_ids) > 0):
        tmp_sample_ids=[sample_ids[-1]]
    else:
        tmp_sample_ids=sample_ids
    word_domain_size=len(tmp_sample_ids)+k

    Probs, idx_outs = max_k(inputTensor, dim_=dim_, k=word_domain_size)
    Max_k_idx = idx_outs[:, :word_domain_size]
    duplicate_list=[]
    for j in range(word_domain_size):
        predicted = Max_k_idx[:, j]
        duplicate_idx=batchwise_in(predicted, tmp_sample_ids)
        duplicate_list.append(duplicate_idx)

    #print(duplicate_list)
    #print(Max_k_idx[duplicate_idx,:])
    #print(Max_k_idx[duplicate_list[2], :])
    Max_k_idx=filter_out_duplicate(Max_k_idx,duplicate_list)
    #print(Max_k_idx[duplicate_idx, :])
    #print(Max_k_idx[duplicate_list[2], :])
    #pdb.set_trace()
    Max_k_idx=Max_k_idx[:,:k] # Non-duplicative(toward tmp_sample_ids) top-k indicies
    return torch.gather(inputTensor,1,Max_k_idx), Max_k_idx

def batchwise_in(batch_elem, batch_list):
    if not batch_list: # if empty list
        return []
    list_len=len(batch_list)
    is_match_=[]
    for k in range(list_len):
        is_match_.append(batch_elem==batch_list[k])
    is_match_=torch.stack(is_match_,1)
    find_idx=is_match_.sum(1)

    return find_idx.nonzero()

def filter_out_duplicate(inputTensor, filter):
    filt_len=len(filter)
    for i in range(filt_len-2,-1,-1):
        for j in range(i,filt_len-1):
            inputTensor[filter[i],j]=inputTensor[filter[i],j+1]
    return inputTensor


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
    def __init__(self, embed_size, vdim, hidden_size1, hidden_size2, vocab_size, num_layers, max_seq_length=20, paramH=256, dropout=0.5):
        """Set the hyper-parameters and build the layers."""
        super(DecoderTopDown, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size2=hidden_size2
        self.hidden_size1 = hidden_size1
        self.TopDownAttentionLSTM = nn.LSTM(hidden_size2+vdim+embed_size, hidden_size1, num_layers, batch_first=True)
        self.LanguageLSTM = nn.LSTM(hidden_size1+vdim, hidden_size2, num_layers, batch_first=True)

        self.max_seg_length = max_seq_length
        self.softmax = nn.Softmax(dim=1)
        self.paramH=paramH

        self.linear_Wva=weight_norm(nn.Linear(vdim,self.paramH ))
        self.linear_Wha = weight_norm(nn.Linear(hidden_size1,self.paramH ))
        self.linear_wa = weight_norm(nn.Linear(self.paramH, 1 ))
        self.linear = weight_norm(nn.Linear(hidden_size2, vocab_size))
        self.linear_mid = weight_norm(nn.Linear(hidden_size1, vocab_size))

        self.dropout = nn.Dropout(p=dropout)

        self.act_tanh=nn.Tanh()

    def forward(self, Vmat, enc_features, union_vfeats, captions, lengths, memory_save=False, isUnion=False):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        batch_size=captions.size(0)

        max_seq_length=max(lengths)

        hidden2=torch.cuda.FloatTensor(batch_size, self.hidden_size2).fill_(0)

        states1=None
        states2=[hidden2.unsqueeze(0), hidden2.unsqueeze(0)]


        # if(isUnion):
        #     # pseudo(Vmat_transpose) -> pVmat
        #     pVmat=[]
        #     for i in range(batch_size):
        #         tmp = torch.transpose(Vmat[i, :, :], 0, 1)
        #         pVmat.append(torch.pinverse(tmp))
        #     pVmat=torch.stack(pVmat,0);
        #     beta=torch.matmul(pVmat,union_vfeats.unsqueeze(2))
        #     Vmat=beta*Vmat


        outputs=[]
        mid_outs=[]
        for i in range(max_seq_length):
            valid_outputs, hidden2, states1, states2, mid_outputs = self.BUTD_LSTM_Module(Vmat, hidden2, union_vfeats, embeddings[:, i, :],
                                                                             states1=states1, states2=states2)

            outputs.append(valid_outputs)
            mid_outs.append(mid_outputs)

            # input1 = torch.cat([hidden2, union_vfeats,
            #                     embeddings[:, i, :]], 1)
            # input1 = input1.unsqueeze(1)
            #
            # hidden1, states1 = self.TopDownAttentionLSTM(input1, states1)
            #
            # atten_logit = self.linear_wa(
            #     self.act_tanh(self.linear_Wva(Vmat) + self.linear_Wha(hidden1)))
            # atten_logit = atten_logit.squeeze(2)
            # atten = self.softmax(atten_logit)
            # atten = atten.unsqueeze(1)
            # atten_vfeats = torch.matmul(atten, Vmat)
            # input2 = torch.cat([atten_vfeats, hidden1], 2)
            # hidden2, states2 = self.LanguageLSTM(input2, states2)
            #
            # valid_outputs = self.linear(hidden2)  # teacher forcing 방식
            # hidden2 = hidden2.squeeze(1)
            # outputs.append(valid_outputs.squeeze(1))
        outputs=torch.stack(outputs,1)
        mid_outs=torch.stack(mid_outs,1)
        return outputs, mid_outs
        #hidden2_out_prev=

        #hiddens, _ = self.lstm(packed)

    def sample(self, Vmat, union_vfeats, isUnion=False, isDuplicate=True):
        """Generate captions for given image features using greedy search."""
        batch_size = Vmat.size(0)
        hidden2 = torch.cuda.FloatTensor(batch_size, self.hidden_size2).fill_(0)
        states2 = [hidden2.unsqueeze(0), hidden2.unsqueeze(0)]
        states1 = None

        # if (isUnion):
        #     # pseudo(Vmat_transpose) -> pVmat
        #     pVmat = []
        #     for i in range(batch_size):
        #         tmp = torch.transpose(Vmat[i, :, :], 0, 1)
        #         pVmat.append(torch.pinverse(tmp))
        #     pVmat = torch.stack(pVmat, 0);
        #     beta = torch.matmul(pVmat, union_vfeats.unsqueeze(2))
        #     Vmat = beta * Vmat

        sampled_ids = []
        input = self.embed(torch.cuda.LongTensor(batch_size).fill_(1)) # [1] = <sos>
        for i in range(self.max_seg_length):
            #pdb.set_trace()
            valid_outputs, hidden2, states1, states2, _ = self.BUTD_LSTM_Module(Vmat, hidden2, union_vfeats, input,
                                                                             states1=states1, states2=states2, isSample=True)

            # _, predicted = valid_outputs.max(1)  # predicted: (batch_size)

            #prevent duplicate elements in a list
            if(isDuplicate):
                _, idx_outs = max_k(valid_outputs, dim_=1, k=1)  # predicted: (batch_size, NumBeams), tmp_probs: (batch_size, NumBeams)
            else:
                _, idx_outs = max_k_NoDuplicate(valid_outputs, sampled_ids, dim_=1,
                                    k=1)  # predicted: (batch_size, NumBeams), tmp_probs: (batch_size, NumBeams)
            predicted=idx_outs[:,0]

            sampled_ids.append(predicted)
            input = self.embed(predicted)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


    def sample_with_answer_BUTD(self, Vmat, union_vfeats, vocab_candidates, isUnion=False, states=None):
        """Generate captions for given image features using greedy search."""

        batch_size = Vmat.size(0)
        hidden2 = torch.cuda.FloatTensor(batch_size, self.hidden_size2).fill_(0)
        states2 = [hidden2.unsqueeze(0), hidden2.unsqueeze(0)]
        states1 = None

        sampled_ids = []
        input = self.embed(torch.cuda.LongTensor([1]))  # [1] = <sos>
        for i in range(self.max_seg_length):
            #pdb.set_trace()
            valid_outputs, hidden2, states1, states2, _ = self.BUTD_LSTM_Module(Vmat, hidden2, union_vfeats, input, states1=states1, states2=states2, isSample=True)

            _, idx_outs = max_k_NoDuplicate(valid_outputs, sampled_ids, dim_=1,
                                            k=1)  # predicted: (batch_size, NumBeams), tmp_probs: (batch_size, NumBeams)
            predicted = idx_outs[:, 0]

            if (i == 0): # BUTD not starts with <sos>
                if(vocab_candidates):
                    outputs_candidate = valid_outputs[0, vocab_candidates]
                    _, output_best = outputs_candidate.max(0)
                    predicted = vocab_candidates[output_best]
                else:
                    outputs_candidate = valid_outputs[0]
                    _,predicted=outputs_candidate.max(0)


                predicted = torch.cuda.LongTensor([predicted])
            sampled_ids.append(predicted)

            input = self.embed(predicted)  # inputs: (batch_size, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def BUTD_LSTM_Module(self, Vmat, init_hidden2, vfeats, init_input, states1=None, states2=None, isSample=False):
        hidden2= init_hidden2
        input1 = torch.cat([hidden2, vfeats,
                            init_input], 1)
        input1 = input1.unsqueeze(1)

        hidden1, states1 = self.TopDownAttentionLSTM(input1, states1)

        if(isSample):
            atten_logit = self.linear_wa(self.act_tanh(self.linear_Wva(Vmat) + self.linear_Wha(hidden1)))
            atten_logit = atten_logit.squeeze(2)
            atten = self.softmax(atten_logit)
            atten = atten.unsqueeze(1)
            atten_vfeats = torch.matmul(atten, Vmat)
            input2 = torch.cat([atten_vfeats, hidden1], 2)
            hidden2, states2 = self.LanguageLSTM(input2, states2)

            valid_outputs = self.linear(hidden2)  # teacher forcing 방식
            mid_outputs = self.linear_mid(hidden1)
        else:
            atten_logit = self.linear_wa(self.dropout(
                self.act_tanh(self.linear_Wva(Vmat) + self.linear_Wha(hidden1))))
            atten_logit = atten_logit.squeeze(2)
            atten = self.softmax(atten_logit)
            atten = atten.unsqueeze(1)
            atten_vfeats = torch.matmul(atten, Vmat)
            input2 = torch.cat([atten_vfeats, hidden1], 2)
            hidden2, states2 = self.LanguageLSTM(input2, states2)

            valid_outputs = self.linear(self.dropout(hidden2))  # teacher forcing 방식
            mid_outputs = self.linear_mid(self.dropout(hidden1))

        hidden2 = hidden2.squeeze(1)
        valid_outputs = valid_outputs.squeeze(1)
        mid_outputs=mid_outputs.squeeze(1)
        return valid_outputs, hidden2, states1, states2, mid_outputs

    def BottomUpBeamAdapter(self, PackedArgs, Traj, NumBeams):
        '''

        :param PackedArgs: (Vmat, hidden2, meanVmat, input, states1_h, states1_c, states2_h, states2_c)
        :param Traj: batchwise-trajectory
        :param NumBeams:
        :return:
        '''


        if(Traj):
            batchsize=len(Traj)
            seq_len=len(Traj[0])
            Traj=[torch.cuda.LongTensor([Traj[x][y] for x in range(batchsize)]) for y in range(seq_len)]


        if (PackedArgs[4] is not None):
            valid_outputs, hidden2_tmp, states1_tmp, states2_tmp, _ = self.BUTD_LSTM_Module(PackedArgs[0], PackedArgs[1],
                                                                                         PackedArgs[2], PackedArgs[3],
                                                                                         [PackedArgs[4].unsqueeze(0),
                                                                                          PackedArgs[5].unsqueeze(0)],
                                                                                         [PackedArgs[6].unsqueeze(0),
                                                                                          PackedArgs[7].unsqueeze(0)], isSample=True)
        else:
            valid_outputs, hidden2_tmp, states1_tmp, states2_tmp, _ = self.BUTD_LSTM_Module(PackedArgs[0], PackedArgs[1],
                                                                                         PackedArgs[2], PackedArgs[3],
                                                                                         None,
                                                                                         [PackedArgs[6].unsqueeze(0),
                                                                                          PackedArgs[7].unsqueeze(0)], isSample=True)
        valid_outputs = self.softmax(valid_outputs)
        tmp_probs, predicted = max_k_NoDuplicate(valid_outputs, Traj, dim_=1,
                                                 k=NumBeams)  # predicted: (batch_size, NumBeams), tmp_probs: (batch_size, NumBeams)

        NewPacks=[[PackedArgs[0], hidden2_tmp, PackedArgs[2], self.embed(predicted[:,x]), states1_tmp[0].squeeze(0), states1_tmp[1].squeeze(0), states2_tmp[0].squeeze(0), states2_tmp[1].squeeze(0)] for x in range(NumBeams)]

        return torch.log(tmp_probs), predicted, NewPacks

    def BeamSearch2(self, Vmat, meanVmat, NumBeams=5, EOS_Token=99999):
        """Generate captions for given image features using greedy search."""
        batch_size = Vmat.size(0)
        hidden2 = torch.cuda.FloatTensor(batch_size, self.hidden_size2).fill_(0)
        states2 = [hidden2.unsqueeze(0), hidden2.unsqueeze(0)]
        states1 = None

        input = self.embed(torch.cuda.LongTensor([1]))  # [1] = <sos>
        input = input.repeat(batch_size, 1)

        if(states1):
            PackArgs=[Vmat, hidden2, meanVmat, input, states1[0].squeeze(0), states1[1].squeeze(0), states2[0].squeeze(0), states2[1].squeeze(0)]
        else:
            PackArgs = [Vmat, hidden2, meanVmat, input, None, None,
                        states2[0].squeeze(0), states2[1].squeeze(0)]

        return beam_decode(self.BottomUpBeamAdapter, PackArgs, NumBeams, 20, EOS_Token)



class BeamSearchNode(object):
    def __init__(self, PackedArgs, trajectory, wordId, logProb):
        '''
        :param PackedArgs (ex: hiddenstate) up to (parent node -> itself):
        :param trajectory up to parent node:
        :param wordId of itself:
        :param logProb up to parent node + logProb of itself:
        '''
        self.packed = PackedArgs
        self.traj = trajectory
        self.wordid = wordId
        self.logp = logProb

        self.reward = 0.0

    def ciderReward(self, reward):
        self.reward = reward

        return self.reward

    def getlogProbs(self):
        return self.logp

    def getPack(self):
        return self.packed

    def getTraj(self):
        return self.traj

class BatchBeamSearchNode(object):
    def __init__(self, Batch_PackedArgs, Batch_trajectory, Batch_wordId, Batch_logProb, Batch_logProbNode):
        '''
        :param PackedArgs (ex: hiddenstate) up to (parent node -> itself):
        :param trajectory up to parent node:
        :param wordId of itself:
        :param logProb up to parent node:
        :param logProb of itself:
        '''

        self.BatchSize=Batch_logProbNode.size(0)
        self.BatchNodes=[[] for x in range(self.BatchSize)]

        self.Batch_trajectory=deepcopy(Batch_trajectory)

        if (not(self.Batch_trajectory)):
            self.Batch_trajectory=[[Batch_wordId[x].item()] for x in range(self.BatchSize)]
        else:
            for batch_idx in range(self.BatchSize):
                self.Batch_trajectory[batch_idx].append(Batch_wordId[batch_idx].item())

        self.Batch_logp = Batch_logProb + Batch_logProbNode
        self.Batch_wordId=Batch_wordId

        for n_ in range(self.BatchSize):
            tmp_Pack=[Arg[n_] for Arg in Batch_PackedArgs]
            self.BatchNodes[n_]=BeamSearchNode(tmp_Pack, self.Batch_trajectory[n_], Batch_wordId[n_], self.Batch_logp[n_])


    def getlogProbs(self):
        return self.Batch_logp

    def getNodes(self):
        return self.BatchNodes

    def getTraj(self):
        return self.Batch_trajectory

    def setNodes(self, index, Node):
        self.BatchNodes[index]=Node
        self.Batch_logp[index]=Node.getlogProbs()
        self.Batch_trajectory[index]=Node.getTraj()


class BeamQueue:
    def __init__(self, maxSize):
        self.maxSize=maxSize
        self.items=[] # [(logProb, node), (logProb, node) ...]

    def isEmpty(self):
        return self.items == []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        tmp=[el[1] for el in self.items[index]]
        return tmp # return node

    def insert(self, data):


        if(self.__len__() >= self.maxSize):

            batch_size = len(data)
            q_idx=0

            for batch_idx in range(batch_size):

                tmp_Traj = [tl[batch_idx][1].getTraj() for tl in self.items]  # if there exists a duplicate trajectory in que
                data_tmp_Traj=data[batch_idx][1].getTraj()
                if (data_tmp_Traj in tmp_Traj):
                    t_idx=tmp_Traj.index(data_tmp_Traj)
                    if(self.items[t_idx][batch_idx][0] < data[batch_idx][0]):
                        self.items[t_idx][batch_idx][0]=-9999 # if duplicate trajectory and new data has higher probability, then remove the old node from the queue
                    else:
                        continue

                for que_idx in range(self.maxSize): # if no duplicate trajectory in que
                    if (self.items[que_idx][batch_idx][0] < data[batch_idx][0]):
                        for tmp_que_idx in range(self.maxSize-2, que_idx-1, -1): # prepare for new order
                            self.items[tmp_que_idx+1][batch_idx]=self.items[tmp_que_idx][batch_idx]
                        q_idx=que_idx # replace based on probability order
                        break
                    else:
                        q_idx=que_idx+1
                if(q_idx < self.maxSize): # update index is in range
                    self.items[q_idx][batch_idx] = data[batch_idx]

        else:
            self.items.append(data) # data=> batch probability
            batch_size=len(data)
            queue_len=self.__len__()

            beam_data=[[self.items[x][y] for x in range(queue_len)] for y in range(batch_size)]
            for beam_idx in range(len(beam_data)):
                beam_data[beam_idx]=sorted(beam_data[beam_idx], key=lambda x: int(x[0]), reverse=True)
            self.items=[[[beam_data[y][x][0],
                          beam_data[y][x][1]] for y in range(batch_size)] for x in range(queue_len)]


def beam_decode(BeamNodeAdapter, PackedArguments, NumBeams, MaxSeqLength, EOS_Token, debug=False):

    '''

    :param BeamNodeAdapter:  BeamNode adapter for several different types of decoder
    :param Decoder:     RNNdecoder (ex: BUTD)
    :param PackedArguments:     (Inputs, States, etc)
    :param NumBeams:    (Number of Beams)
    :param MaxSeqLength:    (Maximum Sequence Length)
    :param EOS_Token:   (EOS tOKEN)
    :return:
    '''

    Trajs=[[] for x in range(NumBeams)]
    logProbs_Trajs=[0.0 for x in range(NumBeams)]
    template=[type(Arg_) for Arg_ in PackedArguments]


    for t_step in range(MaxSeqLength):
        if(debug):
            print('time step is {}'.format(t_step))

        bestNodes=BeamQueue(NumBeams)

        for beam_idx in range(NumBeams):


            if(t_step == 0):
                logProbs, Words, NewPackedArgs = BeamNodeAdapter(PackedArguments, Trajs[beam_idx], NumBeams)
            else:
                logProbs, Words, NewPackedArgs = BeamNodeAdapter(PackedArguments[beam_idx], Trajs[beam_idx], NumBeams)
                batch_size = logProbs.size(0)
                for batch_idx in range(batch_size):
                    if(Trajs[beam_idx][batch_idx][-1]==EOS_Token):
                        Words[batch_idx].fill_(-1)
                        logProbs[batch_idx].fill_(-9999)
                        Words[batch_idx][0]=EOS_Token
                        logProbs[batch_idx][0]=0.0



            for n_ in range(NumBeams):
                # if(t_step > 1):
                #     pdb.set_trace()
                node_Candidate=BatchBeamSearchNode(NewPackedArgs[n_], Trajs[beam_idx], Words[:,n_], logProbs_Trajs[beam_idx], logProbs[:,n_])
                node_data=[node_Candidate.getlogProbs(), node_Candidate.getNodes()]
                batch_size = len(node_data[0])
                node_data = [[node_data[0][y], node_data[1][y]] for y in range(batch_size)]
                bestNodes.insert(node_data)


        batch_size=len(bestNodes[0])

        PackedArguments = [[bestNodes[x][batch_idx].getPack() for batch_idx in range(batch_size)] for x in range(NumBeams)]
        Trajs = [[bestNodes[x][batch_idx].getTraj() for batch_idx in range(batch_size)] for x in range(NumBeams)]
        logProbs_Trajs = [torch.cuda.FloatTensor([bestNodes[x][batch_idx].getlogProbs() for batch_idx in range(batch_size)]) for x in range(NumBeams)]

        PackedArguments=[ [ [PackedArguments[beam_idx][batch_idx][arg_idx] for batch_idx in range(batch_size)] for arg_idx in range(len(template))] for beam_idx in range(NumBeams)]


        for A_beam_PackedArguments in PackedArguments:
            # beam // batch // argu
            for arg_idx in range(len(template)):
                if(template[arg_idx]==torch.Tensor or template[arg_idx] == type(None)):
                    A_beam_PackedArguments[arg_idx]=torch.stack(A_beam_PackedArguments[arg_idx],0)

    Trajs=[torch.cuda.LongTensor(Traj) for Traj in Trajs]
    return torch.stack(Trajs,2)


def sectionwise_Sum(samples, section_lengths): # Per-Sample Average Pooling

    assert (samples.dim() == 1)

    sectionWiseList=[]

    s=0
    e=0
    for sec_length in section_lengths:
        e += sec_length
        sectionWiseList.append(samples[s:e])
        s=e+1

    secWisePool=[torch.sum(section) for section in sectionWiseList]
    return torch.stack(secWisePool,0)