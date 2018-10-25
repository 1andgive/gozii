import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

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
    def __init__(self, embed_size, vdim, num_stages=5):
        super(Encoder_HieStackedCorr, self).__init__()
        self.linear=nn.Linear(vdim,embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.num_stages=num_stages
    def forward(self, Vmat):

        features = self.bn(self.linear(self.MeanVmat(Vmat)))
        return features

    def UnCorrelatedResidualHierarchy(self, num_stages, Vmat):
        ##building Un-Correlated Residual Hierarchy for Visual Matrix
        num_batches = Vmat.size(0)
        num_objects = Vmat.size(1)
        Umat = torch.zeros(num_batches, num_objects, num_objects).cuda()  # initial Umat = zero matrix

        for i in range(num_stages):
            Vmat = Vmat + torch.matmul(Umat, Vmat)  # Vmat = V' (transposed version of V
            psi = torch.matmul(Vmat, torch.transpose(Vmat, 1, 2))

            Diag_Tensor = torch.zeros(num_batches, psi.size(1)).cuda()
            for k in range(num_batches):
                Diag_Tensor[k] = torch.rsqrt(torch.diag(psi[k]) + 1e-6)
            Corr = Diag_Tensor.unsqueeze(1) * psi * Diag_Tensor.unsqueeze(2)
            Umat = 1 - Corr
        v_final = torch.sum(Vmat, 1) / num_objects
        return v_final

    def MeanVmat(self,Vmat):
        v_final=torch.mean(Vmat,1)
        return v_final




class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0]) #teacher forcing 방식
        #pdb.set_trace()
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

class BAN_HSC(nn.Module):
    def __init__(self,BAN,encoder,decoder):
        super(BAN_HSC,self).__init__()
        self.BAN=BAN
        self.encoder=encoder
        self.decoder=decoder

    def generate_caption(self, v, b, q, labels):
        logits, att=self.BAN(v,b,q,None)
        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                               2).unsqueeze(2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]
        #pdb.set_trace()
        atted_v_feats = att_for_v * v  # attended visual features

        encoded_features=self.encoder(atted_v_feats)
        Generated_Captions=self.decoder.sample(encoded_features)
        return Generated_Captions, logits, att

    def forward(self, v, b, q, labels):
        _, att=self.BAN(v,b,q,None)
        att_final = att[:, -1, :, :]
        # 원래는 q_net(q)와 v_net(v)가 Att matrix의 양 끝에 Matrix-Multiplication 된다.
        att_for_v = torch.sum(att_final,
                               2)  # average over question words (phrase-level inspection, each index for q in final stage does not represent word anymore (it rather represents something semantically more meaningful)
        # att_for_v dimension => [b, v_feature_dim, 1]

        atted_v_feats = att_for_v * v  # attended visual features

        encoded_features=self.encoder(atted_v_feats)

        return True
