import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import BottomUp_get_loader
from build_vocab import Vocabulary
from model import Encoder_HieStackedCorr, DecoderTopDown
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import pdb
import utils_hsc as utils



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if(args.t_method=='uncorr'):
        args.isUnion=True
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) #saving model directory
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)



    # Build data loader
    coco_cap_train_path='D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_train2014.json'
    coco_cap_val_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_val2014.json'
    data_loader = BottomUp_get_loader('train+val', [coco_cap_train_path, coco_cap_val_path], vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)



    # data_loader.dataset[i] => tuple[[object1_feature #dim=2048] [object2_..] [object3_...] ...], tuple[[object1_bbox #dim=6] [object2_...] [object3_...] ...], caption]

    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size,2048, model_num=args.model_num, LRdim=args.LRdim).to(device)
    decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers).to(device)



    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # if(args.t_method == 'mean'):
    #     params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # elif(args.t_method == 'uncorr'):
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(
        encoder.linear_U1.parameters()) + list(encoder.linear_U2.parameters())
    #params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)

    epoch_start=0

    if (args.checkpoint_dir != 'None'):
        model_hsc_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            args.checkpoint_dir)
        model_hsc_data=torch.load(model_hsc_path)
        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])
        optimizer.load_state_dict(model_hsc_data['optimizer_state'])
        epoch_start=model_hsc_data['epoch']+1


    if not os.path.exists(os.path.join(args.model_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim))):
        os.makedirs(os.path.join(args.model_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim)))
    for epoch in range(epoch_start,args.num_epochs):
        #for i, (images, captions, lengths) in enumerate(data_loader):

        for i, (features, spatials, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            # if(args.model_num > 6):
            #     lengths[:]=[x-1 for x in lengths] #어차피 <sos> 나 <eos> 둘중 하나는 양쪽 (target, input)에서 제거된다.
            #     targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            # else:
            #     targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            lengths[:] = [x - 1 for x in lengths]  # 어차피 <sos> 나 <eos> 둘중 하나는 양쪽 (target, input)에서 제거된다.
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            features = features.cuda()
            captions = captions.cuda()
            targets=targets.cuda()


            # Forward, backward and optimize
            features_encoded,union_vfeats, UMat=encoder.forward_BUTD(features,t_method=args.t_method,model_num=args.model_num, isUnion=args.isUnion)
            if(args.isUnion):
                dummy_input=torch.eye(features.size(1)) # number of objects
                dummy_input=dummy_input.unsqueeze(0)
                dummy_input=dummy_input.repeat(features.size(0),1,1)
                dummy_input=dummy_input.cuda()
                betas=torch.mean(torch.matmul(UMat,dummy_input),1)
                betas=betas.unsqueeze(2)
                features=betas*features


            outputs = decoder(features, features_encoded, union_vfeats, captions, lengths)

            #pdb.set_trace()
            outputs=pack_padded_sequence(outputs,lengths,batch_first=True)[0]
            loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        model_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(epoch + 1))
        utils.save_model(model_path, encoder, decoder, epoch, optimizer)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_BUTD/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--checkpoint_dir', type=str, default='None', help='loading from this checkpoint')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--t_method', type=str, default='uncorr')
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--isUnion', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)