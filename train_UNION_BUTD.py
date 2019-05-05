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
from address_server_XAI import *
import progressbar
import sys

# union이라고 이름 붙인거는 self-attention. butd를 하는 부분. 
#bottom up을 사용. paper에 안나온것도 알수있음~~~~~~~~


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
    coco_cap_train_path=addr_coco_cap_train_path
    coco_cap_val_path = addr_coco_cap_val_path
    data_loader = BottomUp_get_loader('train+val', [coco_cap_train_path, coco_cap_val_path], vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, adaptive=args.isAdaptive)



    # data_loader.dataset[i] => tuple[[object1_feature #dim=2048] [object2_..] [object3_...] ...], tuple[[object1_bbox #dim=6] [object2_...] [object3_...] ...], caption]

    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size,2048, model_num=args.model_num, LRdim=args.LRdim)
    decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers, paramH=args.paramH)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)

    if(torch.cuda.device_count() > 1):
        encoder=nn.DataParallel(encoder)
        decoder=nn.DataParallel(decoder)
        criterion=nn.DataParallel(criterion)
    encoder.to(device)
    decoder.to(device)



    # Loss and optimizer
    
    # if(args.t_method == 'mean'):
    #     params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # elif(args.t_method == 'uncorr'):
    params = list(decoder.parameters()) + list(encoder.parameters())
    #params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)

    epoch_start=0
    
    print('t_method : {}, isUnion : {}, model_num : {}'.format(args.t_method, args.isUnion, args.model_num))

    if (args.checkpoint_dir != 'None'):
        model_hsc_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            args.checkpoint_dir)
        print('@@@@@@@ LOADING FROM CHECKPOINT : {}'.format(model_hsc_path))
        model_hsc_data=torch.load(model_hsc_path)
        if (torch.cuda.device_count() > 1):
            encoder.module.load_state_dict(model_hsc_data['encoder_state'])
            decoder.module.load_state_dict(model_hsc_data['decoder_state'])
        else:
            encoder.load_state_dict(model_hsc_data['encoder_state'])
            decoder.load_state_dict(model_hsc_data['decoder_state'])
        #optimizer.load_state_dict(model_hsc_data['optimizer_state'])
        epoch_start=model_hsc_data['epoch']+1
    else:
        model_hsc_path=os.path.join(
            'models', args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-16.pth')
        print('@@@@@@@ LOADING ENCODER FROM : {}'.format(model_hsc_path))
        model_hsc_data = torch.load(model_hsc_path)
        if (torch.cuda.device_count() > 1):
            encoder.module.load_state_dict(model_hsc_data['encoder_state'])
        else:
            encoder.load_state_dict(model_hsc_data['encoder_state'])


    if not os.path.exists(os.path.join(args.model_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim))):
        os.makedirs(os.path.join(args.model_path,args.t_method,'model{}_LR{}'.format(args.model_num,args.LRdim)))
    N=args.num_epochs * len(data_loader)
    bar = progressbar.ProgressBar(maxval=N).start()
    i_train=0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    for epoch in range(epoch_start,args.num_epochs):
        #for i, (images, captions, lengths) in enumerate(data_loader):

        for i, (features, spatials, captions, lengths, num_objs) in enumerate(data_loader):
            bar.update(i_train)
            # Set mini-batch dataset
            # if(args.model_num > 6):
            #     lengths[:]=[x-1 for x in lengths] #어차피 <sos> 나 <eos> 둘중 하나는 양쪽 (target, input)에서 제거된다.
            #     targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            # else:
            #     targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            lengths[:] = [x - 1 for x in lengths]  # 어차피 <sos> 나 <eos> 둘중 하나는 양쪽 (target, input)에서 제거된다.
            targets_tmp=captions[:, 1:]
            targets = pack_padded_sequence(targets_tmp, lengths, batch_first=True)[0]

            features = features.cuda()
            captions = captions.cuda()
            num_objs= num_objs.cuda()
            targets=targets.cuda()

            # Forward, backward and optimize
            if(torch.cuda.device_count() > 1):
                features_encoded,union_vfeats, features, betas=encoder.module.forward_BUTD(features,t_method=args.t_method,model_num=args.model_num, isUnion=args.isUnion, obj_nums=num_objs)
            else:
                features_encoded,union_vfeats, features, betas=encoder.forward_BUTD(features,t_method=args.t_method,model_num=args.model_num, isUnion=args.isUnion, obj_nums=num_objs)

            outputs, mid_outs = decoder(features, features_encoded, union_vfeats, captions, lengths)
            #print('output b size: {}, lengths b size : {}'.format(outputs.size(0),len(lengths)))
            #pdb.set_trace()
            mid_outs=mid_outs.max(1)[0]

            targets_d = torch.zeros(mid_outs.size(0), mid_outs.size(1)).to(device)
            targets_d.fill_(-1)

            for length in lengths:
                targets_d[:, :length - 1] = targets_tmp[:, :length - 1]

            outputs=pack_padded_sequence(outputs,lengths,batch_first=True)[0]
            loss_ce = criterion(outputs, targets)
            loss_dis = criterion_dis(mid_outs, targets_d.long())
            loss = loss_ce + (10 * loss_dis)

            decoder.zero_grad()
            encoder.zero_grad()
            if(torch.cuda.device_count() > 1):
                loss=loss.mean()
            loss.backward()

            # Clip gradients when they are getting too large
            torch.nn.utils.clip_grad_norm_(params, 0.25)

            optimizer.step()
            if(epoch > 40):
                scheduler.step()

            i_train+=1

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                sys.stdout.flush()

        # Save the model checkpoints
        model_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(epoch + 1))
        prev_model_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}.pth'.format(epoch-3))
        utils.save_model(model_path, encoder, decoder, epoch, optimizer)
        if(os.path.exists(prev_model_path)):
            os.remove(prev_model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_BUTD_36/standard_vocab/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_standard_train_val.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--checkpoint_dir', type=str, default='None', help='loading from this checkpoint')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=1000, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1000, help='dimension of lstm hidden states')
    parser.add_argument('--paramH', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--t_method', type=str, default='uncorr')
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--isUnion', type=bool, default=False)
    parser.add_argument('--isAdaptive', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)