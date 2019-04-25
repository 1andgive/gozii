import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import BottomUp_get_loader
from build_vocab import Vocabulary
from model import Encoder_HieStackedCorr, DecoderTopDown, sectionwise_Sum
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
import pdb
import utils_hsc as utils
from address_server_XAI import *
import progressbar
from pycocotools.coco import COCO
from tokenizer.ptbtokenizer import PTBTokenizer
import sys


from cider.cider import CiderScorer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_mask(idx, batch_size, label_size, maxSeqLength=20):
    label_from_vqa=torch.cuda.FloatTensor(batch_size, maxSeqLength, label_size).fill_(0)
    for seq in range(maxSeqLength):
        label_from_vqa[range(batch_size),seq,idx[:,seq]]=1
    return label_from_vqa

def caption_refine(explains, NumBeams=1, model_num=1):


    cap_list=[]
    for num_sen in range(NumBeams):
        if(NumBeams > 1):
            explain=explains[num_sen]
            x_caption = ''
        else:
            explain=explains
            x_caption = ''
        for word_ in explain:
            if word_ == '<start>':
                if model_num <7:
                    x_caption = ''
            elif word_ == '<end>':
                break
            else:
                x_caption = x_caption + ' ' + word_

        if (NumBeams > 1):
            cap_list.append([x_caption])

    if (NumBeams > 1):
        return cap_list
    else:
        return x_caption

def main(args):
    if (args.t_method == 'uncorr'):
        args.isUnion = True
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)  # saving model directory

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
    coco_cap_train_path = addr_coco_cap_train_path
    coco_cap_val_path = addr_coco_cap_val_path
    coco_Train=COCO(coco_cap_train_path)
    coco_Val = COCO(coco_cap_val_path)
    train_Ids= coco_Train.getImgIds()
    val_Ids = coco_Val.getImgIds()
    coco_Train_gts={}
    coco_Val_gts={}

    for imgId in train_Ids:
        coco_Train_gts[imgId] = coco_Train.imgToAnns[imgId]
    for imgId in val_Ids:
        coco_Val_gts[imgId] = coco_Val.imgToAnns[imgId]
    overall_gts = {}
    overall_gts.update(coco_Train_gts)
    overall_gts.update(coco_Val_gts)
    tokenizer = PTBTokenizer()
    overall_gts = tokenizer.tokenize(overall_gts)

    data_loader = BottomUp_get_loader('train+valCider', [coco_cap_train_path, coco_cap_val_path], vocab,
                                      transform, args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, adaptive=args.isAdaptive)


    # data_loader.dataset[i] => tuple[[object1_feature #dim=2048] [object2_..] [object3_...] ...], tuple[[object1_bbox #dim=6] [object2_...] [object3_...] ...], caption]

    # Build the models
    encoder = Encoder_HieStackedCorr(args.embed_size, 2048, model_num=args.model_num, LRdim=args.LRdim)
    decoder = DecoderTopDown(args.embed_size, 2048, args.hidden_size, args.hidden_size, len(vocab), args.num_layers,
                             paramH=args.paramH)

    SoftMax_=nn.Softmax(dim=1)

    if (torch.cuda.device_count() > 1):
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    # Loss and optimizer

    # if(args.t_method == 'mean'):
    #     params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # elif(args.t_method == 'uncorr'):
    params = list(decoder.parameters())
    # params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)

    epoch_start = 0

    print('t_method : {}, isUnion : {}, model_num : {}'.format(args.t_method, args.isUnion, args.model_num))

    if (args.checkpoint_dir != 'None'):

        model_hsc_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            args.checkpoint_dir)
        print("loading from {}".format(model_hsc_path))
        model_hsc_data = torch.load(model_hsc_path)
        encoder.load_state_dict(model_hsc_data['encoder_state'])
        decoder.load_state_dict(model_hsc_data['decoder_state'])

    if not os.path.exists(
            os.path.join(args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim))):
        os.makedirs(os.path.join(args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim)))
    N = args.num_epochs * len(data_loader)
    bar = progressbar.ProgressBar(maxval=N).start()
    i_train = 0

    outputs_cache={}

    EOS_Token_=vocab('<end>')
    SOS_Token_=vocab('<start>')

    for epoch in range(args.num_epochs):
        # for i, (images, captions, lengths) in enumerate(data_loader):
        print('epoch start')
        tmp_len=0


        for i, (features, spatials, img_Ids) in enumerate(data_loader):
            bar.update(i_train)
            cider_scorer = CiderScorer(n=4, sigma=6.0)


            features = features.cuda()


            with torch.no_grad():
                # Forward, backward and optimize
                if (torch.cuda.device_count() > 1):
                    features_encoded, union_vfeats, features, _ = encoder.module.forward_BUTD(features, t_method=args.t_method,
                                                                                           model_num=args.model_num,
                                                                                           isUnion=args.isUnion)
                else:
                    features_encoded, union_vfeats, features, _ = encoder.forward_BUTD(features, t_method=args.t_method,
                                                                                    model_num=args.model_num,
                                                                                    isUnion=args.isUnion)

                1. ############################################ Q - V approach #########################################

                # if (epoch % args.SelfCriticFrequency == 0):  # periodically update
                #
                #     outputs = decoder.BeamSearch2(features, union_vfeats, NumBeams=args.NumBeams, EOS_Token=vocab('<end>'))
                #     for img_Id, output in zip(img_Ids, outputs):
                #         outputs_cache[img_Id]=output
                #     tmp_len += len(img_Ids)
                #     if i % args.log_step == 0:
                #         print(' {} of {} caption updated'.format(tmp_len, len(overall_gts)))
                #         sys.stdout.flush()
                #     continue
                #
                # else:
                #
                #     outputs = torch.stack([outputs_cache[img_Id] for img_Id in img_Ids],0)

                1. #################################### OR POLICY GRADIENT #############################################

                outputs = decoder.BeamSearch2(features, union_vfeats, NumBeams=args.NumBeams, EOS_Token=EOS_Token_)

                1. #####################################################################################################








                output_baseline=decoder.sample(features,union_vfeats)
                # print('output b size: {}, lengths b size : {}'.format(outputs.size(0),len(lengths)))
                pdb.set_trace()

                2. ################################################## RL HERE ##############################################
                caption_list=[]
                for batch_idx in range(outputs.size(0)):
                    beam_list = []
                    baseline=[]
                    for beam_idx in range(outputs.size(2)):
                        caption = [vocab.idx2word[outputs[batch_idx][w_idx][beam_idx].item()] for w_idx in
                                   range(outputs.size(1))]
                        beam_list.append(caption)
                        baseline = [vocab.idx2word[output_baseline[batch_idx][w_idx].item()] for w_idx in
                               range(output_baseline.size(1))]
                    beam_list=caption_refine(beam_list,NumBeams=args.NumBeams)
                    baseline=caption_refine(baseline)
                    beam_list.append([baseline])
                    caption_list.append(beam_list)



                # 1. CIDER REWARD
                for batch_idx in range(outputs.size(0)):

                    ref=overall_gts[img_Ids[batch_idx]]

                    for beam_idx in range(outputs.size(2)+1):
                        hypo = caption_list[batch_idx][beam_idx]
                        cider_scorer += (hypo[0], ref)
                (score, scores) = cider_scorer.compute_score('corpus')

                scores=torch.Tensor(scores).view(outputs.size(0), args.NumBeams + 1) # args.NumBeams <= NumBeams // 1 <=  baseline caption
                score_baseline=scores[:,args.NumBeams]
                score_beams=scores[:,:args.NumBeams]
                Reward_from_baseline = score_beams - score_baseline.unsqueeze(1)


                Reward_from_baseline, best_beam = torch.max(Reward_from_baseline, 1) # single-agent RL!!! only use the best-beam!!
                outputs=outputs[range(outputs.size(0)),  :, best_beam]

                captions=torch.cat( (torch.cuda.LongTensor(outputs.size(0), 1).fill_(SOS_Token_), outputs) , 1)
                targets=outputs
                new_length=torch.sum(captions != 2,1)
                new_length[new_length > 50] = 50 # 50 is max_seq_length




            3. ####################################### # 2. POLICY GRADIENT #############################################

                #target_packed=pack_padded_sequence(outputs, new_length, batch_first=True)[0]  # NEW GT from beam

            # single-agent RL!!! only use the best-beam!!
            output_logit, _=decoder(features, features_encoded,
                    union_vfeats, captions, new_length)

            new_length, o_idx = torch.sort(new_length, dim=0, descending=True)  # batch re-ordering
            targets = targets[o_idx]
            Reward_from_baseline = Reward_from_baseline[o_idx]
            output_logit=output_logit[o_idx]

            output_logit = \
            pack_padded_sequence(output_logit, new_length, batch_first=True)[
                0]  # new logit for optimization


            # tmp_loss=criterion(output_logit, target).to(device)

            mask = gen_mask(targets, targets.size(0), output_logit.size(1))
            mask = pack_padded_sequence(mask, new_length, batch_first=True)[0]
            output_logit=SoftMax_(output_logit)
            output_logit=torch.log(output_logit)
            tmp_loss = - output_logit * mask
            tmp_loss = torch.sum(tmp_loss, 1)

            tmp_loss = sectionwise_Sum(tmp_loss,new_length)
            deserved_samples= ( Reward_from_baseline > 0 )
            loss = torch.mean(Reward_from_baseline[deserved_samples].cuda() * tmp_loss[deserved_samples])

            decoder.zero_grad()
            if (torch.cuda.device_count() > 1):
                loss = loss.mean()
            pdb.set_trace()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            optimizer.step()

            3. ##########################################################################################################

            i_train += 1
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, Cider: {:1.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()), torch.mean(score_baseline)))
                sys.stdout.flush()

        # Save the model checkpoints
        model_path = os.path.join(
            args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
            'model-{}-CiderOpt-GradClip-{}.pth'.format(epoch + 1, args.grad_clip))
        utils.save_model(model_path, encoder, decoder, epoch, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_BUTD_36/standard_vocab/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--checkpoint_dir', type=str, default='model-40.pth', help='loading from this checkpoint')
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=1000, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1000, help='dimension of lstm hidden states')
    parser.add_argument('--paramH', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--t_method', type=str, default='mean')
    parser.add_argument('--LRdim', type=int, default=64)
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--NumBeams', type=int, default=5)
    parser.add_argument('--isUnion', type=bool, default=False)
    parser.add_argument('--SelfCriticFrequency', type=int, default=5)
    parser.add_argument('--isAdaptive', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)