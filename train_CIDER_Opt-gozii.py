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
import platform
import copy
import torch.nn.modules.distance as F
#=======================================================================================================================
if(platform.system() == 'Linux'):
    address_to_Yolov3='/mnt/server5_hard1/seungjun/VisualGrounding/PyTorch-YOLOv3_train/'
elif(platform.system() == 'Windows'):
    address_to_Yolov3='D:\\VisualGrounding/PyTorch-YOLOv3_train/'

sys.path.insert(0, address_to_Yolov3)
from object_detect_pop import Yolov3
#=======================================================================================================================



from cider.cider import CiderScorer

#self critical sequence training . reinforce 공부!!!!!!!!!!!!!
#butd나 fast rcn에 의해서 학습 된거에 부가적으로 붙음. 성능++,. 필 수 ! (최근기법)

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

    if (args.split=='test2014'):
        data_loader = BottomUp_get_loader('test2014', addr_coco_cap_test2014_path, vocab,
                                          transform, args.batch_size,
                                 shuffle=False, num_workers=0, adaptive=args.isAdaptive)
    elif(args.split=='val'):
        name='val2014'
        data_loader = BottomUp_get_loader(name, addr_coco_cap_val_path, vocab,
                                          transform, args.batch_size,
                                          shuffle=False, num_workers=0, adaptive=args.isAdaptive)


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
    N = len(data_loader)
    bar = progressbar.ProgressBar(maxval=N).start()
    i_train = 0

    outputs_cache={}

    EOS_Token_=vocab('<end>')
    SOS_Token_=vocab('<start>')


    coco_train_img_file_template='COCO_train2014_{0:012d}.jpg'
    coco_val_img_file_template = 'COCO_val2014_{0:012d}.jpg'
    coco_test_img_file_template = 'COCO_test2014_{0:012d}.jpg'

    coco_train_img_file_template=os.path.join(addr_train_imgs,coco_train_img_file_template)
    coco_val_img_file_template=os.path.join(addr_val_imgs,coco_val_img_file_template)
    coco_test_img_file_template=os.path.join(addr_test2014_imgs, coco_test_img_file_template)
    
    cosine_sim=F.CosineSimilarity()

    tmp_len=0
    best_captions_list = []
    img_id_list=[]

    for i, (features, spatials, img_Ids, obj_nums) in enumerate(data_loader):
        print('{} / {} samples being processed'.format(i_train,N))
        sys.stdout.flush()
        
        bar.update(i_train)
        
        cider_scorer = CiderScorer(n=4, sigma=6.0)


        features = features.cuda()
        obj_nums=obj_nums.cuda()

        with torch.no_grad():
            # Forward, backward and optimize
            if (torch.cuda.device_count() > 1):
                features_encoded, union_vfeats, features, _ = encoder.module.forward_BUTD(features, obj_nums=obj_nums, t_method=args.t_method,
                                                                                       model_num=args.model_num,
                                                                                       isUnion=args.isUnion)
            else:
                features_encoded, union_vfeats, features, _ = encoder.forward_BUTD(features, obj_nums=obj_nums, t_method=args.t_method,
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


            #output_baseline=decoder.sample(features,union_vfeats)
            # print('output b size: {}, lengths b size : {}'.format(outputs.size(0),len(lengths)))

            2. ################################################## RL HERE ##############################################
            caption_list=[]
            caption_list_word=[]
            for batch_idx in range(outputs.size(0)):
                beam_list_word = []
                baseline=[]
                for beam_idx in range(outputs.size(2)):
                    caption = [vocab.idx2word[outputs[batch_idx][w_idx][beam_idx].item()] for w_idx in
                               range(outputs.size(1))]
                    beam_list_word.append(caption)
                    #baseline = [vocab.idx2word[output_baseline[batch_idx][w_idx].item()] for w_idx in
                       #    range(output_baseline.size(1))]
                beam_list=caption_refine(beam_list_word,NumBeams=args.NumBeams)
                #baseline=caption_refine(baseline)
                #beam_list.append([baseline])
                caption_list.append(beam_list)
                caption_list_word.append(beam_list_word)


            ############### 코딩을 해보자


            #

            # 핀포인트찍어서 디버깅
            img_path_list=[]
            for imgID in img_Ids:

                if(os.path.exists(coco_train_img_file_template.format(imgID))):
                    img_path_list.append(coco_train_img_file_template.format(imgID))
                elif(os.path.exists(coco_val_img_file_template.format(imgID))):
                    img_path_list.append(coco_val_img_file_template.format(imgID))
                elif(os.path.exists(coco_test_img_file_template.format(imgID))):
                    img_path_list.append(coco_test_img_file_template.format(imgID))                     
                else:
                    print('No image to read')
                    assert False

            Yolo_outs=Yolov3(img_path_list, addr_to_Yolov3 = address_to_Yolov3, )
            Yolo_words=[[obj[-1] for obj in sample] for sample in Yolo_outs]
            person_embed=decoder.embed(torch.cuda.LongTensor([vocab('person')]))
            man_embed=decoder.embed(torch.cuda.LongTensor([vocab('man')]))

            embed_list=[]
            for sample in Yolo_outs:
                sample_embed=[]
                for obj in sample:
                    tmp_embed=decoder.embed(torch.cuda.LongTensor([vocab(obj[-1])]))
                    sample_embed.append(tmp_embed) #0부터 (n-1)까지 각 sample마다의 object class를 embeding시킨 것.이 sample_embed에들어감. object 개수가 여러개면 여러개 들어감. 배치사이즈 :n 
                embed_list.append(sample_embed)#batch마다!!!!!!!!!


            #pdb.set_trace()        
            # object별로 embeding.         q
            batch_size=outputs.size(0)
            caption_list_dummy=copy.deepcopy(caption_list)
            Yolo_not_hit_table=torch.Tensor(outputs.size(0),args.NumBeams).fill_(0)
            for (sample, sample_in_wordlist, refined_sample_in_wordlist, Yolo_not_hit) in zip(Yolo_outs, caption_list_word, caption_list_dummy, Yolo_not_hit_table):
                for obj in sample :
                    g=obj[-1]

                    g_embed=decoder.embed(torch.cuda.LongTensor([vocab(g)]))
                    for beam_idx in range(args.NumBeams):
                        #if(not(g in sample_in_wordlist[beam_idx])):
                            #refined_sample_in_wordlist[beam_idx][0] += ' avengers'
                        Yolo_In_Sentence=False
                        for t_step in range(20):
                            t_step_word_embed=decoder.embed(torch.cuda.LongTensor([vocab(sample_in_wordlist[beam_idx][t_step])]))
                            if( 0.6 <= cosine_sim(g_embed, t_step_word_embed) ):
                                Yolo_In_Sentence=True
                        if(not(Yolo_In_Sentence)):    
                            #refined_sample_in_wordlist[beam_idx][0] += ' avengers'
                            Yolo_not_hit[beam_idx] += 1

            best_idx=[]
            for sample_idx in range(batch_size):
                minimum_hit=torch.min(Yolo_not_hit_table[sample_idx],0)[0]
                min_hit_idx=(minimum_hit==Yolo_not_hit_table[sample_idx])
                for beam_idx in range(args.NumBeams):
                    if(min_hit_idx[beam_idx]==1):
                        break
                best_idx.append(beam_idx)

            best_captions=[]
            for sample_idx in range(batch_size):
                best_captions.append(caption_list[sample_idx][best_idx[sample_idx]][0])

            #                for j in range(5-1) :
            #                    for k in range(20-1) :
            #                        if Yolo_outs == caption_list_word[i][j][k] :
            #                         
            #                            pass
            #                    else :
            #                        caption_list_word[i][j][k].append('avengers')


            #score_opt = score_beams[:,0] #0 대신 index. (0으로해놓으면) 첫번째애들만 데려옴. 0대신 뭐넣을건지 ~~~ YOLO 연동시키고 
            best_captions_list.extend(best_captions)
            img_id_list.extend(img_Ids)
            i_train += 1

    result_json=make_json(best_captions_list, img_id_list)

    if(args.split == 'val'):
        args.split += '2014'

    with open(args.result_json_path + '/captions_%s_%s_results.json' \
              % (args.split, args.method), 'w') as f:

        json.dump(result_json, f)


def make_json(captions, ImgIds):
    utils.assert_eq(len(captions), len(ImgIds))
    results = []
    for i in range(len(captions)):
        result = {}
        result['image_id'] = ImgIds[i]
        result['caption'] = captions[i]
        results.append(result)
    return results
           
        # Save the model checkpoints # 결과를 저장할 위치
       #  model_path = os.path.join(
        #    args.model_path, args.t_method, 'model{}_LR{}'.format(args.model_num, args.LRdim),
        #    'model-{}-CiderOpt-GradClip-{}.pth'.format(epoch + 1, args.grad_clip))
       # utils.save_model(model_path, encoder, decoder, epoch, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_BUTD_36/standard_vocab/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/mnt/server5_hard1/seungjun/XAI/BAN_XAI/data/vocab.pkl', help='path for vocabulary wrapper')
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
    parser.add_argument('--split', type=str, default='test2014', help='test2014 // val')
    parser.add_argument('--result_json_path', type=str, default='test_json/', help='saving path for captioning json')
    parser.add_argument('--method', type=str, default='gozii+version+0', help='applied method')
    args = parser.parse_args()
    print(args)
    main(args)