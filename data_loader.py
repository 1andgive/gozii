import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import torch.nn.functional as F
import pdb
import _pickle as cPickle
import warnings
import json
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
from nltk.tokenize import word_tokenize
import sys

from address_server_XAI import *

sys.path.append(addr_BAN)

from codes.dataset import VQAFeatureDataset
from torch.utils.data import ConcatDataset



class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


class BottomUp_CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    #def __init__(self, name, json_, vocab, dataroot=addr_dataroot, hdf5path=addr_hdf5path, adaptive_=True):
    def __init__(self, name, json_, vocab, dataroot=addr_pklfix_path, hdf5path=addr_hdf5fix_path, adaptive_=False):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            hdf5path: pre-trained bottom-up directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
        """

        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test2014', 'val2014', 'trainCider', 'valCider']
        self.adaptive = adaptive_
        self.name=name
        if(name=='val2014' or name == 'valCider'):
            name='val'
        elif(name == 'trainCider'):
            name='train'
        #image_id => hdf pre-trained value's index mapping
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        self.h5_path = os.path.join(hdf5path, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        if not os.path.isfile(self.h5_path):
            print('file %s not found' % self.h5_path)

        if not os.access(self.h5_path, os.R_OK):
            print('file %s not readable' % self.h5_path)

        print('loading features from h5 file : {}'.format(self.h5_path))
        hf = h5py.File(self.h5_path, 'r')
        self.hf = hf
        # self.features = np.array(hf.get('image_features')) # visual features, change this to vs_features=hf.get('image_features'), self.features_batch=np.array(vs_features[batch_idx])
        self.features_hf = hf.get('image_features')
        print('loading imagefeatures')
        # self.spatials = np.array(hf.get('spatial_features')) # bbox
        self.spatials_hf = hf.get('spatial_features')  # bbox
        print('loading spatial features')
        if self.adaptive:
            self.pos_boxes = np.array(hf.get('pos_boxes'))

        if self.adaptive:
            self.v_dim = self.features_hf.shape[1]
            self.s_dim = self.spatials_hf.shape[1]
        else:
            self.v_dim = self.features_hf.shape[2]
            self.s_dim = self.spatials_hf.shape[2]

        if (self.name in ['train', 'val']):
            self.coco = COCO(json_)
            self.ids = list(self.coco.anns.keys())
        else:
            self.coco=json.load(open(json_))
            self.ids=self.coco['images']
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        target=None

        if(self.name in ['train', 'val']):
            ann_id = self.ids[index]
            caption = coco.anns[ann_id]['caption']
            img_id = coco.anns[ann_id]['image_id']
        else:
            img_id = self.ids[index]['id']
        hdf_img_idx=self.img_id2idx[img_id]
        self.features_hf = self.hf.get('image_features')
        self.spatials_hf = self.hf.get('spatial_features')  # bbox
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features_hf[hdf_img_idx]))
            spatials = torch.from_numpy(np.array(self.spatials_hf[hdf_img_idx]))
        else:
            features = torch.from_numpy(
                np.array(self.features_hf[self.pos_boxes[hdf_img_idx][0]:self.pos_boxes[hdf_img_idx][1], :]))
            spatials = torch.from_numpy(
                np.array(self.spatials_hf[self.pos_boxes[hdf_img_idx][0]:self.pos_boxes[hdf_img_idx][1], :]))

        # Convert caption (string) to word ids.
        if (self.name in ['train', 'val']):
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            return features,spatials, target
        else:
            return features, spatials, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data, use_VQAE=False, use_VQAX=False, isTest=False, unknownToken=-1):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    if(isTest and not(use_VQAE)):
        features, spatials, img_ids = zip(*data)
        captions=[]
        idcies=range(len(img_ids))
    else:
        if (not(use_VQAE) and not(use_VQAX)):
            data.sort(key=lambda x: len(x[2]), reverse=True)
            features, spatials, captions = zip(*data)
        elif(use_VQAE):
            data.sort(key=lambda x: len(x[4]), reverse=True)
            if(isTest):
                features, spatials, questions, answers, captions, img_ids = zip(*data)
            else:
                features, spatials, questions, answers, captions = zip(*data)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]

        captions=list(captions)
        features=list(features)
        spatials=list(spatials)

        for i in range(len(lengths)-1, 0, -1):
            cap=captions[i]
            #length=lengths[i]
            if unknownToken in cap:
                del captions[i] # skip if the caption has <unk> token
                del lengths[i]
                del features[i]
                del spatials[i]

        captions = tuple(captions)
        features = tuple(features)
        spatials = tuple(spatials)

        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        # lengths for 'pack_padded_sequence' should be sorted!!
        idcies = np.argsort(lengths)[::-1]  # sorting in descending order
        new_lengths = [lengths[idx] for idx in idcies]

    # Merge images (from tuple of 3D tensor to 4D tensor).





    if features[0] is not None:
        new_features = [features[idx] for idx in idcies]
        new_features = trim_collate(new_features)
    else:
        new_features=features

    if spatials[0] is not None:
        new_spatials = [spatials[idx] for idx in idcies]
        new_spatials = trim_collate(new_spatials)
    else:
        new_spatials=spatials

    if (isTest and not(use_VQAE)):
        return new_features, new_spatials, img_ids
    else:

        new_targets=[targets[idx] for idx in idcies]
        new_targets=torch.stack(new_targets)

        #pdb.set_trace()
        if (not (use_VQAE) and not (use_VQAX)):
            return new_features, new_spatials, new_targets, new_lengths
        elif(use_VQAE):
            new_questions=[questions[idx] for idx in idcies]
            new_answers=[answers[idx] for idx in idcies]
            new_questions=torch.stack(new_questions)
            new_answers=torch.stack(new_answers)
            if(isTest):
                new_img_ids=[img_ids[idx] for idx in idcies]
                return new_features, new_spatials, new_questions, new_answers, new_img_ids, new_lengths
            else:
                return new_features, new_spatials, new_questions, new_answers, new_targets, new_lengths

def get_loader(root, json_, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json_,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def BottomUp_get_loader(name, json, vocab, transform, batch_size, shuffle, num_workers,adaptive=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    assert name in ['train', 'val', 'test-dev2015', 'test2015',  'train+val', 'test2014', 'val2014','trainCider', 'valCider', 'train+valCider']

    if name=='train+val':
        coco_train = BottomUp_CocoDataset(name='train',
                                          json_=json[0],
                                    vocab=vocab, adaptive_=adaptive)
        coco_val = BottomUp_CocoDataset(name='val',
                                        json_=json[1],
                                          vocab=vocab, adaptive_=adaptive)
        coco=ConcatDataset([coco_train,coco_val])
    elif name == 'train+valCider':
        coco_train = BottomUp_CocoDataset(name='trainCider',
                                          json_=json[0],
                                          vocab=vocab, adaptive_=adaptive)
        coco_val = BottomUp_CocoDataset(name='valCider',
                                        json_=json[1],
                                        vocab=vocab, adaptive_=adaptive)
        coco = ConcatDataset([coco_train, coco_val])
    else:
        coco = BottomUp_CocoDataset(name=name,
                           json_=json,
                           vocab=vocab, adaptive_=adaptive)



    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    if (name in ['train', 'val', 'train+val']):
        data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=lambda b: collate_fn(b, unknownToken=vocab('<unk>')))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn= lambda b: collate_fn(b, isTest=True, unknownToken=vocab('<unk>')))


    return data_loader

def trim_collate(batch):
    max_num_boxes = max([x.size(0) for x in batch])
    numel = len(batch) * max_num_boxes * batch[0].size(-1)
    for i in range(len(batch)):
        try:
            storage = batch[i].storage()._new_shared(numel)
            out = batch[i].new(storage)
        except:
            pass
        else:
            return torch.stack([F.pad(x, (0, 0, 0, max_num_boxes - x.size(0))).data for x in batch], 0, out=out)




class VQA_E_loader(data.Dataset):
    """VQA_E Dataset (VQA_E) compatible with torch.utils.data.DataLoader."""

    def __init__(self, VQAE_train, VQAE_val, vqaDict, captionVcoab, captionMaxSeqLength=40,
                 dataroot='codes\\tools\\data'):
        self.VQA_E = VQAE_train + VQAE_val
        self.vqaDict = vqaDict
        self.capVoc = captionVcoab
        self.captionMaxSeqLength = captionMaxSeqLength
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

    def __getitem__(self, index):
        answer = self.VQA_E[index]['multiple_choice_answer']
        if answer in self.ans2label.keys():
            answer = self.ans2label[answer]
        else:
            answer = 545  # self.ans2label['unknown'] ==> 545
        question = self.VQA_E[index]['question'][:-1].lower()
        Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
        Wq_list = word_tokenize(question)
        if len(Wq_list) <= 14:
            for idx in range(len(Wq_list)):
                if Wq_list[idx] in self.vqaDict.word2idx.keys():
                    Wq_[0, idx] = self.vqaDict.word2idx[Wq_list[idx]]
                else:
                    Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
                    answer = 545  # self.ans2label['unknown'] ==> 545
        else:
            Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
            answer = 545  # self.ans2label['unknown'] ==> 545

        explanation = self.VQA_E[index]['explanation'][0]
        Wc_list = word_tokenize(explanation.lower())
        Wc_ = torch.cuda.LongTensor(1, self.captionMaxSeqLength).fill_(2)  # <'end'>

        Wc_[0, 0] = 1  # <'start'>
        if len(Wc_list) <= self.captionMaxSeqLength - 1:
            for idx in range(1, len(Wc_list)):
                if Wc_list[idx] in self.capVoc.word2idx.keys():
                    Wc_[0, idx] = self.capVoc.word2idx[Wc_list[idx]]
                else:
                    Wc_ = torch.cuda.LongTensor(1, self.captionMaxSeqLength).fill_(0)  # <'pad'>
                    answer = 545  # self.ans2label['unknown'] ==> 545
        else:
            Wc_ = torch.cuda.LongTensor(1, self.captionMaxSeqLength).fill_(0)  # <'pad'>
            answer = 545  # self.ans2label['unknown'] ==> 545

        return Wq_, Wc_, answer  # Wq_ <= question in MSCOCO-VQA index // Wc_ <== caption in MSCOCO_Caption index // answer <== MSCOCO-VQA answer label

    def __len__(self):
        return len(self.VQA_E)


class VQA_E_finetuning_Dataset(VQAFeatureDataset):
    """VQA_E Dataset (VQA_E) compatible with torch.utils.data.DataLoader."""

    def __init__(self,name,vqaDict, captionVcoab, dataroot=os.path.join('codes', 'tools', 'data'), hdf5path=addr_hdf5path,
                 adaptive=True, vqa_E_train_path=addr_vqae_train_path,
                 vqa_E_val_path=addr_vqae_val_path):
        assert name in ['train', 'val', 'eval']
        self.name = name
        if(name=='eval'):
            name = 'val'
        super(VQA_E_finetuning_Dataset,self).__init__(name,vqaDict,dataroot,hdf5path,adaptive)
        if (name == 'train'):
            self.VQA_E=json.load(open(vqa_E_train_path))
        elif(name == 'val'):
            self.VQA_E=json.load(open(vqa_E_val_path))

        self.vqaDict=vqaDict
        self.capVoc=captionVcoab



    def __getitem__(self, index):
        answer = self.VQA_E[index]['multiple_choice_answer']

        img_id = self.VQA_E[index]['img_id']
        if answer in self.ans2label.keys():
            answer = self.ans2label[answer]
        else:
            answer = 545  # self.ans2label['unknown'] ==> 545
        question=self.VQA_E[index]['question'][:-1].lower()
        Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
        Wq_list = word_tokenize(question)
        if len(Wq_list) <= 14:
            for idx in range(len(Wq_list)):
                if Wq_list[idx] in self.vqaDict.word2idx.keys():
                    Wq_[0, idx] = self.vqaDict.word2idx[Wq_list[idx]]
                else:
                    Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
                    answer = 545  # self.ans2label['unknown'] ==> 545
        else:
            Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
            answer = 545  # self.ans2label['unknown'] ==> 545


        explanation=self.VQA_E[index]['explanation'][0]
        tokens=word_tokenize(explanation.lower())
        Wc_list=[]
        Wc_list.append(self.capVoc('<start>')) #<'start'>
        Wc_list.extend([self.capVoc(token) for token in tokens])
        Wc_list.append(self.capVoc('<end>'))
        Wc_=torch.Tensor(Wc_list)


        img_id=self.VQA_E[index]['img_id']
        if(self.name == 'eval'):
            ann_id = self.VQA_E[index]['ann_id']
        hdf_img_idx=self.img_id2idx[img_id]
        self.features_hf=self.hf.get('image_features')
        self.spatials_hf=self.hf.get('spatial_features')

        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features_hf[hdf_img_idx]))
            spatials = torch.from_numpy(np.array(self.spatials_hf[hdf_img_idx]))
        else:
            features=torch.from_numpy(
                np.array(self.features_hf[self.pos_boxes[hdf_img_idx][0]:self.pos_boxes[hdf_img_idx][1],:]))
            spatials=torch.from_numpy(
                np.array(self.spatials_hf[self.pos_boxes[hdf_img_idx][0]:self.pos_boxes[hdf_img_idx][1],:]))

        target = torch.zeros(self.num_ans_candidates)
        if answer is not None:
            target.scatter_(0, torch.LongTensor([answer]), 1)
        if (self.name =='eval'):
            return features, spatials, Wq_, target.cuda(), Wc_, ann_id  # Wq_ <= question in MSCOCO-VQA index // Wc_ <== caption in MSCOCO_Caption index // answer <== MSCOCO-VQA answer label
        else:
            return features, spatials, Wq_, target.cuda(), Wc_  # Wq_ <= question in MSCOCO-VQA index // Wc_ <== caption in MSCOCO_Caption index // answer <== MSCOCO-VQA answer label

    def __len__(self):
        return len(self.VQA_E)


def VQAE_FineTunning_loader(name, dictionary_vqa, vocab_VQAE, batch_size, shuffle, num_workers):
    assert name in ['train', 'val', 'train+val', 'eval']

    if (name == 'train'):
        vqaE_dset = VQA_E_finetuning_Dataset('train', dictionary_vqa, vocab_VQAE)
    elif (name == 'val'):
        vqaE_dset = VQA_E_finetuning_Dataset('val', dictionary_vqa, vocab_VQAE)
    elif (name == 'eval'):
        vqaE_dset = VQA_E_finetuning_Dataset('eval', dictionary_vqa, vocab_VQAE)
    elif (name == 'train+val'):
        vqaE_dset = ConcatDataset([VQA_E_finetuning_Dataset('train', dictionary_vqa, vocab_VQAE), VQA_E_finetuning_Dataset('val', dictionary_vqa, vocab_VQAE)])

    if(name == 'eval'):
        data_loader = torch.utils.data.DataLoader(dataset=vqaE_dset, batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=lambda b: collate_fn(b, use_VQAE=True, use_VQAX=False, isTest=True))
    else:
        data_loader=torch.utils.data.DataLoader(dataset=vqaE_dset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, use_VQAE=True, use_VQAX=False))


    return data_loader


class vqaE_CapEnc_Dataset(data.Dataset):
    """VQA_E Dataset (VQA_E) compatible with torch.utils.data.DataLoader."""

    def __init__(self,name,vqaDict, captionVcoab, dataroot='codes\\tools\\data', hdf5path='D:\\Data_Share\\Datas\\VQA_COCO\\BottomUpPreTrain\\hdf5',
                 adaptive=True, vqa_E_train_path='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json',
                 vqa_E_val_path='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json'):
        assert name in ['train', 'val', 'train+val']
        super(vqaE_CapEnc_Dataset,self).__init__()
        if (name == 'train'):
            self.VQA_E=json.load(open(vqa_E_train_path))
        elif(name == 'val'):
            self.VQA_E=json.load(open(vqa_E_val_path))

        self.vqaDict=vqaDict
        self.capVoc=captionVcoab
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

    def __getitem__(self, index):
        answer = self.VQA_E[index]['multiple_choice_answer']
        if answer in self.ans2label.keys():
            answer = self.ans2label[answer]
        else:
            answer = 545  # self.ans2label['unknown'] ==> 545
        question = self.VQA_E[index]['question'][:-1].lower()
        Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
        Wq_list = word_tokenize(question)
        if len(Wq_list) <= 14:
            for idx in range(len(Wq_list)):
                if Wq_list[idx] in self.vqaDict.word2idx.keys():
                    Wq_[0, idx] = self.vqaDict.word2idx[Wq_list[idx]]
                else:
                    Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
                    answer = 545  # self.ans2label['unknown'] ==> 545
        else:
            Wq_ = torch.cuda.LongTensor(1, 14).fill_(19901)
            answer = 545  # self.ans2label['unknown'] ==> 545

        explanation = self.VQA_E[index]['explanation'][0]
        tokens = word_tokenize(explanation.lower())
        Wc_list = []
        Wc_list.append(self.capVoc('<start>'))  # <'start'>
        Wc_list.extend([self.capVoc(token) for token in tokens])
        Wc_list.append(self.capVoc('<end>'))
        Wc_ = torch.Tensor(Wc_list)

        target = torch.zeros(self.num_ans_candidates)
        if answer is not None:
            target.scatter_(0, torch.LongTensor([answer]), 1)

        return None, None, Wq_, target, Wc_  # Wq_ <= question in MSCOCO-VQA index // Wc_ <== caption in MSCOCO_Caption index // answer <== MSCOCO-VQA answer label

    def __len__(self):
        return len(self.VQA_E)


def vqaE_CapEnc_Loader(name, dictionary_vqa, vocab_VQAE, batch_size, shuffle, num_workers):
    assert name in ['train', 'val', 'train+val']

    if (name == 'train'):
        vqaE_dset = vqaE_CapEnc_Dataset('train', dictionary_vqa, vocab_VQAE)
    elif (name == 'val'):
        vqaE_dset = vqaE_CapEnc_Dataset('val', dictionary_vqa, vocab_VQAE)
    elif (name == 'train+val'):
        vqaE_dset = ConcatDataset([vqaE_CapEnc_Dataset('train', dictionary_vqa, vocab_VQAE), vqaE_CapEnc_Dataset('val', dictionary_vqa, vocab_VQAE)])

    data_loader=torch.utils.data.DataLoader(dataset=vqaE_dset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, use_VQAE=True, use_VQAX=False))
    return data_loader