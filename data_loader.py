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
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py



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

    def __init__(self, name, json, vocab, dataroot='codes\\tools\\data', hdf5path='D:\\Data_Share\\Datas\\VQA_COCO\\BottomUpPreTrain\\hdf5'):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            hdf5path: pre-trained bottom-up directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
        """
        assert name in ['train', 'val', 'test-dev2015', 'test2015']
        self.adaptive = True

        #image_id => hdf pre-trained value's index mapping
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        self.h5_path = os.path.join(hdf5path, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))



        print('loading features from h5 file')
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


        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
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
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return features,spatials, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
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
    data.sort(key=lambda x: len(x[1]), reverse=True)

    features, spatials, captions = zip(*data)



    # Merge images (from tuple of 3D tensor to 4D tensor).




    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # lengths for 'pack_padded_sequence' should be sorted!!
    idcies=np.argsort(lengths)[::-1] # sorting in descending order
    new_lengths=[lengths[idx] for idx in idcies]
    new_features = [features[idx] for idx in idcies]
    new_spatials = [spatials[idx] for idx in idcies]
    new_targets=[targets[idx] for idx in idcies]

    new_features = trim_collate(new_features)
    new_spatials = trim_collate(new_spatials)
    new_targets=torch.stack(new_targets)

    #pdb.set_trace()
    return new_features, new_spatials, new_targets, new_lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
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


def BottomUp_get_loader(name, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    assert name in ['train', 'val', 'test-dev2015', 'test2015']

    coco = BottomUp_CocoDataset(name=name,
                       json=json,
                       vocab=vocab)

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

def trim_collate(batch):
    max_num_boxes = max([x.size(0) for x in batch])
    numel = len(batch) * max_num_boxes * batch[0].size(-1)
    storage = batch[0].storage()._new_shared(numel)
    out = batch[0].new(storage)
    return torch.stack([F.pad(x, (0, 0, 0, max_num_boxes - x.size(0))).data for x in batch], 0, out=out)
