import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import pdb
import json


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_VQAE_vocab(vqa_E, threshold):
    """Build a simple vocabulary wrapper."""

    counter = Counter()
    for i, id in enumerate(vqa_E):
        caption = str(id['explanation'][0])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(vqa_E)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    #pdb.set_trace()
    return vocab

def main(args):
    VQA_E_train=json.load(open(args.train_path))
    VQA_E_val = json.load(open(args.val_path))
    VQA_E=VQA_E_train+VQA_E_val
    vocab = build_VQAE_vocab(VQA_E, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total VQA_E vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json',
                        help='path for train annotation file')
    parser.add_argument('--val_path', type=str,
                        default='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json',
                        help='path for validation annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/VQA_E_vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)