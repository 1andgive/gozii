
import os
import sys


import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

sys.path.append('D:\\VQA\\BAN')

from codes.dataset import Dictionary, VQAFeatureDataset
import codes.base_model as base_model
import codes.utils as utils

import pdb


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]

def get_image(q_id, Dict_qid2vid, data_root='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\test2015'): #q_id => single scalar
    img_id=Dict_qid2vid[q_id]
    img_name='COCO_test2015_%012d.jpg' % img_id
    img_path=os.path.join(data_root,img_name)
    img=plt.imread(img_path)

    return img


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def get_logits(model, dataloader,Dict_qid2vid):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N).start()
    for v, b, q, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        q = q.type(torch.LongTensor)
        with torch.no_grad():
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            logits, att = model(v, b, q, None)
            pred[idx:idx + batch_size, :].copy_(logits.data)
            qIds[idx:idx + batch_size].copy_(i)
            idx += batch_size
            img_list=[]
            question_list=[]
            answer_list=[]
            for idx in range(len(i)):
                img_list.append(get_image(i[idx].item(),Dict_qid2vid))
                question_list.append(get_question(q.data[idx],dataloader))
                answer_list.append(get_answer(logits.data[idx], dataloader))
        print(question_list[0])
        print(answer_list[0])
        showAttention(question_list[0],img_list[0],answer_list[0],att[0,:,:,:],b[0,:,:4])
        #pdb.set_trace()
    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

#######################################################################################

def showAttention(input_question, image, output_answer, attentions,bbox):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(19,25))
    #pdb.set_trace()
    attentions=attentions.cpu()



    att1 = attentions[0, :, :]
    att2 = attentions[1, :, :]
    #pdb.set_trace()
    att1=torch.t(att1)
    att2=torch.t(att2)

    #pdb.set_trace()

    att1_sum=torch.sum(att1,0)
    att2_sum = torch.sum(att2, 0)

    _,idx_att1=torch.sort(att1_sum,descending=True)
    _, idx_att2 = torch.sort(att2_sum, descending=True)

    idx_att1=idx_att1[:3]
    idx_att2 = idx_att2[:3]

    bbox_att1=bbox[idx_att1,:]
    bbox_att2 = bbox[idx_att2, :]

    im1=fig.add_subplot(221)
    im1.imshow(image)
    im_height=image.shape[0]
    im_width = image.shape[1]
    color_zero=[0,0,0]
    for i in range(3):
        pt=bbox_att1[i,:]
        pt[0]=pt[0]*im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color=color_zero.copy()
        color[i]=1
        im1.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        im1.text(pt[0], pt[1], str(idx_att1[i].item()+1), bbox={'facecolor':color, 'alpha':0.5})


    ax = fig.add_subplot(222)

    cax = ax.matshow(att1.numpy(), cmap='bone')
    fig.colorbar(cax)

    lis = range(attentions.size(1))
    x_axis = ["{:02d}".format(x) for x in lis]

    # Set up axes
    ax.set_xticklabels(x_axis, rotation=90)
    ax.set_yticklabels([''] + input_question.split(' ') +
                       ['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    im2 = fig.add_subplot(223)
    im2.imshow(image)

    for i in range(3):
        pt=bbox_att2[i,:]
        pt[0]=pt[0]*im_width
        pt[1] = pt[1] * im_height
        pt[2] = pt[2] * im_width
        pt[3] = pt[3] * im_height
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color=color_zero.copy()
        color[i]=1
        im2.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        im2.text(pt[0], pt[1], str(idx_att2[i].item()+1), bbox={'facecolor':color, 'alpha':0.5})

    ax2 = fig.add_subplot(224)

    cax2 = ax2.matshow(att2.numpy(), cmap='bone')
    fig.colorbar(cax2)

    # Set up axes
    ax2.set_xticklabels(x_axis, rotation=90)
    second_yaxis= ["f{:02d}".format(y) for y in range(len(input_question))]

    ax2.set_yticklabels(second_yaxis)

    # Show label at every tick
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

#######################################################################################`

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    name = 'test2015'

    question_path = os.path.join(
        'codes\\tools\\data', 'v2_OpenEnded_mscoco_%s_questions.json' % \
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    Dict_qid2vid = {}
    Dict_qid2ques={}
    for i in range(len(questions)):
        #pdb.set_trace()
        Dict_qid2vid[questions[i]['question_id']]=questions[i]['image_id']
    dictionary = Dictionary.load_from_file('codes/tools/data/dictionary.pkl')

    # Visualization through eval_dataset

    eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)


    def process(args, model, eval_loader,Dict_qid2vid):
        model_path = args.input + '/model%s.pth' % \
                     ('' if 0 > args.epoch else '_epoch%d' % args.epoch)

        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)

        logits, qIds = get_logits(model, eval_loader,Dict_qid2vid)
        results = make_json(logits, qIds, eval_loader)
        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)

        if args.logits:
            utils.create_dir('logits/' + model_label)
            torch.save(logits, 'logits/' + model_label + '/logits%d.pth' % args.index)

        utils.create_dir(args.output)
        if 0 <= args.epoch:
            model_label += '_epoch%d' % args.epoch

        with open(args.output + '/%s_%s.json' \
                  % (args.split, model_label), 'w') as f:
            # pdb.set_trace()
            json.dump(results, f)


    process(args, model, eval_loader,Dict_qid2vid)
