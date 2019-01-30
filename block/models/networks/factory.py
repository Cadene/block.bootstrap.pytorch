import sys
import copy
import torch
import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from .vqa_net import VQANet
from .vrd_net import VRDNet

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']

    if opt['name'] == 'vqa_net':
        net = VQANet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            attention=opt['attention'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    elif opt['name'] == 'vrd_net':
        net = VRDNet(opt)

    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net
