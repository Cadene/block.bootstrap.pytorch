import os
import os.path as osp
import sys
import csv
import base64
import json
import numpy as np
import torch
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from .vqa2 import VQA2
from .vg import VG

class VQA2VG(Dataset):

    def __init__(self,
            dir_data='data/vqa2',
            dir_data_vg='data/vg',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn',
            dir_rcnn_vg='data/vg/extract_rcnn'):
        super(VQA2VG, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        self.vqa2 = VQA2(
            dir_data=dir_data,
            split=split,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            dir_rcnn=dir_rcnn)
        self.vg = VG(
            dir_data=dir_data_vg,
            split='train',
            nans=10000,
            minwcount=0,
            nlp=nlp,
            dir_rcnn=dir_rcnn_vg)
        self.collate_fn = self.vqa2.collate_fn

    def __getattr__(self, key):
        try:
            return super(VQA2VG, self).__getattr__(key)
        except AttributeError:
            return self.vqa2.__getattribute__(key)

    def __getitem__(self, index):
        if index < len(self.vqa2):
            item = self.vqa2[index]
        else:
            item = self.vg[index-len(self.vqa2)]
        return item

    def __len__(self):
        return len(self.vqa2) + len(self.vg)
