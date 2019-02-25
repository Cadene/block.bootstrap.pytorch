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
from .vqa_utils import AbstractVQA

class TDIUC(AbstractVQA):

    def __init__(self,
            dir_data='data/tdiuc',
            split='train',
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            dir_rcnn='data/tdiuc/extract_rcnn'):
        super(TDIUC, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split='trainval',
            samplingans=False,
            has_valset=False,
            has_testset=True,
            has_testset_anno=True,
            has_testdevset=False,
            has_answers_occurence=False,
            do_tokenize_answers=False)
        self.dir_rcnn = dir_rcnn

    def add_answer(self, annotations):
        for item in annotations:
            item['answer'] = item['answers'][0]['answer']
        return annotations

    def add_rcnn_to_item(self, item):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn['norm_rois']
        item['nb_regions'] = item['visual'].size(0)
        return item

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        #item['original_question'] = question
        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = self.get_image_name(question['image_id'])

        # TODO: UGLY TO FIX
        # proc_split=trainval
        # if split=train -> train2014
        # if split=val -> train2014
        # if split=test -> val2014
        item['image_name'] = item['image_name'].replace('val', 'train')
        item['image_name'] = item['image_name'].replace('test2015', 'val2014')

        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            #item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']

        return item

    def download(self):
        os.system('wget http://kushalkafle.com/data/TDIUC.zip -P '+self.dir_raw)
        os.system('unzip '+os.path.join(self.dir_raw, 'TDIUC.zip')+' -d '+self.dir_raw)
        dir_zip = os.path.join(self.dir_raw, 'TDIUC')
        dir_ann = os.path.join(self.dir_raw, 'annotations')
        os.system('mv '+os.path.join(self.dir_zip, 'Annotations')+'/* '+dir_ann)
        os.system('mv '+os.path.join(self.dir_zip, 'Questions')+'/* '+dir_ann)
