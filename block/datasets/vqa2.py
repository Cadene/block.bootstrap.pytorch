import os
import csv
import copy
import json
import torch
import numpy as np
from os import path as osp
from bootstrap.lib.logger import Logger
from .vqa_utils import AbstractVQA

class VQA2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa2',
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
            dir_rcnn='data/coco/extract_rcnn'):
        super(VQA2, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False)
        self.dir_rcnn = dir_rcnn
        # to activate manually in visualization context (notebook)
        self.load_original_annotation = False

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
        if self.load_original_annotation:
            item['original_question'] = question

        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = question['image_name']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        else:
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False
        return item

    def download(self):
        dir_zip = osp.join(self.dir_raw, 'zip')
        os.system('mkdir -p '+dir_zip)
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_train2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_val2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))
