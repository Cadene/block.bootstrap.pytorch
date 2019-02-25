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

class VG(AbstractVQA):

    def __init__(self,
            dir_data='data/vg',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            dataset_img=None,
            nans=2000,
            minwcount=10,
            nlp='mcb',
            dir_rcnn='data/vqa/vgenome/extract_rcnn'):
        self.dir_data = dir_data
        self.dir_raw = os.path.join(self.dir_data, 'raw')
        dir_anno = os.path.join(self.dir_raw, 'annotations')
        if not os.path.isdir(dir_anno):
            self.make_annotations()
        super(VG, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split='train',
            samplingans=False,
            has_valset=False,
            has_testset=False,
            has_testset_anno=False,
            has_testdevset=False,
            has_answers_occurence=False,
            do_tokenize_answers=True)
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
        item['image_name'] = question['image_id']

        # Process Object, Attribut and Relational features
        item = self.add_rcnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            
            item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        return item

    def download(self):
        dir_raw = self.dir_raw
        dir_img = os.path.join(dir_raw, 'images')
        # TODO: question_answers.json must be in vgenome/raw/json
        os.system('wget http://visualgenome.org/static/data/dataset/image_data.json.zip -P '+dir_raw)
        os.system('wget http://visualgenome.org/static/data/dataset/question_answers.json.zip -P '+dir_raw)
        os.system('wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P '+dir_raw)
        os.system('wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P '+dir_raw)

        os.system('unzip '+os.path.join(dir_raw, 'image_data.json.zip')+' -d '+dir_raw)
        os.system('unzip '+os.path.join(dir_raw, 'question_answers.json.zip')+' -d '+dir_raw)
        os.system('unzip '+os.path.join(dir_raw, 'images.zip')+' -d '+dir_raw)
        os.system('unzip '+os.path.join(dir_raw, 'images2.zip')+' -d '+dir_raw)

        os.system('mv '+os.path.join(dir_raw, 'VG_100K')+' '+dir_img)

        #os.system('mv '+os.path.join(dir_raw, 'VG_100K_2', '*.jpg')+' '+dir_img)
        os.system('find '+os.path.join(dir_raw, 'VG_100K_2')+' -type f -name \'*\' -exec mv {} '+dir_img+' \\;')
        os.system('rm -rf '+os.path.join(dir_raw, 'VG_100K_2'))

        # remove images with 0 octet in a ugly but efficient way :')
        #print('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do rm '+dir_img+'/${f}; done;')
        os.system('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do echo '+dir_img+'/${f}; done;')
        os.system('for f in $(ls -lh '+dir_img+' | grep " 0 " | cut -s -f14 --delimiter=" "); do rm '+dir_img+'/${f}; done;')

        self.make_annotations()

    def make_annotations(self):
        # transform vgenome annotations into vqa2 format
        dir_json = os.path.join(self.dir_raw, 'json')
        path_qa = os.path.join(dir_json, 'question_answers.json')
        qa = json.load(open(path_qa))

        dir_anno = os.path.join(self.dir_raw, 'annotations')
        if not os.path.isdir(dir_anno):
            os.system('mkdir -p '+dir_anno)
        path_train_ann = osp.join(dir_anno, 'mscoco_train2014_annotations.json')
        path_train_ques = osp.join(dir_anno, 'OpenEnded_mscoco_train2014_questions.json')
        #path_val_ann = osp.join(dir_anno, 'mscoco_val2014_annotations.json')
        #path_val_ques = osp.join(dir_anno, 'OpenEnded_mscoco_val2014_questions.json')

        train_ques = {}
        train_ques['data_subtype'] = 'train2014'
        train_ques['task_type'] = 'Open-Ended'
        #train_ques['info'] = {}
        #train_ques['license'] = {}
        #train_ques['data_type'] = 'mscoco'
        train_ques['questions'] = []

        train_ann = {}
        train_ann['data_subtype'] = 'train2014'
        train_ann['task_type'] = 'Open-Ended'
        #train_ann['info'] = {}
        #train_ann['license'] = {}
        #train_ann['data_type'] = 'mscoco'
        train_ann['annotations'] = []

        for i in tqdm(range(len(qa))):
            for qas in qa[i]['qas']:
                ques = {}
                ques['question'] = qas['question']
                ques['question_id'] = qas['qa_id']
                ques['image_id'] = qas['image_id']
                train_ques['questions'].append(ques)
                ann = {}
                ann['answer_type'] = 'vgenome'
                ann['multiple_choice_answer'] = qas['answer']
                ann['image_id'] = qas['image_id']
                ann['question_id'] = qas['qa_id']
                ann['question_type'] = 'vgenome'
                ann['answers'] = [{
                    'answer_id': 1,
                    'answer': qas['answer'],
                    'answer_confidence': 'yes'
                }]
                train_ann['annotations'].append(ann)

        Logger()('Saving annotations...')

        with open(path_train_ques, 'w') as f:
            json.dump(train_ques, f)

        with open(path_train_ann, 'w') as f:
            json.dump(train_ann, f)

        Logger()('Done')
