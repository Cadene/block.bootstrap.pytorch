import os
from os import path as osp
import json
from tqdm import tqdm
import cv2
import torch
import numpy as np
import itertools
from torch.utils.data.sampler import WeightedRandomSampler
from bootstrap.lib.logger import Logger
from bootstrap.datasets import transforms
from bootstrap.datasets.dataset import Dataset

class VRD(Dataset):
    """Documentation for VRD

    """
    def __init__(self,
            dir_data,
            split,
            neg_ratio=0.,
            batch_size=100,
            nb_threads=0,
            seed=1234,
            shuffle=True,
            pin_memory=True,
            mode='predicate'):
        super(VRD, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle)
        assert(split in ['train', 'val', 'test', 'trainval'])
        self.neg_ratio = neg_ratio
        self.seed = seed
        assert(mode in ['predicate', 'rel_phrase'])
        self.mode = mode
        self.dir_raw_json = osp.join(self.dir_data, 'annotations','raw')
        self.dir_images = osp.join(self.dir_data, 'images')
        self.dir_processed = osp.join(self.dir_data, 'annotations','processed')

        if not osp.exists(self.dir_raw_json):
            self.download_json()
        if not osp.exists(self.dir_images):
            self.download_images()

        self.vocabs = self.load_vocabs()
        if not osp.exists(self.dir_processed):
            self.process_json()

        self.json = self.load_json()
        self.ids = sorted(list(self.json.keys()))
        self.ids = self.remove_no_bboxes_images()

        if self.mode == 'predicate':
            if self.split in ['train', 'val']:
                self.make_train_val_split()
            if self.split in ['train', 'val', 'trainval']:
                self.dir_features = osp.join(self.dir_data, 'features', 'gt_boxes', 'train')
            else:
                self.dir_features = osp.join(self.dir_data, 'features', 'gt_boxes', 'test')

        elif self.mode == 'rel_phrase':
            assert(self.split == 'test')
            self.dir_features = osp.join(self.dir_data, 'features', 'pred_boxes', 'test')
            path_jraw = osp.join(self.dir_raw_json, 'annotations_test.json')
            with open(path_jraw, 'r') as f:
                self.json_raw = json.load(f)

        if not osp.exists(self.dir_features):
            self.download_features()

        if self.split in ['train', 'trainval']:
            self.shuffle = False
            self.sampler = WeightedRandomSampler(
                weights=[1]*len(self),
                num_samples=len(self),
                replacement=True)
        else:
            self.sampler = None

        self.collate_fn = transforms.Compose([
            transforms.ListDictsToDictLists(),
            transforms.CatTensors()
        ])

    def load_json(self):
        if self.split in ['train', 'val', 'trainval']:
            json_path = osp.join(self.dir_processed, 'train.json')
        else:
            json_path = osp.join(self.dir_processed, 'test.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        #data = {k.replace('.png','.jpg'):v for k,v in data.items()}
        return data

    def load_vocabs(self):
        rel_vocab_path = osp.join(self.dir_raw_json, 'predicates.json')
        obj_vocab_path = osp.join(self.dir_raw_json, 'objects.json')
        vocabs = {
            'relationships': self.extract_vocab(rel_vocab_path,bg=True),
            'objects': self.extract_vocab(obj_vocab_path,bg=True)
        }
        return vocabs

    def extract_vocab(self, path, bg=True):
        if bg:
            vocab = {
                'nametoi': {'__background__':0},
                'itoname': {0:'__background__'}
            }
        else:
            vocab = {
                'nametoi': {},
                'itoname': {}
            }

        with open(path, 'r') as f:
            data = json.load(f)

        for w in data:
            i = len(vocab['nametoi'])
            vocab['nametoi'][w] = i
            vocab['itoname'][i] = w
        return vocab

    def process_json(self):
        os.system('mkdir -p '+self.dir_processed)
        for split in ['train', 'test']:
            processed_split = self.process_split(split)
            path_processed = osp.join(self.dir_processed, f'{split}.json')
            with open(path_processed, 'w') as f:
                f.write(json.dumps(processed_split))

    def process_split(self, split):
        _json = {}
        annot_path = osp.join(self.dir_raw_json, 'annotations_%s.json' % split)
        raw_annot = json.load(open(annot_path))
        Logger()('Start processing %s split' % split)
        for image_id in tqdm(raw_annot):
            path_img = osp.join(self.dir_images, split, image_id)
            im = cv2.imread(path_img)
            height, width, _ = im.shape
            image_info = {
                'image_id':image_id,
                'height': height,
                'width': width
            }
            rels = raw_annot[image_id]
            # all the objects in the image
            raw_objects = [r[k]['bbox'] + [r[k]['category']] \
                       for r in rels for k in 'subject object'.split()]
            s_objects = [str(o) for o in raw_objects] # To make it hashable
            sobj_to_oid = dict()
            for sobj in s_objects:
                if sobj not in sobj_to_oid:
                    sobj_to_oid[sobj] = str(len(sobj_to_oid))+'_'+image_id

            parsed_objs = set()
            objects = []
            for obj in raw_objects:
                oid = sobj_to_oid[str(obj)]
                if oid not in parsed_objs:
                    parsed_objs.add(oid)
                    objects.append({
                        'name':self.vocabs['objects']['itoname'][1+obj[-1]],
                        'object_id':oid,
                        'x':obj[2],
                        'y':obj[0],
                        'w':obj[3]-obj[2],
                        'h':obj[1]-obj[0]
                    })

            relationships = []
            for rel in rels:
                sobj = str(rel['object']['bbox'] + [rel['object']['category']])
                ssubj = str(rel['subject']['bbox'] + [rel['subject']['category']])
                oid = sobj_to_oid[sobj]
                sid = sobj_to_oid[ssubj]
                relationships.append({
                    'subject_id':sid,
                    'object_id':oid,
                    'predicate':self.vocabs['relationships']['itoname'][rel['predicate']+1]
                })

            _json[image_id] = {
                'image_info':image_info,
                'objects':objects,
                'relationships':relationships
            }
        return _json

    def remove_no_bboxes_images(self):
        img_ids_2rm = {}
        for k, v in self.json.items():
            if len(v['objects']) == 0:
                img_id = v['image_info']['image_id']
                img_ids_2rm[img_id] = True
        new_ids = [idx for idx in self.ids if idx not in img_ids_2rm.keys()]
        return new_ids

    def download_json(self):
        dir_json = self.dir_raw_json
        os.system('mkdir -p '+dir_json)
        os.system('wget http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip -P '+dir_json)
        os.system('unzip '+osp.join(dir_json,'json_dataset.zip')+' -d '+dir_json)

    def download_features(self):
        os.system(f'wget http://data.lip6.fr/cadene/vrd/features.tar.gz -P {self.dir_data}')
        path_tar = osp.join(self.data, 'features.tar.gz')
        os.system(f'tar -xzvf {path_tar} -C {self.data}')

    def download_images(self):
        os.system('mkdir -p '+self.dir_images)
        os.system('wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip -P '+self.dir_images)
        os.system('unzip '+osp.join(self.dir_images, 'sg_dataset.zip')+' -d '+self.dir_images)
        os.system('mv '+osp.join(self.dir_images,'sg_dataset','sg_train_images')+' '+osp.join(self.dir_images,'train'))
        os.system('mv '+osp.join(self.dir_images,'sg_dataset','sg_test_images')+' '+osp.join(self.dir_images,'test'))
        os.system('rm -r ' +osp.join(self.dir_images,'sg_dataset'))

    def make_train_val_split(self, split_ratio=0.95):
        assert(self.split in ['train','val'])
        rnd = np.random.RandomState(self.seed)
        indices = rnd.choice(len(self),
                             size=int(len(self)*split_ratio),
                             replace=False)
        if self.split == 'val':
            indices = np.array(list(set(np.arange(len(self))) - set(indices)))
        self.ids = [self.ids[i] for i in indices]

    def __getitem__(self, index):
        if self.mode == 'predicate':
            item = self.getitem_predicate(index)
        elif self.mode == 'rel_phrase':
            item = self.getitem_rel_phrase(index)
        else:
            raise ValueError(self.mode)
        return item

    def getitem_rel_phrase(self, index):
        image_id = self.ids[index]
        path_feats = osp.join(self.dir_features, image_id.replace('.jpg','.pth').replace('.png','.pth'))
        imfeats = torch.load(path_feats)
        n_boxes = len(imfeats['rois'])
        if n_boxes == 0:
            item = {
                'idx': index,
                'image_id': image_id,
                'subject_cls_id': torch.ones(1).long(),
                'object_cls_id': torch.ones(1).long(),
                'subject_boxes': torch.ones(1,4),
                'object_boxes': torch.ones(1,4),
                'subject_features': torch.ones(1,2048),
                'object_features': torch.ones(1,2048),
                'n_boxes': n_boxes,
                'rois': torch.ones(1,4),
                'rois_nonorm': torch.ones(1,4),
                'cls': torch.ones(1),
                'cls_scores': torch.ones(1),
                'n_pairs': 1
            }
        else:
            width = self.json[image_id]['image_info']['width']
            height = self.json[image_id]['image_info']['height']

            pairs_id = np.array([(i,j) for i,j in itertools.product(range(n_boxes),repeat=2)])
            pooled_feat = torch.Tensor(imfeats['pooled_feat'].numpy()[pairs_id])
            imfeats['rois_nonorm'] = imfeats['rois'].clone()
            imfeats['rois'][:,[0,2]] /= width
            imfeats['rois'][:,[1,3]] /= height
            boxes = torch.Tensor(imfeats['rois'].numpy()[pairs_id])
            # Find a way to get 2501 automaticaly, or store it in the .pth file
            cls_ = torch.Tensor(imfeats['cls'].numpy()[pairs_id])
            item = {
                'idx': index,
                'image_id': image_id,
                'subject_cls_id': cls_[:,0].long(),
                'object_cls_id': cls_[:,1].long(),
                'subject_boxes': boxes[:,0,:],
                'object_boxes': boxes[:,1,:],
                'subject_features': pooled_feat[:,0,:],
                'object_features': pooled_feat[:,1,:],
                'n_boxes': n_boxes,
                'rois': imfeats['rois'],
                'rois_nonorm': imfeats['rois_nonorm'],
                'cls': imfeats['cls'],
                'cls_scores': imfeats['cls_scores'],
                'n_pairs': pairs_id.shape[0]
            }
        return item

    def getitem_predicate(self, index):
        image_id = self.ids[index]

        key = 'relationships'
        objects = self.json[image_id]['objects']
        rels = self.json[image_id]['relationships']
        oid_to_index = {o['object_id']:i for i,o in enumerate(objects)}
        # sanity check for nans
        path_feats = osp.join(self.dir_features, image_id.replace('.jpg','.pth').replace('.png','.pth'))
        oid_to_features = torch.load(path_feats)
        oid_to_features = {k+'.jpg':v for k,v in oid_to_features.items()}

        #oid_to_features = {k: v for k,v in torch.load(osp.join(self.features_dir, image_id+'.pth')).items() if max(v!=v) == 0}
        #rels = [r for r in rels if ((r['subject_id'] in oid_to_features) and (r['object_id'] in oid_to_features))]

        width = self.json[image_id]['image_info']['width']
        height = self.json[image_id]['image_info']['height']

        item = {}
        item['subject_boxes'] = []
        item['object_boxes'] = []
        item['subject_boxes_raw'] = []
        item['object_boxes_raw'] = []
        item['subject_cls_id'] = []
        item['object_cls_id'] = []
        item['target_cls_id'] = []
        item['subject_cls'] = []
        item['object_cls'] = []
        item['target_cls'] = []
        item['rels'] = rels
        item['objects'] = objects
        item['image_id'] = image_id
        item['idx'] = index
        item['target_oh'] = torch.zeros(len(rels),
                                        len(self.vocabs['relationships']['itoname']))
        item['subject_cls_oh'] = torch.zeros(len(rels),
                                             len(self.vocabs['objects']['itoname']))
        item['object_cls_oh'] = torch.zeros(len(rels),
                                            len(self.vocabs['objects']['itoname']))
        item['subject_features'] = torch.zeros(len(rels), 2048)
        item['object_features'] = torch.zeros(len(rels), 2048)
        positive_pairs = set()
        for i, rel in enumerate(rels):
            for btype in ['subject', 'object']:
                box = objects[oid_to_index[rel[btype+'_id']]]
                coords = [box['x'], box['y'], box['w'], box['h']]
                coords[0] = coords[0] / width
                coords[1] = coords[1] / height
                coords[2] = coords[0] + coords[2]/width
                coords[3] = coords[1] + coords[3]/height
                cls_ = box['name']
                cls_id = self.vocabs['objects']['nametoi'][box['name']]
                features = oid_to_features[box['object_id'].replace('.png','.jpg')]
                item[btype+'_boxes'].append(coords)
                item[btype+'_boxes_raw'].append([box['x'], box['y'], box['w'], box['h']])
                item[btype+'_cls'].append(cls_)
                item[btype+'_cls_id'].append(cls_id)
                item[btype+'_features'][i,:] = torch.Tensor(features)
                item[btype+'_cls_oh'][i,cls_id] = 1
            positive_pairs.add((oid_to_index[rel['subject_id']],
                                oid_to_index[rel['object_id']]))
            item['target_cls'].append(rel['predicate'])
            item['target_cls_id'].append(self.vocabs['relationships']['nametoi'][rel['predicate']])
            item['target_oh'][i,item['target_cls_id'][-1]] = 1

        if self.split in ['train','trainval'] and self.neg_ratio>0.:
            possible_pairs = set(itertools.product(oid_to_index.values(),repeat=2))
            candidate_pairs = list(possible_pairs - positive_pairs)
            np.random.shuffle(candidate_pairs)
            Nnegmax = int(self.neg_ratio*len(candidate_pairs))
            negative_pairs = candidate_pairs[:Nnegmax]
            item['subject_cls_oh'] = torch.cat([item['subject_cls_oh'],
                                            torch.zeros(len(negative_pairs),len(self.vocabs['objects']['itoname']))])
            item['object_cls_oh'] = torch.cat([item['object_cls_oh'],
                                            torch.zeros(len(negative_pairs),len(self.vocabs['objects']['itoname']))])
            item['target_oh'] = torch.cat([item['target_oh'],
                                            torch.zeros(len(negative_pairs),len(self.vocabs['relationships']['itoname']))])
            item['subject_features'] = torch.cat([item['subject_features'],
                                                  torch.zeros(len(negative_pairs), 2048)])
            item['object_features'] = torch.cat([item['object_features'],
                                                  torch.zeros(len(negative_pairs), 2048)])
            for neg_id, neg_pair in enumerate(negative_pairs):
                for i, btype in enumerate(['subject', 'object']):
                    box = objects[neg_pair[i]]
                    coords = [box['x'], box['y'], box['w'], box['h']]
                    coords[0] = coords[0] / width
                    coords[1] = coords[1] / height
                    coords[2] = coords[0] + coords[2]/width
                    coords[3] = coords[1] + coords[3]/height
                    cls_ = box['name']
                    cls_id = self.vocabs['objects']['nametoi'][box['name']]
                    features = oid_to_features[box['object_id'].replace('.png','.jpg')]
                    item[btype+'_boxes'].append(coords)
                    item[btype+'_boxes_raw'].append([box['x'], box['y'], box['w'], box['h']])
                    item[btype+'_cls'].append(cls_)
                    item[btype+'_cls_id'].append(cls_id)
                    item[btype+'_features'][neg_id+len(rels),:] = torch.Tensor(features)
                    item[btype+'_cls_oh'][neg_id+len(rels), cls_id] = 1
                item['target_cls'].append('__background__')
                item['target_cls_id'].append(
                    self.vocabs['relationships']['nametoi']['__background__']
                )
        item['target_cls_id'] = torch.LongTensor(item['target_cls_id'])
        for btype in 'subject object'.split():
            item[btype+'_cls_id'] = torch.LongTensor(item[btype+'_cls_id'])
            item[btype+'_boxes'] = torch.Tensor(item[btype+'_boxes'])
            item[btype+'_boxes_raw'] = torch.Tensor(item[btype+'_boxes_raw'])
        return item

    def __len__(self):
        return len(self.ids)
