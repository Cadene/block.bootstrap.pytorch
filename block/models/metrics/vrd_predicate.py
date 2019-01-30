import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.metrics.accuracy import accuracy
from ..criterions.vrd_bce import VRDBCELoss
from . import vrd_utils

class VRDPredicate(nn.Module):

    def __init__(self, engine=None, split='test', nb_classes=71):
        super(VRDPredicate, self).__init__()
        self.split = split
        self.k = nb_classes
        self.reset()
        if engine:
            engine.register_hook('%s_on_end_epoch'%split, self.calculate_metrics)

    def reset(self):
        self.tps = {50:[], 100:[]}
        self.fps = {50:[], 100:[]}
        self.scores = {50:[], 100:[]}
        self.total_num_gts = {50:0, 100:0}

    def forward(self, cri_out, net_out, batch):
        out = {}
        nb_classes = net_out['rel_scores'].size(1)
        out['rel_scores'] = net_out['rel_scores']
        det_labels, det_boxes = [], []
        gt_labels, gt_boxes = [], []
        rel_scores = torch.sigmoid(net_out['rel_scores'])
        acc_out = accuracy(
            rel_scores[(batch['target_cls_id'] > 0)],
            batch['target_cls_id'][batch['target_cls_id'] > 0],
            topk=[1,5])
        out['accuracy_top1'] = acc_out[0].item()
        out['accuracy_top5'] = acc_out[1].item()

        for batch_id in range(len(batch['rels'])):
            _index = (batch['batch_id'] == batch_id) * (batch['target_cls_id'] > 0)
            n_index = int(_index.int().sum().cpu().data)
            oid_to_box = {
                obj['object_id']: [obj['x'], obj['y'], obj['w'], obj['h']] \
                for obj in batch['objects'][batch_id]
            }
            rel_pred = rel_scores[_index]
            subj_pred_boxes = batch['subject_boxes_raw'][_index]
            obj_pred_boxes = batch['object_boxes_raw'][_index]

            subj_gt_boxes = np.array([
                oid_to_box[rel['subject_id']] \
                for rel in batch['rels'][batch_id]
            ])
            obj_gt_boxes = np.array([
                oid_to_box[rel['object_id']] \
                for rel in batch['rels'][batch_id]
            ])

            subj_gt_boxes[:,2] += subj_gt_boxes[:,0] 
            obj_gt_boxes[:,2] += obj_gt_boxes[:,0]
            subj_gt_boxes[:,3] += subj_gt_boxes[:,1]
            obj_gt_boxes[:,3] += obj_gt_boxes[:,1]

            _gt_boxes = np.concatenate([
                subj_gt_boxes[:,None,:], obj_gt_boxes[:,None,:]
            ], 1)
            _gt_labels = torch.cat([
                batch['subject_cls_id'][_index][:,None],
                batch['target_cls_id'][_index][:,None],
                batch['object_cls_id'][_index][:,None]
            ],1).cpu().data.numpy()
            top_score, top_pred = rel_pred.topk(self.k)
            top_score = top_score.cpu().data.numpy()
            top_pred = top_pred.cpu().data.numpy()
            _det_labels, _det_boxes = [], []
            for i in range(n_index):
                s = _gt_labels[i,0]
                o = _gt_labels[i,2]
                box_s = _gt_boxes[i,0]
                box_o = _gt_boxes[i,1]
                _det_labels.append(
                    np.concatenate([
                        np.ones((self.k,1)),
                        top_score[i][:,None],
                        np.ones((self.k,1)),
                        s*np.ones((self.k,1)),
                        top_pred[i][:,None],
                        [o]*np.ones((self.k,1))
                    ], 1))
                _det_boxes.append(
                    np.tile(
                        _gt_boxes[i][None,:,:],
                        (self.k,1,1)))
            det_labels.append(np.vstack(_det_labels))
            det_boxes.append(np.vstack(_det_boxes))
            gt_labels.append(_gt_labels)
            gt_boxes.append(_gt_boxes)

        for R in [50, 100]:
            _tp, _fp, _score, _num_gts = vrd_utils.eval_batch(
                [det_labels, det_boxes],
                [gt_labels, gt_boxes],
                num_dets=R)
            self.total_num_gts[R] += _num_gts
            self.tps[R] += _tp
            self.fps[R] += _fp
            self.scores[R] += _score

        return out

    def calculate_metrics(self):
        for R in [50, 100]:
            top_recall = vrd_utils.calculate_recall(
                R, self.tps, self.fps, self.scores, self.total_num_gts)
            Logger().log_value(f'{self.split}_epoch.predicate.R_{R}', top_recall, should_print=True)
        self.reset()
