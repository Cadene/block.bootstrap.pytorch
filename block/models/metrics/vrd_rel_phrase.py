import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.metrics.accuracy import accuracy
from . import vrd_utils

class VRDRelationshipPhrase(nn.Module):

    def __init__(self, engine=None, split='test'):
        super(VRDRelationshipPhrase, self).__init__()
        self.split = split
        self.activation = torch.sigmoid
        self.reset()
        self.dataset = engine.dataset['eval']
        if engine:
            engine.register_hook(f'{split}_on_end_epoch', self.calculate_metrics)

    def reset(self):
        # Relationship task metrics
        self.tps = {50:[], 100:[]}
        self.fps = {50:[], 100:[]}
        self.scores = {50:[], 100:[]}
        self.total_num_gts = {50:0, 100:0}
        # Phrase task metrics
        self.tps_union = {50:[], 100:[]}
        self.fps_union = {50:[], 100:[]}
        self.scores_union = {50:[], 100:[]}
        self.total_num_gts_union = {50:0, 100:0}

    def forward(self, cri_out, net_out, batch):
        det_labels, det_boxes, gt_labels, gt_boxes = [], [], [], []

        npairs_count = 0
        nboxes_count = 0
        batch_size = len(batch['idx'])
        total_npairs = net_out['rel_scores'].shape[0]
        nclasses = net_out['rel_scores'].shape[1]

        rel_scores = self.activation(net_out['rel_scores'])

        for idx in range(batch_size):
            image_id = batch['image_id'][idx]
            annot = self.dataset.json_raw[image_id]
            _gt_labels, _gt_boxes = vrd_utils.annot_to_gt(annot)

            nboxes = batch['n_boxes'][idx]
            npairs = batch['n_pairs'][idx]
            if nboxes == 0:
                _det_labels = np.zeros((0,6))
                _det_boxes = np.zeros((0,2,4))
            else:
                begin_ = npairs_count
                end_ = begin_ + npairs

                rel_score = rel_scores[begin_:end_].view(
                    nboxes,
                    nboxes,
                    nclasses)
                rel_score = rel_score.data.cpu()
                rel_prob = rel_score.topk(10)

                begin_ = nboxes_count
                end_ = begin_ + nboxes

                item = {
                    'rel_score': rel_score,
                    'rel_prob': rel_prob,
                    'rois': batch['rois_nonorm'][begin_:end_].cpu(),
                    'cls': batch['cls'][begin_:end_].cpu(),
                    'cls_scores': batch['cls_scores'][begin_:end_].cpu()
                }
                _det_labels, _det_boxes = vrd_utils.item_to_det(item)
            
            npairs_count += npairs
            nboxes_count += nboxes

            det_labels.append(_det_labels)
            det_boxes.append(_det_boxes)
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

            _tp, _fp, _score, _num_gts = vrd_utils.eval_batch_union(
                [det_labels, det_boxes],
                [gt_labels, gt_boxes],
                num_dets=R)
            self.total_num_gts_union[R] += _num_gts
            self.tps_union[R] += _tp
            self.fps_union[R] += _fp
            self.scores_union[R] += _score

        return {}

    def calculate_metrics(self):
        for R in [50, 100]:
            top_recall = vrd_utils.calculate_recall(
                R, self.tps, self.fps, self.scores, self.total_num_gts)
            Logger().log_value(f'{self.split}_epoch.rel.R_{R}', top_recall, should_print=True)

            top_recall = vrd_utils.calculate_recall(
                R, self.tps_union, self.fps_union, self.scores_union, self.total_num_gts_union)
            Logger().log_value(f'{self.split}_epoch.phrase.R_{R}', top_recall, should_print=True)
        self.reset()
