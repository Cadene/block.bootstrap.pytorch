import torch
import numpy as np

def calculate_recall(R, tps, fps, scores, total_num_gts):
    tp = np.array(tps[R])
    fp = np.array(fps[R])
    score = np.array(scores[R])
    total_num_gt = total_num_gts[R]
    inds = np.argsort(score)
    inds = inds[::-1]
    tp = tp[inds]
    fp = fp[inds]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = (tp + 0.0) / total_num_gt
    top_recall = recall[-1] * 100
    return top_recall

def eval_batch(dets_file, gts_file, num_dets=50, ov_thresh=0.5):
    dets, det_bboxes = dets_file
    all_gts, all_gt_bboxes = gts_file
    num_img = len(dets)
    tp = []
    fp = []
    score = []
    total_num_gts = 0
    for i in range(num_img):
        gts = all_gts[i]
        gt_bboxes = all_gt_bboxes[i]
        num_gts = gts.shape[0]
        total_num_gts += num_gts
        gt_detected = np.zeros(num_gts)
        if isinstance(dets[i], np.ndarray) and dets[i].shape[0] > 0:
            det_score = np.log(dets[i][:, 0]) + np.log(dets[i][:, 1]) + np.log(dets[i][:, 2])
            inds = np.argsort(det_score)[::-1]
            if num_dets > 0 and num_dets < len(inds):
                inds = inds[:num_dets]
            top_dets = dets[i][inds, 3:]
            top_scores = det_score[inds]
            top_det_bboxes = det_bboxes[i][inds, :]
            num_dets = len(inds)
            for j in range(num_dets):
                ov_max = 0
                arg_max = -1
                for k in range(num_gts):
                    if gt_detected[k] == 0 and top_dets[j, 0] == gts[k, 0] and top_dets[j, 1] == gts[k, 1] and top_dets[j, 2] == gts[k, 2]:
                        ov = computeOverlap(top_det_bboxes[j, :, :], gt_bboxes[k, :, :])
                        if ov >= ov_thresh and ov > ov_max:
                            ov_max = ov
                            arg_max = k
                if arg_max != -1:
                    gt_detected[arg_max] = 1
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
                score.append(top_scores[j])
    return tp, fp, score, total_num_gts

def eval_batch_union(dets_file, gts_file, num_dets=50, ov_thresh=0.5):
    dets, det_bboxes = dets_file
    all_gts, all_gt_bboxes = gts_file
    num_img = len(dets)
    tp = []
    fp = []
    score = []
    total_num_gts = 0
    for i in range(num_img):
        gts = all_gts[i]
        gt_bboxes = all_gt_bboxes[i]
        gt_ubbs = []
        num_gts = gts.shape[0]
        for j in range(num_gts):
            gt_ubbs.append(getUnionBB(gt_bboxes[j, 0, :], gt_bboxes[j, 1, :]))
        total_num_gts += num_gts
        gt_detected = np.zeros(num_gts)
        if isinstance(dets[i], np.ndarray) and dets[i].shape[0] > 0:
            det_score = np.log(dets[i][:, 0]) + np.log(dets[i][:, 1]) + np.log(dets[i][:, 2])
            inds = np.argsort(det_score)[::-1]
            if num_dets > 0 and num_dets < len(inds):
                inds = inds[:num_dets]
            top_dets = dets[i][inds, 3:]
            top_scores = det_score[inds]
            top_det_bboxes = det_bboxes[i][inds, :]
            top_det_ubbs = []
            num_dets = len(inds)
            for j in range(num_dets):
                top_det_ubbs.append(getUnionBB(top_det_bboxes[j, 0, :], top_det_bboxes[j, 1, :]))
            for j in range(num_dets):
                ov_max = 0
                arg_max = -1
                for k in range(num_gts):
                    if gt_detected[k] == 0 and top_dets[j, 0] == gts[k, 0] and top_dets[j, 1] == gts[k, 1] and top_dets[j, 2] == gts[k, 2]:
                        ov = computeIoU(top_det_ubbs[j], gt_ubbs[k])
                        if ov >= ov_thresh and ov > ov_max:
                            ov_max = ov
                            arg_max = k
                if arg_max != -1:
                    gt_detected[arg_max] = 1
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
                score.append(top_scores[j])
    return tp, fp, score, total_num_gts

def computeOverlap(detBBs, gtBBs):
    aIoU = computeIoU(detBBs[0, :], gtBBs[0, :])
    bIoU = computeIoU(detBBs[1, :], gtBBs[1, :])
    return min(aIoU, bIoU)

def computeArea(bb):
    return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
    ibb = [max(bb1[0], bb2[0]), \
        max(bb1[1], bb2[1]), \
        min(bb1[2], bb2[2]), \
        min(bb1[3], bb2[3])]
    iArea = computeArea(ibb)
    uArea = computeArea(bb1) + computeArea(bb2) - iArea
    return (iArea + 0.0) / uArea

def getUnionBB(aBB, bBB):
    return [min(aBB[0], bBB[0]), \
            min(aBB[1], bBB[1]), \
            max(aBB[2], bBB[2]), \
            max(aBB[3], bBB[3])] 

def annot_to_gt(annot):
    gt_labels, gt_boxes = [], []
    for triplet in annot:
        gt_labels.append([triplet['subject']['category']+1,
                          triplet['predicate']+1,
                          triplet['object']['category']+1])
        sbox = triplet['subject']['bbox'] # ymin, ymax, xmin, xmax
        obox = triplet['object']['bbox']
        gt_boxes.append([
            [sbox[2], sbox[0], sbox[3], sbox[1]], #xmin, ymin, xmax, ymax
            [obox[2], obox[0], obox[3], obox[1]]
        ])
    return np.array(gt_labels), np.array(gt_boxes)

def item_to_det(item):
    det_labels, det_boxes = [], []
    rel_score = item['rel_score']
    s_score = item['cls_scores'][:,None,None].expand_as(rel_score).numpy()
    o_score = item['cls_scores'][None,:,None].expand_as(rel_score).numpy()
    r_score = rel_score.numpy()
    s_label = item['cls'][:,None,None].expand_as(rel_score).numpy()
    o_label = item['cls'][None,:,None].expand_as(rel_score).numpy()
    r_label = torch.arange(rel_score.size(-1))[None,None,:].expand_as(rel_score).numpy()
    s_index = torch.arange(len(item['rois']))[:,None,None].expand_as(rel_score).numpy()
    o_index = torch.arange(len(item['rois']))[None,:,None].expand_as(rel_score).numpy()
    det_labels = np.concatenate([x.reshape(-1)[:,None] for x in [s_score, r_score, o_score, s_label, r_label, o_label]], 1)

    keep = np.argsort(-np.prod(det_labels[:,:3], -1))[:500]
    s_index = s_index.reshape(-1)[keep]
    o_index = o_index.reshape(-1)[keep]
    rois = item['rois'].numpy()
    s_rois = rois[s_index.astype(np.int32)][:,None,:]
    o_rois = rois[o_index.astype(np.int32)][:,None,:]
    det_boxes = np.concatenate([s_rois,o_rois],1)

    det_labels = det_labels[keep]

    return det_labels, det_boxes
