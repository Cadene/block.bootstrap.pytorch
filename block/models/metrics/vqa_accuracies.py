import os
import json
import torch
import torch.nn as nn
import numpy  as np
from scipy import stats
from collections import defaultdict
from bootstrap.lib.logger import Logger
from .vqa_accuracy import VQAAccuracy

class VQAAccuracies(nn.Module):

    def __init__(self,
            engine=None,
            mode='eval',
            open_ended=True,
            tdiuc=True,
            dir_exp='',
            dir_vqa=''):
        super(VQAAccuracies, self).__init__()
        self.engine = engine
        self.mode = mode
        self.open_ended = open_ended
        self.tdiuc = tdiuc
        self.dir_exp = dir_exp
        self.dir_vqa = dir_vqa
        self.dataset = engine.dataset[mode]
        self.ans_to_aid = self.dataset.ans_to_aid
        self.results = None
        self.results_testdev = None
        self.dir_rslt = None
        self.path_rslt = None

        # Module
        if self.tdiuc or self.dataset.split != 'test':
            self.accuracy = VQAAccuracy()
        else:
            self.accuracy = None

        if self.open_ended:
            engine.register_hook(
                '{}_on_start_epoch'.format(mode),
                self.reset_oe)
            engine.register_hook(
                '{}_on_end_epoch'.format(mode),
                self.compute_oe_accuracy)

            # if self.dataset.split == 'test':
            #     engine.register_hook(
            #         '{}_on_end_epoch'.format(mode),
            #         self.save_logits)

        if self.tdiuc:
            engine.register_hook(
                '{}_on_start_epoch'.format(mode),
                self.reset_tdiuc)
            engine.register_hook(
                '{}_on_end_epoch'.format(mode),
                self.compute_tdiuc_metrics)

    def reset_oe(self):
        self.results = []
        self.dir_rslt = os.path.join(
            self.dir_exp,
            'results',
            self.dataset.split,
            'epoch,{}'.format(self.engine.epoch))
        os.system('mkdir -p '+self.dir_rslt)
        self.path_rslt = os.path.join(
            self.dir_rslt,
            'OpenEnded_mscoco_{}_model_results.json'.format(
                self.dataset.get_subtype()))

        if self.dataset.split == 'test':
            self.results_testdev = []
            self.path_rslt_testdev = os.path.join(
                self.dir_rslt,
                'OpenEnded_mscoco_{}_model_results.json'.format(
                    self.dataset.get_subtype(testdev=True)))

            self.path_logits = os.path.join(self.dir_rslt, 'logits.pth')
            os.system('mkdir -p '+os.path.dirname(self.path_logits))

            self.logits = {}
            self.logits['aid_to_ans'] = self.engine.model.network.aid_to_ans
            self.logits['qid_to_idx'] = {}
            self.logits['tensor'] = None

            self.idx = 0

            path_aid_to_ans = os.path.join(self.dir_rslt, 'aid_to_ans.json')
            with open(path_aid_to_ans, 'w') as f:
                json.dump(self.engine.model.network.aid_to_ans, f)

    def save_logits(self):
        torch.save(self.logits, self.path_logits)

    def reset_tdiuc(self):
        self.pred_aids = []
        self.gt_aids = []
        self.gt_types = []
        self.gt_aid_not_found = 0
        self.res_by_type = defaultdict(list)

    def forward(self, cri_out, net_out, batch):
        out = {}
        if self.accuracy is not None:
            out = self.accuracy(cri_out, net_out, batch)

        if self.open_ended and self.dataset.split == 'test':
            logits = torch.nn.functional.softmax(net_out['logits'], dim=1).data.cpu()

        # add answers and answer_ids keys to net_out
        net_out = self.engine.model.network.process_answers(net_out)

        batch_size = len(batch['index'])
        for i in range(batch_size):
            # Open Ended Accuracy (VQA-VQA2)
            if self.open_ended:
                pred_item = {
                    'question_id': batch['question_id'][i],
                    'answer': net_out['answers'][i]
                }
                self.results.append(pred_item)

                if self.dataset.split == 'test':
                    if 'is_testdev' in batch and batch['is_testdev'][i]:
                        self.results_testdev.append(pred_item)

                    if self.logits['tensor'] is None:
                        self.logits['tensor'] = torch.FloatTensor(len(self.dataset), logits.size(1))

                    self.logits['tensor'][self.idx] = logits[i]
                    self.logits['qid_to_idx'][batch['question_id'][i]] = self.idx
                    
                    self.idx += 1

            # TDIUC metrics
            if self.tdiuc:
                qid = batch['question_id'][i]
                pred_aid = net_out['answer_ids'][i]
                self.pred_aids.append(pred_aid)

                gt_aid = batch['answer_id'][i]
                gt_ans = batch['answer'][i]
                gt_type = batch['question_type'][i]
                self.gt_types.append(gt_type)
                self.res_by_type[gt_type+'_pred'].append(pred_aid)

                if gt_ans in self.ans_to_aid:
                    self.gt_aids.append(gt_aid)
                    self.res_by_type[gt_type+'_gt'].append(gt_aid)
                    if gt_aid == pred_aid:
                        self.res_by_type[gt_type+'_t'].append(pred_aid)
                    else:
                        self.res_by_type[gt_type+'_f'].append(pred_aid)
                else:
                    self.gt_aids.append(-1)
                    self.res_by_type[gt_type+'_gt'].append(-1)
                    self.res_by_type[gt_type+'_f'].append(pred_aid)
                    self.gt_aid_not_found += 1

        return out

    def compute_oe_accuracy(self):
        with open(self.path_rslt, 'w') as f:
            json.dump(self.results, f)
        
        if self.dataset.split == 'test':
            with open(self.path_rslt_testdev, 'w') as f:
                json.dump(self.results_testdev, f)

        if 'test' not in self.dataset.split:
            call_to_prog = 'python -m block.models.metrics.compute_oe_accuracy '\
                + '--dir_vqa {} --dir_exp {} --dir_rslt {} --epoch {} --split {} &'\
                .format(self.dir_vqa, self.dir_exp, self.dir_rslt, self.engine.epoch, self.dataset.split)
            Logger()('`'+call_to_prog+'`')
            os.system(call_to_prog)
            # TODO: make it a subprocess call

    def compute_tdiuc_metrics(self):
        Logger()('{} of validation answers were not found in ans_to_aid'.format(self.gt_aid_not_found))
        
        accuracy = float(100*np.mean(np.array(self.pred_aids)==np.array(self.gt_aids)))
        Logger()('Overall Traditional Accuracy is {:.2f}'.format(accuracy))
        Logger().log_value('{}_epoch.tdiuc.accuracy'.format(self.mode), accuracy, should_print=False)
        
        types = list(set(self.gt_types))
        sum_acc = []
        eps = 1e-10

        Logger()('---------------------------------------')
        Logger()('Not using per-answer normalization...')
        for tp in types:
            acc = 100*(len(self.res_by_type[tp+'_t'])/len(self.res_by_type[tp+'_t']+self.res_by_type[tp+'_f']))
            sum_acc.append(acc+eps)
            Logger()("Accuracy for class '{}' is {:.2f}".format(tp, acc))
            Logger().log_value('{}_epoch.tdiuc.perQuestionType.{}'.format(self.mode, tp), acc, should_print=False)

        acc_mpt_a = float(np.mean(np.array(sum_acc)))
        Logger()('Arithmetic MPT Accuracy is {:.2f}'.format(acc_mpt_a))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_a'.format(self.mode), acc_mpt_a, should_print=False)

        acc_mpt_h = float(stats.hmean(sum_acc))
        Logger()('Harmonic MPT Accuracy is {:.2f}'.format(acc_mpt_h))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_h'.format(self.mode), acc_mpt_h, should_print=False)
        
        Logger()('---------------------------------------')
        Logger()('Using per-answer normalization...')
        for tp in types:
            per_ans_stat = defaultdict(int)
            for g,p in zip(self.res_by_type[tp+'_gt'],self.res_by_type[tp+'_pred']):
                per_ans_stat[str(g)+'_gt']+=1
                if g==p:
                    per_ans_stat[str(g)]+=1
            unq_acc = 0
            for unq_ans in set(self.res_by_type[tp+'_gt']):
                acc_curr_ans = per_ans_stat[str(unq_ans)]/per_ans_stat[str(unq_ans)+'_gt']
                unq_acc +=acc_curr_ans
            acc = 100*unq_acc/len(set(self.res_by_type[tp+'_gt']))
            sum_acc.append(acc+eps)
            Logger()("Accuracy for class '{}' is {:.2f}".format(tp, acc))
            Logger().log_value('{}_epoch.tdiuc.perQuestionType_norm.{}'.format(self.mode, tp), acc, should_print=False)

        acc_mpt_a = float(np.mean(np.array(sum_acc)))
        Logger()('Arithmetic MPT Accuracy is {:.2f}'.format(acc_mpt_a))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_a_norm'.format(self.mode), acc_mpt_a, should_print=False)

        acc_mpt_h = float(stats.hmean(sum_acc))
        Logger()('Harmonic MPT Accuracy is {:.2f}'.format(acc_mpt_h))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_h_norm'.format(self.mode), acc_mpt_h, should_print=False)
