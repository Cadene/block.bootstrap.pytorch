import torch
import torch.nn as nn
from bootstrap.models.metrics.accuracy import accuracy

class VQAAccuracy(nn.Module):

    def __init__(self, topk=[1,5]):
        super(VQAAccuracy, self).__init__()
        self.topk = topk

    def __call__(self, cri_out, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        class_id = batch['class_id'].data.cpu()
        acc_out = accuracy(logits,
                           class_id,
                           topk=self.topk)

        for i, k in enumerate(self.topk):
            out['accuracy_top{}'.format(k)] = acc_out[i]
        return out
