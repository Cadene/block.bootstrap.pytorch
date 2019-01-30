# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import Variable
# from bootstrap.lib.logger import Logger

# class KLDivLoss(nn.Module):

#     def __init__(self, engine=None):
#         super(KLDivLoss, self).__init__()
#         self.loss = nn.KLDivLoss()
#         if engine is not None:
#             engine.register_hook('train_on_print', self.print)

#     def print(self):
#         Logger()('       loss_kl: {}'.format(self.loss_out))

#     def forward(self, net_out, batch):
#         out = {}
#         logits = F.log_softmax(net_out['logits'], dim=1)
#         target = Variable(logits.data.new(logits.data.size()).fill_(0), requires_grad=False)
#         target.data.scatter_(1, batch['class_id'].data, 1)
#         target = F.softmax(target, dim=1)
#         out['loss'] = self.loss(logits, target)
#         self.loss_out = out['loss'].data[0]
#         return out
