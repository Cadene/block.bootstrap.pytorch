import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skipthoughts
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.lib.options import Options
from .fusions.factory import factory as factory_fusion

def mask_softmax(x, lengths):#, dim=1)
    mask = torch.zeros_like(x).to(device=x.device, non_blocking=True)
    t_lengths = lengths[:,:,None].expand_as(mask)
    arange_id = torch.arange(mask.size(1)).to(device=x.device, non_blocking=True)
    arange_id = arange_id[None,:,None].expand_as(mask)

    mask[arange_id<t_lengths] = 1
    x = torch.exp(x)
    x = x * mask
    x = x / torch.sum(x, dim=1, keepdim=True).expand_as(x)
    return x

def factory_text_enc(vocab_words, opt):
    list_words = [vocab_words[i+1] for i in range(len(vocab_words))]
    if opt['name'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           list_words,
                           dropout=opt['dropout'],
                           fixed_emb=opt['fixed_emb'])
    else:
        raise NotImplementedError
    return seq2vec


class VQANet(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            attention={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(VQANet, self).__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.attention = Attention(**attention)
        self.fusion = factory_fusion(classif['fusion'])

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths']

        q = self.process_question(q, l)
        v = self.attention(q, v)
        logits = self.fusion([q, v])

        out = {'logits': logits}
        return out

    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = self.txt_enc._select_last(q, l)

        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out


class Attention(nn.Module):

    def __init__(self, mlp_glimpses=0, fusion={}):
        super(Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = factory_fusion(fusion)
        if self.mlp_glimpses > 0:
            self.linear0 = nn.Linear(fusion['output_dim'], 512)
            self.linear1 = nn.Linear(512, mlp_glimpses)

    def forward(self, q, v):
        alpha = self.process_attention(q, v)

        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        if alpha.size(2) > 1: # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha*v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            v_out = v_out.sum(1)
        return v_out

    def process_attention(self, q, v):
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:,None,:].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([
            q.contiguous().view(batch_size*n_regions, -1),
            v.contiguous().view(batch_size*n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        return alpha
