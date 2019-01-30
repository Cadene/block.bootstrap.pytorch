import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.optimizers.factory import factory_optimizer
from .lr_scheduler import ReduceLROnPlateau
from .lr_scheduler import BanOptimizer

def factory(model, engine):
    opt = Options()['optimizer']
    #optimizer = factory_optimizer(model)

    optimizer = BanOptimizer(engine,
        name=Options()['optimizer'].get('name', 'Adamax'),
        lr=Options()['optimizer']['lr'],
        gradual_warmup_steps=Options()['optimizer'].get('gradual_warmup_steps', [0.5, 2.0, 4]),
        lr_decay_epochs=Options()['optimizer'].get('lr_decay_epochs', [10, 20, 2]),
        lr_decay_rate=Options()['optimizer'].get('lr_decay_rate', .25))

    if opt.get('lr_scheduler', None):
        optimizer = ReduceLROnPlateau(optimizer, engine,
            **opt['lr_scheduler'])

    if opt.get('init', None) == 'glorot':
        for p in model.network.parameters():
            if p.dim()==1:
                p.data.fill_(0)
            elif p.dim()>=2:
                nn.init.xavier_uniform_(p.data)
            else:
                raise ValueError(p.dim())

    return optimizer
