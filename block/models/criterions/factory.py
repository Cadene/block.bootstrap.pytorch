from bootstrap.lib.options import Options
from .vqa_cross_entropy import VQACrossEntropyLoss
from .vrd_bce import VRDBCELoss

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split
    eval_only = 'train' not in engine.dataset

    if name == 'vqa_cross_entropy':
        if split == 'test':
            return None
        criterion = VQACrossEntropyLoss()

    elif name == 'vrd_bce':
        if split == 'test' and eval_only:
            return None
        criterion = VRDBCELoss()

    else:
        raise ValueError(name)

    return criterion
