from bootstrap.lib.options import Options
from .vqa_accuracies import VQAAccuracies
from .vrd_predicate import VRDPredicate
from .vrd_rel_phrase import VRDRelationshipPhrase

def factory(engine, mode):
    name = Options()['model.metric.name']
    metric = None

    if name == 'vqa_accuracies':
        if mode == 'train':
            split = engine.dataset['train'].split
            if split == 'train':
                metric = VQAAccuracies(engine,
                    mode='train',
                    open_ended=('tdiuc' not in Options()['dataset.name']),
                    tdiuc=True,
                    dir_exp=Options()['exp.dir'],
                    dir_vqa=Options()['dataset.dir'])
            elif split == 'trainval':
                metric = None
            else:
                raise ValueError(split)
        elif mode == 'eval':
            metric = VQAAccuracies(engine,
                mode='eval',
                open_ended=('tdiuc' not in Options()['dataset.name']),
                tdiuc=('tdiuc' in Options()['dataset.name'] or Options()['dataset.eval_split'] != 'test'),
                dir_exp=Options()['exp.dir'],
                dir_vqa=Options()['dataset.dir'])
        else:
            metric = None

    elif name == 'vrd_predicate':
        assert(engine.dataset[mode].mode == 'predicate')
        metric = VRDPredicate(engine, mode,
            nb_classes=Options()['model.network.predictor.dimensions'][-1])

    elif name == 'vrd_rel_phrase':
        assert(mode == 'eval')
        assert(engine.dataset[mode].split == 'test')
        assert(engine.dataset[mode].mode == 'rel_phrase')
        metric = VRDRelationshipPhrase(engine, mode)

    else:
        raise ValueError(name)
    return metric
