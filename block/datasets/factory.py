from bootstrap.lib.options import Options
from .vqa2 import VQA2
from .tdiuc import TDIUC
from .vrd import VRD
from .vg import VG
from .vqa_utils import ListVQADatasets

def factory(engine=None):
    opt = Options()['dataset']

    dataset = {}
    if opt.get('train_split', None):
        dataset['train'] = factory_split(opt['train_split'])
    if opt.get('eval_split', None):
        dataset['eval'] = factory_split(opt['eval_split'])

    return dataset

def factory_split(split):
    opt = Options()['dataset']
    shuffle = ('train' in split)

    if opt['name'] == 'tdiuc':
        dataset = TDIUC(
            dir_data=opt['dir'],
            split=split,
            batch_size=opt['batch_size'],
            nb_threads=opt['nb_threads'],
            pin_memory=Options()['misc']['cuda'],
            shuffle=shuffle,
            nans=opt['nans'],
            minwcount=opt['minwcount'],
            nlp=opt['nlp'],
            dir_rcnn=opt['dir_rcnn'])

    elif opt['name'] == 'vqa2':
        assert(split in ['train', 'val', 'test'])
        samplingans = (opt['samplingans'] and split == 'train')

        if opt['vg']:
            assert(opt['proc_split'] == 'trainval')

            # trainvalset 
            vqa2 = VQA2(
                dir_data=opt['dir'],
                split='train',
                nans=opt['nans'],
                minwcount=opt['minwcount'],
                nlp=opt['nlp'],
                proc_split=opt['proc_split'],
                samplingans=samplingans,
                dir_rcnn=opt['dir_rcnn'])

            vg = VG(
                dir_data=opt['dir_vg'],
                split='train',
                nans=10000,
                minwcount=0,
                nlp=opt['nlp'],
                dir_rcnn=opt['dir_rcnn_vg'])

            vqa2vg = ListVQADatasets(
                [vqa2,vg],
                split='train',
                batch_size=opt['batch_size'],
                nb_threads=opt['nb_threads'],
                pin_memory=Options()['misc.cuda'],
                shuffle=shuffle)

            if split == 'train':
                dataset = vqa2vg
            else:
                dataset = VQA2(
                    dir_data=opt['dir'],
                    split=split,
                    batch_size=opt['batch_size'],
                    nb_threads=opt['nb_threads'],
                    pin_memory=Options()['misc.cuda'],
                    shuffle=False,
                    nans=opt['nans'],
                    minwcount=opt['minwcount'],
                    nlp=opt['nlp'],
                    proc_split=opt['proc_split'],
                    samplingans=samplingans,
                    dir_rcnn=opt['dir_rcnn'])
                dataset.sync_from(vqa2vg)

        else:
            dataset = VQA2(
                dir_data=opt['dir'],
                split=split,
                batch_size=opt['batch_size'],
                nb_threads=opt['nb_threads'],
                pin_memory=Options()['misc.cuda'],
                shuffle=shuffle,
                nans=opt['nans'],
                minwcount=opt['minwcount'],
                nlp=opt['nlp'],
                proc_split=opt['proc_split'],
                samplingans=samplingans,
                dir_rcnn=opt['dir_rcnn'])

    elif opt['name'] == 'vrd':
        dataset = VRD(
            dir_data=opt['dir'],
            split=split,
            neg_ratio=opt['neg_ratio'],
            batch_size=opt['batch_size'],
            nb_threads=opt['nb_threads'],
            seed=Options()['misc.seed'],
            pin_memory=Options()['misc.cuda'],
            mode=opt['mode'])

    return dataset
