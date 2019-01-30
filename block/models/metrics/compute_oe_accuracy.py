import argparse
import json
import random
import os
from os.path import join
import sys
from bootstrap.lib.logger import Logger
from block.external.VQA.PythonHelperTools.vqaTools.vqa import VQA
from block.external.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

def real_split_name(split):
    if split in ['train', 'val']:
        return split+'2014'
    elif split == 'test':
        return split+'2015'
    elif split == 'testdev':
        return 'test-dev2015'
    else:
        raise ValueError()

def main(dir_vqa, dir_exp, dir_rslt, epoch, split, cmd_line=True):
    real_split = real_split_name(split)
    if cmd_line:
        Logger(dir_exp, name='logs_{}_oe'.format(split))

    diranno  = join(dir_vqa, 'raw', 'annotations')
    annFile  = join(diranno, 'mscoco_%s_annotations.json' % (real_split))
    quesFile = join(diranno, 'OpenEnded_mscoco_%s_questions.json' % (real_split))
    vqa = VQA(annFile, quesFile)
    
    taskType    = 'OpenEnded'
    dataType    = 'mscoco'
    dataSubType = real_split
    resultType  = 'model'
    fileTypes = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
    
    [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
        ['%s/%s_%s_%s_%s_%s.json' % (dir_rslt, taskType, dataType,
            dataSubType, resultType, fileType) for fileType in fileTypes] 
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    quesIds = [int(d['question_id']) for d in json.loads(open(resFile).read())]
    # if split != 'train':
    #     annQuesIds = [int(d['question_id']) for d in json.loads(open(annFile).read())['annotations']]
    #     assert len(set(quesIds) - set(annQuesIds)) == 0, "Some questions in results are not in annotations"
    #     assert len(set(annQuesIds) - set(quesIds)) == 0, "Some questions in annotations are not in results"
    vqaEval.evaluate(quesIds=quesIds)
    
    mode = 'train' if 'train' in split else 'eval'

    Logger().log_value(f'{mode}_epoch.epoch', epoch)
    Logger().log_value(f'{mode}_epoch.overall', vqaEval.accuracy['overall'])

    for key in vqaEval.accuracy['perQuestionType']:
        rkey = key.replace(' ', '_')
        Logger().log_value(f'{mode}_epoch.perQuestionType.{rkey}', vqaEval.accuracy['perQuestionType'][key])

    for key in vqaEval.accuracy['perAnswerType']:
        rkey = key.replace(' ', '_')
        Logger().log_value(f'{mode}_epoch.perAnswerType.{rkey}', vqaEval.accuracy['perAnswerType'][key])

    Logger().flush()
    json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
    os.system('rm -rf '+dir_rslt)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vqa',   type=str, default='/local/cadene/data/vqa')
    parser.add_argument('--dir_exp', type=str, default='logs/16_12_13_20:39:55/')
    parser.add_argument('--dir_rslt', type=str, default='logs/16_12_13_20:39:55/results/train/epoch,1')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--split',  type=str, default='train')
    args = parser.parse_args()

    main(args.dir_vqa, args.dir_exp, args.dir_rslt, args.epoch, args.split)
    
    #json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    #json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    #json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
