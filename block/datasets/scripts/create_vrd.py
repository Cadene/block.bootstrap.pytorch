import os
from os import path as osp
import json
import torch
import torch.utils.data as data

from PIL import Image
import numpy as np
import random

import xml.etree.ElementTree as ET
import cv2
import sys
sys.path.insert(0,'./')
from supervqa.datasets import process_utils as p_utils

from tqdm import tqdm

raw_dir = "/local/benyounes/data/vrd/"
xml_folder = osp.join(raw_dir, 'xml')
split = "train"
img_relative = osp.join('sg_dataset', 'sg_%s_images' % split)
objs_vocab = json.load(open(osp.join(raw_dir,'objects.json')))
preds_vocab = json.load(open(osp.join(raw_dir,'predicates.json')))

def _create_xml(fname, rels):
    # Create root
    try:
        im = cv2.imread(osp.join(raw_dir, img_relative, fname))
        height, width, _ = im.shape
    except:
        import ipdb; ipdb.set_trace()
    name = fname.split('.')[0]
    # All the objects in the triplet
    objects = [r[k]['bbox'] + [r[k]['category']] \
               for r in rels for k in "subject object".split()]
    s_objects = [str(o) for o in objects] # To make it hashable
    sobj_to_oid = dict()
    for sobj in s_objects:
        if sobj not in sobj_to_oid:
            sobj_to_oid[sobj] = str(len(sobj_to_oid))+'_'+name
    root = ET.Element('annotation')
    source = ET.SubElement(root, 'source')
    size = ET.SubElement(root, 'size')
    ET.SubElement(source, 'image_id').text = name
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'width').text = str(width)

    parsed_objs = set()
    for obj in objects:
        oid = sobj_to_oid[str(obj)]
        if oid not in parsed_objs:
            parsed_objs.add(oid)
            obj_el = ET.SubElement(root, "object")
            ET.SubElement(obj_el, 'object_id').text = oid
            ET.SubElement(obj_el,'name').text = objs_vocab[obj[-1]]
            bndbox = ET.SubElement(obj_el, 'bndbox')
            ET.SubElement(bndbox,'ymin').text = str(obj[0])
            ET.SubElement(bndbox,'ymax').text = str(obj[1])
            ET.SubElement(bndbox,'xmin').text = str(obj[2])
            ET.SubElement(bndbox,'xmax').text = str(obj[3])

    for rel in rels:
        sobj = str(rel['object']['bbox'] + [rel['object']['category']])
        ssubj = str(rel['subject']['bbox'] + [rel['subject']['category']])
        oid = sobj_to_oid[sobj]
        sid = sobj_to_oid[ssubj]
        rel_el = ET.SubElement(root, 'relation')
        ET.SubElement(rel_el,'object_id').text = oid
        ET.SubElement(rel_el,'subject_id').text = sid
        ET.SubElement(rel_el,'predicate').text = preds_vocab[rel['predicate']]

    tree = ET.ElementTree(root)
    return tree, name

def _convert_vocabs():
    vocab_dir = osp.join(raw_dir, '%d-0-%d') % \
                (len(objs_vocab), len(preds_vocab))
    if os.path.exists(vocab_dir):
        return None
    os.system('mkdir -p %s' % vocab_dir)
    rel_vocab_file = open(osp.join(vocab_dir, 'relations_vocab.txt'), 'w')
    for w in preds_vocab:
        rel_vocab_file.write(w+'\n')
    rel_vocab_file.close()

    obj_vocab_file = open(osp.join(vocab_dir, 'objects_vocab.txt'), 'w')
    for w in objs_vocab:
        obj_vocab_file.write(w+'\n')
    obj_vocab_file.close()

    return None
def main():
    _convert_vocabs()
    annot_path = osp.join(raw_dir, 'annotations_%s.json' % split)
    annotations = json.load(open(annot_path))

    # Create xml folder
    if not osp.exists(xml_folder):
        os.system('mkdir -p %s' % xml_folder)
    # Go through all the annotations
    split_file = open(osp.join(raw_dir, split+'.txt'), 'w')
    for fname in tqdm(sorted(annotations)):
        rels = annotations[fname]

        if not fname.endswith('.jpg'):
            new_fname = fname.split('.')[0]+'.jpg'
            os.system('convert %s %s' % (osp.join(raw_dir, img_relative, fname),
                                         osp.join(raw_dir, img_relative, new_fname)))
            fname = new_fname
        xml_file, name = _create_xml(fname, rels)
        imname = osp.join(img_relative, fname)
        xmlname = osp.join('xml',name+'.xml')
        xml_file.write(osp.join(raw_dir,xmlname))
        split_file.write("%s %s\n" % (imname, xmlname))
    split_file.close()


if __name__ =="__main__":
    main()
    #name, rels, new_tree = main()
    #ex_tree = ET.parse("/local/cadene/data/faster-rcnn.pytorch/vgenome/xml/9.xml")
    #
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    print("im info")
    #    print(p_utils.extract_img_info(root))
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    _objects = [p_utils.extract_obj(obj) for obj in root.findall('object')]
    #    print("Len objects = %d" % len(_objects))
    #    print(_objects[0])
    #for root in [ex_tree.getroot(), new_tree.getroot()]:
    #    _relationships = [p_utils.extract_rel(rel) for rel in root.findall('relation')]
    #    print("Len rels = %d" % len(_relationships))
    #    print(_relationships[0])
