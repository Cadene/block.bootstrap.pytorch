mkdir -p data/vqa
cd data/vqa
wget http://data.lip6.fr/cadene/block/vqa2.tar.gz
wget http://data.lip6.fr/cadene/block/coco.tar.gz
tar -xzvf vqa2.tar.gz
tar -xzvf coco.tar.gz

mkdir -p data/vqa/coco/extract_rcnn
cd data/vqa/coco/extract_rcnn
wget http://data.lip6.fr/cadene/block/coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36.tar
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar
