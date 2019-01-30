mkdir -p data/vqa
cd data/vqa
wget http://data.lip6.fr/cadene/block/tdiuc.tar.gz
tar -xzvf tdiuc.tar.gz

mkdir -p data/vqa/tdiuc/extract_rcnn
cd data/vqa/tdiuc/extract_rcnn
wget http://data.lip6.fr/cadene/block/tdiuc/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36.tar
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar
