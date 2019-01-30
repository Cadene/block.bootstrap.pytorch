mkdir -p data/vqa
cd data/vqa
wget http://data.lip6.fr/cadene/block/vgenome.tar.gz
tar -xzvf vgenome.tar.gz

mkdir -p data/vqa/vgenome/extract_rcnn
cd data/vqa/vgenome/extract_rcnn
wget http://data.lip6.fr/cadene/block/vgenome/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36.tar
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar
