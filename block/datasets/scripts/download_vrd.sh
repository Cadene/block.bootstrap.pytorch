mkdir -p data/vrd
cd data/vrd
wget http://data.lip6.fr/cadene/block/vrd/images.tar.gz
wget http://data.lip6.fr/cadene/block/vrd/annotations.tar.gz
wget http://data.lip6.fr/cadene/block/vrd/features.tar.gz
tar -xzvf images.tar.gz
tar -xzvf annotations.tar.gz
tar -xzvf features.tar.gz
