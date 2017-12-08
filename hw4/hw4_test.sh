#!/bin/bash 
wget -O model1_weight.h5 https://www.dropbox.com/s/244m9g1pv6blg1k/model1_weight.h5?dl=1
wget -O model2_weight.h5 https://www.dropbox.com/s/vezwre11g7mrwkz/model2_weight.h5?dl=1
python3 hw4_test.py $1 $2