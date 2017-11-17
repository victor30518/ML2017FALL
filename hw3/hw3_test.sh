#!/bin/bash 
wget -O cnn_model.h5 https://www.dropbox.com/s/2z3aeg78vqj9bb1/cnn_model.h5?dl=1
python3 cnn_test.py $1 $2