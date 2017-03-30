#!/usr/bin/env bash

#python 1.1_convert_folder_to_dataset.py
python 1.2_split_dataset.py
python 1.3_duplicate_seed.py
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=1 python 2.1_features_extraction.py
