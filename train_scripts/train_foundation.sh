#!/bin/bash
version=1
python train_foundation_features.py -v $version --config 1 &
python train_foundation_features.py -v $version --config 2 &
python train_foundation_features.py -v $version --config 3 &
python train_foundation_features.py -v $version --config 4 

wait

python ensemble.py -v $version