#!/bin/bash
# python train_foundation_features.py -m --config 1 &
# python train_foundation_features.py -m --config 2 &
# python train_foundation_features.py -m --config 3 &
# python train_foundation_features.py -m --config 4 

# wait

predictions_dir="foundation_preds/foundation_validation_preds"
ground_truth_dir="foundation_preds/ground_truth"
out_dir="foundation_preds/foundation_out"
train_predictions_dir="foundation_preds/foundation_train_preds"
train_out="foundation_preds/training_preds"
python -m debugpy --listen 0.0.0.0:5678 ensemble.py --predictions_dir $predictions_dir --ground_truth_dir \
     $ground_truth_dir --out_dir $out_dir --train_predictions_dir $train_predictions_dir \
     --train_out $train_out