
python train_foundation_features.py -m --config 1 > log_dir/run1 &
python train_foundation_features.py -m --config 2 > log_dir/run2 &
python train_foundation_features.py -m --config 3 > log_dir/run3 &
python train_foundation_features.py -m --config 4 > log_dir/run4 &

wait

# predictions_dir="foundation_validation_preds"
# ground_truth_dir="ground_truth"
# out_dir="foundation_validation_preds"
# train_predictions_dir="foundation_training_preds"
# train_out="foundation_training_preds"
# python ensemble.py --predictions_dir $predictions_dir --ground_truth_dir $ground_truth_dir --out_dir $out_dir --train_foundation_features $train_foundation_features --train_out $train_out