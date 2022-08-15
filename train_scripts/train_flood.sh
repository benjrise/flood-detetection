python train_flood.py -m --config 1 &
python train_flood.py -m --config 2 &
python train_flood.py -m --config 3 &
python train_flood.py -m --config 4 

wait

predictions_dir="flood_preds/flood_validation_preds"
gt_dir="flood_preds/ground_truth_flood"
out_dir="flood_preds/flood_out"
python ensemble_flood.py --predictions_dir $predictions_dir --ground_truth_dir $gt_dir --out_dir $out_dir