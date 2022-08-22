python train_flood.py -v 1 --config 1 &
python train_flood.py -v 1 --config 2 &
python train_flood.py -v 1 --config 3 &
python train_flood.py -v 1 --config 4 

wait

python ensemble_flood.py

# predictions_dir="flood_preds/flood_validation_preds"
# gt_dir="flood_preds/ground_truth_flood"
# out_dir="flood_preds/flood_out"
# python ensemble_flood.py --predictions_dir $predictions_dir --ground_truth_dir $gt_dir --out_dir $out_dir