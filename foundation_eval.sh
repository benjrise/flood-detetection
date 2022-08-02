
MODEL_PATH="experiments/{'TEST1'}_lr1.00e-04_bs8_27-07-2022-10-28/best_model.pth"
VAL_CSV="areas_of_interest/sn8_data_val.csv"
SAVE_PRED_DIR="experiments/{'TEST1'}_lr1.00e-04_bs8_27-07-2022-10-28/preds"

python foundation_eval.py --model_path $MODEL_PATH --in_csv $VAL_CSV --save_fig_dir $SAVE_PRED_DIR --gpu 0 --model_name hrnet

