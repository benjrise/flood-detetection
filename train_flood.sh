TRAIN_CSV="areas_of_interest/sn8_data_train.csv"
VAL_CSV="areas_of_interest/sn8_data_val.csv"
python train_flood.py --train_csv $TRAIN_CSV --val_csv $VAL_CSV --save_dir flood/ --model_name resnet34_siamese --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0