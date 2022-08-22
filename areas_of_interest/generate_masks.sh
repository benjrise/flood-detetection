#!/bin/bash
# RUN FROM THIS DIRECTORY

python data_prep/geojson_prep.py --root_dir areas_of_interest --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

python data_prep/create_masks.py --root_dir areas_of_interest --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

python data_prep/geojson_prep.py --root_dir areas_of_interest --aoi_dirs Louisiana-West_Test_Public

python data_prep/create_masks.py --root_dir areas_of_interest --aoi_dirs Louisiana-West_Test_Public

python data_prep/generate_train_val_test_csvs.py --root_dir areas_of_interest --aoi_dirs Louisiana-West_Test_Public --out_csv_basename test_data --val_percent 0.0 --out_dir areas_of_interest