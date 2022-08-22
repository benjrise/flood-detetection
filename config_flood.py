from dataclasses import dataclass
import torch
import os
from utils.utils import check_dir_exists

@dataclass
class FloodConfig:

    MODEL : str = "efficientnet-b4"
    SIAMESE : bool = True
    RUN_NAME : str = "debug"

    TRAIN_CSV : str = "areas_of_interest/sn8_data_train.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv"
    TEST_CSV : str = ""

    SAVE_DIR : str = "flood_debug"
    MULTI_SAVE_PATH : str = "flood_validation_preds"
    

    VAL_EVERY_N_EPOCH : int = 1
    NUM_EPOCHS : int = 60
    FINAL_EVAL_LOOP : bool = True
    MIXED_PRECISION : bool = True
    USE_FOUNDATION_PREDS : bool = False
    GPU : int = 1
    LR : float = 1e-4
    BATCH_SIZE : int = 2
    VAL_BATCH_SIZE : int = 2

    PRETRAIN : bool = True
    IMG_SIZE : tuple = (2600, 2600)

    TRAIN_CROP : tuple = (1024, 1024)
    VALIDATION_CROP : tuple = (1024, 1024)

    BCE_WEIGHT : float = 1.00
    JACCARD_WEIGHT : float = 0.0


    MULTI_MODEL : bool = False
    SAVE_FIG : bool = False
    SAVE_PRED : bool = False
    TESTING : bool = True
    FINAL_SAVE_PATH : str = "flood_out"

    GT_SAVE : bool = False
    GT_SAVE_PATH : str = "ground_truth_flood"
    TEST_CSV : str = "areas_of_interest/test_data.csv"

    BAGGING : bool = False
    BAGGING_RATIO : float = 0.8

    
    def add_root_dir(self, root_dir):
        self.SAVE_DIR = os.path.join(root_dir, self.SAVE_DIR)
        self.MULTI_SAVE_PATH = os.path.join(root_dir, self.MULTI_SAVE_PATH)
        self.GT_SAVE_PATH = os.path.join(root_dir, self.GT_SAVE_PATH)
        self.FINAL_SAVE_PATH = os.path.join(root_dir, self.FINAL_SAVE_PATH)
    

def get_multi_flood_config(num) -> FloodConfig:
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=70
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    USE_FOUNDATION_PREDS=False
    
    BAGGING = False
    BAGGING_RATIO  = 0.8

    FINAL_EVAL_LOOP=True

    root_dir = "flood_cross"
    SAVE_DIR="flood_save"
    MULTI_SAVE_PATH = "flood_validation_preds"
    FINAL_SAVE_PATH = "flood_out"

    GT_SAVE = False
    GT_SAVE_PATH = "ground_truth_flood"

    MULTI_MODEL = True
    TESTING = True
    TEST_CSV = "areas_of_interest/test_data.csv"
    if num==1:
        config = FloodConfig(
            VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
            NUM_EPOCHS=NUM_EPOCHS,
            SAVE_DIR=SAVE_DIR,
            IMG_SIZE=IMG_SIZE,
            TRAIN_CROP=TRAIN_CROP,
            VALIDATION_CROP=VALIDATION_CROP,
            FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
            MULTI_SAVE_PATH=MULTI_SAVE_PATH,
            MULTI_MODEL=MULTI_MODEL,
            TEST_CSV=TEST_CSV,
            FINAL_SAVE_PATH=FINAL_SAVE_PATH,
            GT_SAVE=GT_SAVE,
            GT_SAVE_PATH=GT_SAVE_PATH,
            BAGGING=BAGGING,
            BAGGING_RATIO=BAGGING_RATIO,
            TESTING=TESTING,

            
            USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,
            BATCH_SIZE=2,
            VAL_BATCH_SIZE=2,
            SIAMESE=True,
            RUN_NAME="efficientnet-b4_1",
            MODEL="efficientnet-b4",
            GPU=num-1
            )

    if num==2:
        config = FloodConfig(
            VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
            NUM_EPOCHS=NUM_EPOCHS,
            SAVE_DIR=SAVE_DIR,
            IMG_SIZE=IMG_SIZE,
            TRAIN_CROP=TRAIN_CROP,
            VALIDATION_CROP=VALIDATION_CROP,
            FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
            MULTI_SAVE_PATH=MULTI_SAVE_PATH,
            MULTI_MODEL=MULTI_MODEL,
            TEST_CSV=TEST_CSV,
            FINAL_SAVE_PATH=FINAL_SAVE_PATH,
            GT_SAVE=GT_SAVE,
            GT_SAVE_PATH=GT_SAVE_PATH,
            BAGGING=BAGGING,
            BAGGING_RATIO=BAGGING_RATIO,

            BATCH_SIZE=2,
            VAL_BATCH_SIZE=2,
            RUN_NAME="efficientnet-b4_2",
            MODEL="efficientnet-b4",
            SIAMESE=True,
            GPU=num-1,
            )
    
    if num==3:
        config = FloodConfig(
            VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
            NUM_EPOCHS=NUM_EPOCHS,
            SAVE_DIR=SAVE_DIR,
            IMG_SIZE=IMG_SIZE,
            TRAIN_CROP=TRAIN_CROP,
            VALIDATION_CROP=VALIDATION_CROP,
            FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
            MULTI_SAVE_PATH=MULTI_SAVE_PATH,
            MULTI_MODEL=MULTI_MODEL,
            TEST_CSV=TEST_CSV,
            FINAL_SAVE_PATH=FINAL_SAVE_PATH,
            GT_SAVE=GT_SAVE,
            GT_SAVE_PATH=GT_SAVE_PATH,
            BAGGING=BAGGING,
            BAGGING_RATIO=BAGGING_RATIO,


            BATCH_SIZE=2,
            VAL_BATCH_SIZE=2,
            SIAMESE=True,
            RUN_NAME="efficientnet-b4_3",
            MODEL="efficientnet-b4",
            GPU=num-1,
            )

    if num==4:
        config = FloodConfig(
            VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
            NUM_EPOCHS=NUM_EPOCHS,
            SAVE_DIR=SAVE_DIR,
            IMG_SIZE=IMG_SIZE,
            TRAIN_CROP=TRAIN_CROP,
            VALIDATION_CROP=VALIDATION_CROP,
            FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
            MULTI_SAVE_PATH=MULTI_SAVE_PATH,
            MULTI_MODEL=MULTI_MODEL,
            TEST_CSV=TEST_CSV,
            FINAL_SAVE_PATH=FINAL_SAVE_PATH,
            GT_SAVE=GT_SAVE,
            GT_SAVE_PATH=GT_SAVE_PATH,
            BAGGING=BAGGING,
            BAGGING_RATIO=BAGGING_RATIO,


            BATCH_SIZE=2,
            VAL_BATCH_SIZE=2,
            RUN_NAME="efficientnet-b4_4",
            MODEL="efficientnet-b4",
            SIAMESE=True,
            GPU=num-1,
            )

    config.add_root_dir(root_dir)
    return config


def get_multi_flood_config2(num):
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=70
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    USE_FOUNDATION_PREDS=False

    FINAL_EVAL_LOOP=True

    root_dir = "debug_ensemble"
    SAVE_DIR="flood_save"
    MULTI_SAVE_PATH = "flood_validation_preds"
    FINAL_SAVE_PATH = "flood_out"

    GT_SAVE = True
    GT_SAVE_PATH = "ground_truth_flood"

    MULTI_MODEL = True
    TEST_CSV = "areas_of_interest/sn8_data_val.csv"
    if num==1:
        config = FloodConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                TEST_CSV=TEST_CSV,
                FINAL_SAVE_PATH=FINAL_SAVE_PATH,
                GT_SAVE=GT_SAVE,
                GT_SAVE_PATH=GT_SAVE_PATH,
                
                USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,
                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                SIAMESE=True,
                RUN_NAME="efficientnet-b4",
                MODEL="efficientnet-b4",
                GPU=num-1
                )

    if num==2:
        config = FloodConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=(512, 512),
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                TEST_CSV=TEST_CSV,
                FINAL_SAVE_PATH=FINAL_SAVE_PATH,
                GT_SAVE=GT_SAVE,
                GT_SAVE_PATH=GT_SAVE_PATH,


                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                RUN_NAME="efficientnet-b4_train_crop",
                MODEL="efficientnet-b4",
                SIAMESE=True,
                GPU=num-1,
                )
    
    if num==3:
        config = FloodConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=(512, 512),
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                TEST_CSV=TEST_CSV,
                FINAL_SAVE_PATH=FINAL_SAVE_PATH,
                GT_SAVE=GT_SAVE,
                GT_SAVE_PATH=GT_SAVE_PATH,

                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                SIAMESE=True,
                RUN_NAME="hrnet_train_crop",
                MODEL="hrnet",
                GPU=num-1,
                )

    if num==4:
        config = FloodConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                TEST_CSV=TEST_CSV,
                FINAL_SAVE_PATH=FINAL_SAVE_PATH,
                GT_SAVE=GT_SAVE,
                GT_SAVE_PATH=GT_SAVE_PATH,

                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                RUN_NAME="hrnet",
                MODEL="hrnet",
                SIAMESE=True,
                GPU=num-1,
                )

    config.add_root_dir(root_dir)
    return config
