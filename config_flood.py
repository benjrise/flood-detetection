from dataclasses import dataclass
import torch
import os
from utils.utils import check_dir_exists

@dataclass
class FloodConfig:
    MIXED_PRECISION : bool = True

    TRAIN_CSV : str = "areas_of_interest/sn8_data_train.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv"

    SAVE_DIR : str = "flood_debug"
    MULTI_SAVE_PATH : str = "flood_validation_preds"
    # MODEL_NAME : str = "resnet34"
    # SIAMESE : bool = False
    # MODEL_NAME = "resnet34_siamese"
    # SIAMESE : bool = True
    VAL_EVERY_N_EPOCH : int = 1
    NUM_EPOCHS : int = 30
    FINAL_EVAL_LOOP : bool = True

    USE_FOUNDATION_PREDS : bool = False

    MODEL_NAME : str = "efficientnet-b3"
    SIAMESE : bool = True
    RUN_NAME : str = "efficienetX2"
    
    PRETRAIN : bool = True
    LR : float = 1e-4
    BATCH_SIZE : int = 2
    VAL_BATCH_SIZE : int = 2

    IMG_SIZE : tuple = (2600, 2600)

    TRAIN_CROP : tuple = (1024, 1024)
    VALIDATION_CROP : tuple = (1024, 1024)

    GPU : int = 0

    MULTI_MODEL : bool = True
    SAVE_FIG : bool = True
    SAVE_PRED : bool = True

    GT_SAVE : bool = True
    GT_SAVE_PATH = "ground_truth_flood"
    TEST_CSV : str = ""
    
    def add_root_dir(self, root_dir):
        self.SAVE_DIR = os.path.join(root_dir, self.SAVE_DIR)
        self.MULTI_SAVE_PATH = os.path.join(root_dir, self.MULTI_SAVE_PATH)
        self.GT_SAVE_PATH = os.path.join(root_dir, self.GT_SAVE_PATH)
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
            os.chmod(root_dir, 0o777)

def get_multi_flood_config(num):
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=80
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    USE_FOUNDATION_PREDS=False

    FINAL_EVAL_LOOP=False

    root_dir = "flood_preds"
    SAVE_DIR="flood_save"
    MULTI_SAVE_PATH = "flood_validation_preds"

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
                
                USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,
                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                SIAMESE=True,
                RUN_NAME="efficientnet-b3_X2",
                MODEL_NAME="efficientnet-b3",
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
                USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,


                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                RUN_NAME="efficientnet-b3",
                MODEL_NAME="efficientnet-b3",
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
                USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,


                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                SIAMESE=True,
                RUN_NAME="efficientnet-b4_X2",
                MODEL_NAME="efficientnet-b4",
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
                USE_FOUNDATION_PREDS=USE_FOUNDATION_PREDS,


                BATCH_SIZE=2,
                VAL_BATCH_SIZE=2,
                RUN_NAME="efficientnet-b4_X2_JACC",
                MODEL_NAME="efficientnet-b4",
                SIAMESE=True,
                GPU=num-1,
                )

    config.add_root_dir(root_dir)
    return config
