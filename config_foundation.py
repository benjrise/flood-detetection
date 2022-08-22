from dataclasses import dataclass
import torch
import os
from utils.utils import check_dir_exists

@dataclass
class FoundationConfig:
    
    RUN_NAME : str = "SCHEDULER"
    MODEL : str = "efficientnet-b4"

    # CSV FILES
    TRAIN_CSV : str = "areas_of_interest/sn8_full_data.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv" # for individual model ensembles
    TEST_CSV : str = "areas_of_interest/test_data.csv" # goes into save predictions before ensemble loop


    # TRAINING PARAMS
    VAL_EVERY_N_EPOCH : int = 1
    NUM_EPOCHS : int = 1
    GPU : int = 0
    BATCH_SIZE : int = 4
    VAL_BATCH_SIZE : int = 4
    MIXED_PRECISION : bool = True
    PRETRAIN : bool = True


    # OUT DIRS
    SAVE_DIR : str = "upsample_experiments2" # save model checkpoints and logs here
    PATH_SAVE_PREDS :   str = "foundation_predictions" # save model predictions here
    PATH_SAVE_TRAIN_PREDS : str = "foundation_training_predictions" # save training predictions here
    PATH_GT_SAVE : str = "ground_truth" # save gt here
    PATH_SAVE_FINAL_TRAIN_PREDS : str = "training_preds" # save training example predictions here
    PATH_SAVE_FINAL_PREDS : str = "foundation_out" # save final ensembled predictions here

    # THIS IS THE LOCATION OF THE MODEL CHECKPOINTS WHEN THEY ARE SAVED THIS IS ADDED BY
    # set_checkpoint_dir function
    CHECKPOINT_DIR : str = ''

    # IMG_SIZE AFTER UPSAMPLING
    IMG_SIZE : tuple = (2600, 2600)

    # LOSS TERMS
    # ROAD
    SOFT_DICE_LOSS_WEIGHT : float = 0.25 
    ROAD_FOCAL_LOSS_WEIGHT :float = 0.75 
    # BUILDING
    BCE_LOSS_WEIGHT : float = 0.75
    BUILDING_FOCAL_WEIGHT : float = 0.0
    BUILDING_JACCARD_WEIGHT : float = 0.25
    BUILDING_SOFT_DICE : float = 0.0
    # ROAD VS BUILDING PREFERENCE
    ROAD_LOSS_WEIGHT : float = 0.5
    BUILDING_LOSS_WEIGHT : float= 0.5

    # AUGMENTATION 
    NORMALIZE : float = True
    TRAIN_CROP : tuple = (1024, 1024)
    VALIDATION_CROP : tuple = (1024,1024)
    P_FLIPS : float = 0

    # LR SCHEDULER
    PATIENCE : int = 8
    FACTOR : float = 0.5

    # FINAL EVAL LOOP
    FINAL_EVAL_LOOP : bool = False
    SAVE_FIG : bool = None
    SAVE_PRED : bool = None

    # CUDNN FLAGS
    CUDNN_BENCHMARK : bool = False
    CUDNN_ENABLED : bool = False

    # OPTIMIZER PARAMS
    OPTIMIZER : str = 'adamw'
    LR : float = 1e-4
    MOMENTUM : float = 0.0
    NEW_SCHEDULER : bool = False

    # MULTI_MODEL
    MULTI_MODEL : bool = True
    MULTI_MODEL_EVAL : bool = False
    SAVE_TRAINING_PREDS : bool = False
    TESTING : bool = True # this flag so we don't accidently try to tranform 0s in save_preds_loop

    # BAGGING
    BAGGING : bool = True
    BAGGING_RATIO : float = 0.8
    
    def __post_init__(self):
        if self.OPTIMIZER == 'sgd':
            self.LR = 1e-2
            self.MOMENTUM = 0.9
        else:
            self.LR = 1e-4
        
        if not self.TEST_CSV:
            self.TEST_CSV = self.VALIDATION_LOOP_CSV

        # In case MULTI_MODEL_EVAL is not turned off as can't evalutate test set
        if self.TESTING == True:
            self.MULTI_MODEL_EVAL = False

    def set_checkpoint_dir(self, dir):
        self.CHECKPOINT_DIR = dir

    def get_optimizer(self, model):
        eps = 1e-7
        if self.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(model.parameters(), 
                                    lr=self.LR, 
                                    eps=eps,
                                    )
        if self.OPTIMIZER == 'adam':
            return torch.optim.AdamW(model.parameters(), 
                                    lr=self.LR, 
                                    eps=eps,
                                    )
        if self.OPTIMIZER == 'sgd':
            return torch.optim.SGD(model.parameters(), 
                                    lr=self.LR,
                                    momentum=self.MOMENTUM)

    def add_root_dir(self, root_dir):
        check_dir_exists(root_dir)
        self.SAVE_DIR = os.path.join(root_dir, self.SAVE_DIR)
        self.PATH_SAVE_PREDS = os.path.join(root_dir, self.PATH_SAVE_PREDS)
        self.PATH_SAVE_TRAIN_PREDS = os.path.join(root_dir, self.PATH_SAVE_TRAIN_PREDS)
        self.PATH_GT_SAVE = os.path.join(root_dir, self.PATH_GT_SAVE)
        self.PATH_SAVE_FINAL_PREDS = os.path.join(root_dir, self.PATH_SAVE_FINAL_PREDS)
        self.PATH_SAVE_FINAL_TRAIN_PREDS = os.path.join(root_dir, self.PATH_SAVE_FINAL_TRAIN_PREDS)

    def get_dirs():
        pass

def get_multi_config(num):
    TRAIN_CSV="areas_of_interest/sn8_full_data.csv"
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=70
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    
    FINAL_EVAL_LOOP=True
    SAVE_PRED=False
    SAVE_FIG=False

    MULTI_MODEL = True
    
    root_dir = "cross_folds"
    SAVE_DIR="foundation_save"
    PATH_GT_SAVE = "ground_truth"

    SAVE_TRAINING_PREDS = False
    PATH_SAVE_TRAIN_PREDS = "foundation_train_preds"
    PATH_SAVE_FINAL_TRAIN_PREDS = "training_preds"
    
    BAGGING = False
    BAGGING_RATIO = 0.8
    
    PATH_SAVE_PREDS = "test_preds"
    PATH_SAVE_FINAL_PREDS = "test_out"
    TEST_CSV = "areas_of_interest/test_data.csv"
    #TEST_CSV = "areas_of_interest/sn8_data_val.csv"
    TESTING = True
    MULTI_MODEL_EVAL = False

    if num==1:
        config = FoundationConfig(
                TRAIN_CSV=TRAIN_CSV,
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_1",
                MODEL="efficientnet-b4",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==2:
        config = FoundationConfig(
                TRAIN_CSV=TRAIN_CSV,
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_2",
                MODEL="efficientnet-b4",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    if num==3:
        config = FoundationConfig(
                TRAIN_CSV=TRAIN_CSV,
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_3",
                MODEL="efficientnet-b4",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==4:
        config = FoundationConfig(
                TRAIN_CSV=TRAIN_CSV,
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_4",
                MODEL="efficientnet-b4",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    config.add_root_dir(root_dir)
    return config

def get_multi_config2(num):
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=70
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    
    FINAL_EVAL_LOOP=True
    SAVE_PRED=False
    SAVE_FIG=False

    MULTI_MODEL = True
    
    root_dir = "focal_dice_test"
    SAVE_DIR="foundation_save"
    PATH_GT_SAVE = "ground_truth"

    SAVE_TRAINING_PREDS = False
    PATH_SAVE_TRAIN_PREDS = "foundation_train_preds"
    PATH_SAVE_FINAL_TRAIN_PREDS = "training_preds"
    
    BAGGING = False
    BAGGING_RATIO = 0.9
    
    PATH_SAVE_PREDS = "foundation_validation_preds"
    PATH_SAVE_FINAL_PREDS = "foundation_out"
    # TEST_CSV = "areas_of_interest/test_data.csv"
    TEST_CSV = "areas_of_interest/sn8_data_val.csv"
    TESTING = True
    MULTI_MODEL_EVAL = False

    if num==1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_1_DICE_FOCAL",
                MODEL="efficientnet-b4",
                BUILDING_SOFT_DICE=0.5,
                BUILDING_FOCAL_WEIGHT=0.5,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==2:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3_1_DICE_FOCAL",
                MODEL="efficientnet-b3",
                BUILDING_SOFT_DICE=0.5,
                BUILDING_FOCAL_WEIGHT=0.5,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    if num==3:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_DICE",
                MODEL="efficientnet-b4",
                BUILDING_SOFT_DICE=1.00,
                BCE_LOSS_WEIGHT=0.0,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==4:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_DICE",
                MODEL="efficientnet-b3",
                BUILDING_SOFT_DICE=1.0,
                BUILDING_FOCAL_WEIGHT=0.0,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    config.add_root_dir(root_dir)
    return config



def get_multi_config3(num):
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=70
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    
    FINAL_EVAL_LOOP=True
    SAVE_PRED=False
    SAVE_FIG=False

    MULTI_MODEL = True
    
    root_dir = "dice_test"
    SAVE_DIR="foundation_save"
    PATH_GT_SAVE = "ground_truth"

    SAVE_TRAINING_PREDS = False
    PATH_SAVE_TRAIN_PREDS = "foundation_train_preds"
    PATH_SAVE_FINAL_TRAIN_PREDS = "training_preds"
    
    BAGGING = False
    BAGGING_RATIO = 0.9
    
    PATH_SAVE_PREDS = "foundation_validation_preds"
    PATH_SAVE_FINAL_PREDS = "foundation_out"
    # TEST_CSV = "areas_of_interest/test_data.csv"
    TEST_CSV = "areas_of_interest/sn8_data_val.csv"
    TESTING = True
    MULTI_MODEL_EVAL = False

    if num==1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_1_DICE",
                MODEL="efficientnet-b4",
                BUILDING_SOFT_DICE=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==2:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_2_JACC",
                MODEL="efficientnet-b4",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    if num==3:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,


                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3_3_JACC",
                MODEL="efficientnet-b3",
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )

    if num==4:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                PATH_SAVE_PREDS=PATH_SAVE_PREDS,
                PATH_SAVE_TRAIN_PREDS=PATH_SAVE_TRAIN_PREDS,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                PATH_SAVE_FINAL_PREDS=PATH_SAVE_FINAL_PREDS,
                PATH_GT_SAVE=PATH_GT_SAVE,
                PATH_SAVE_FINAL_TRAIN_PREDS=PATH_SAVE_FINAL_TRAIN_PREDS,
                SAVE_TRAINING_PREDS=SAVE_TRAINING_PREDS,
                TESTING=TESTING,
                SAVE_PRED=SAVE_PRED,
                SAVE_FIG=SAVE_FIG,

                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b4_2_JACC",
                MODEL="efficientnet-b3",
                BUILDING_SOFT_DICE=0.25,
                BCE_LOSS_WEIGHT=0.75,
                GPU=num-1,
                BAGGING=BAGGING,
                BAGGING_RATIO=BAGGING_RATIO,
                )
    
    config.add_root_dir(root_dir)
    return config

def get_config1(num):
    if num == 1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),     
        )

    elif num == 2:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX3",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                
        )
    elif num==3:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX2",
                MODEL="resnet34",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                
        )
    elif num==4:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX3",
                MODEL="resnet34",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
        )
    elif num == 5:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2_JAC",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
        )
    elif num == 6:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX3_JAC",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
        )
    elif num==7:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX2_JACC",
                MODEL="resnet34",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
        )
    elif num== 8:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX3_JACC",
                MODEL="resnet34",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
                
        )
    elif num==9:
        # Test this works
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="unet_UPSAMPLEX2_JACC",
                MODEL="unet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
                )
    
    elif num == 10:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2_JAC_FOC",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                BUILDING_FOCAL_WEIGHT=0.25
        )
    elif num == 11:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX3_JAC_FOC",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                
                
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                BUILDING_FOCAL_WEIGHT=0.25
        )
    elif num==12:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX2_JACC_FOC",
                MODEL="resnet34",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                BUILDING_FOCAL_WEIGHT=0.25
        )
    elif num==13:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34_UPSAMPLEX3_JACC_FOC",
                MODEL="resnet34",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                BUILDING_FOCAL_WEIGHT=0.25
                
        )
    elif num==14:
        # Test this works
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="efficinetb3_UPSAMPLEX2_JACC_FOC",
                MODEL="efficientnet-b3",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
            
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                BUILDING_FOCAL_WEIGHT=0.25
                )
    if num == 15:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2_NOPRETRAIN",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                PRETRAIN=False
        )
    if num ==17: 
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2_SGD",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                OPTIMIZER="sgd"
        )
    if num == 18:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="efficinetb3_UPSAMPLEX2_JACC",
                MODEL="efficientnet-b3",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
            
                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75,
                )
    
    if num == 18:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="efficinetb3_UPSAMPLEX2",
                MODEL="efficientnet-b3",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
        
                )

    if num==19:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="resnet34",
                MODEL="resnet34",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                )

    if num==20:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="hrnet",
                MODEL="hrnet",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                )

    if num==21:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="efficientnet-b3",
                MODEL="hrnet",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                )

    return config
    
def get_config(num):
    if num == 1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=65,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="BATCH_SIZE_4",
                MODEL="efficientnet-b4",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
        )

    elif num==2:
        config = FoundationConfig(
            VAL_EVERY_N_EPOCH=1,
            NUM_EPOCHS=65,
            SAVE_DIR="upsample_experiments2",
            BATCH_SIZE=3,
            VAL_BATCH_SIZE=4,
            RUN_NAME="BATCH_SIZE_3",
            MODEL="efficientnet-b4",
            IMG_SIZE=(2600,2600),
            TRAIN_CROP=(1024, 1024),
            VALIDATION_CROP=(1024, 1024),
            

            BUILDING_JACCARD_WEIGHT=0.25,
            BCE_LOSS_WEIGHT=0.75
    )
    # if num==1:
    #     config = FoundationConfig(
    #             VAL_EVERY_N_EPOCH=1,
    #             NUM_EPOCHS=120,
    #             SAVE_DIR="upsample_experiments2",
    #             BATCH_SIZE=4,
    #             VAL_BATCH_SIZE=4,
    #             RUN_NAME="hrnet_JAC",
    #             MODEL="hrnet",
    #             IMG_SIZE=(1300,1300),
    #             TRAIN_CROP=(1024, 1024),
    #             VALIDATION_CROP=(1024, 1024),

    #             BUILDING_JACCARD_WEIGHT=0.25,
    #             BCE_LOSS_WEIGHT=0.75
    #             )

   
    elif num == 3:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="HRENT_UPSAMPLEX3_JAC",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
        )
    

    elif num == 4:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="HRENT_UPSAMPLEX2",
                MODEL="hrnet",
                IMG_SIZE=(2600, 2600),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),

        )
    elif num == 5:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="HRENT_UPSAMPLEX3",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
        )

    if num==6:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="HRNET",
                MODEL="hrnet",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                )

    if num==7:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3",
                MODEL="efficientnet-b3",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                )
    
    if num==8:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3_X2",
                MODEL="efficientnet-b3",
                IMG_SIZE=(2600, 2600),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                )



    if num == 9:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb4_X2",
                    MODEL="efficientnet-b4",
                    IMG_SIZE=(2600, 2600),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),
            
                    )
    if num == 10: 
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb5",
                    MODEL="efficientnet-b5",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),
                    )

    if num == 11:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb5_X2",
                    MODEL="efficientnet-b5",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),
                    )

    if num == 12:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb4_X3",
                    MODEL="efficientnet-b4",
                    IMG_SIZE=(3900, 3900),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),
                    )

    
    if num==13:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3_JACC",
                MODEL="efficientnet-b3",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),
                PRETRAIN=False,

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
                )
    
    if num==14:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments2",
                BATCH_SIZE=4,
                VAL_BATCH_SIZE=4,
                RUN_NAME="efficientnet-b3_X2_JACC",
                MODEL="efficientnet-b3",
                IMG_SIZE=(2600, 2600),
                TRAIN_CROP=(1024, 1024),
                VALIDATION_CROP=(1024, 1024),

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
                )



    if num == 15:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb4_X2_JACC",
                    MODEL="efficientnet-b4",
                    IMG_SIZE=(2600, 2600),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),

                    BUILDING_JACCARD_WEIGHT=0.25,
                    BCE_LOSS_WEIGHT=0.75
            
                    )
    if num == 16: 
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb5_JACC",
                    MODEL="efficientnet-b5",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),

                    BUILDING_JACCARD_WEIGHT=0.25,
                    BCE_LOSS_WEIGHT=0.75
                    )

    if num == 17:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb5_X2_JACC",
                    MODEL="efficientnet-b5",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),

                    BUILDING_JACCARD_WEIGHT=0.25,
                    BCE_LOSS_WEIGHT=0.75
                    )

    if num == 18:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments2",
                    BATCH_SIZE=4,
                    VAL_BATCH_SIZE=4,
                    RUN_NAME="efficinetb4_X3_JACC",
                    MODEL="efficientnet-b4",
                    IMG_SIZE=(3900, 3900),
                    TRAIN_CROP=(1024,1024),
                    VALIDATION_CROP=(1024, 1024),

                    BUILDING_JACCARD_WEIGHT=0.25,
                    BCE_LOSS_WEIGHT=0.75
                    )

    
    
    return config
    


