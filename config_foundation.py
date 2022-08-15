from dataclasses import dataclass
import torch
import os
from utils.utils import check_dir_exists

@dataclass
class FoundationConfig:
    RUN_NAME : str = "efficient-b3_bagging"
    MODEL : str = "efficientnet-b3"
    PRETRAIN : bool = True

    DEBUG : bool = True
    TRAIN_CSV : str = "areas_of_interest/sn8_data_train.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv"

    VAL_EVERY_N_EPOCH : int = 3
    NUM_EPOCHS : int = 120

    GPU : int = 0
    BATCH_SIZE : int = 2
    VAL_BATCH_SIZE : int = 2
    MIXED_PRECISION : bool = True

    # OUT DIRS
    SAVE_DIR : str = "upsample_experiments2"
    MULTI_SAVE_PATH :   str = "foundation_predictions"
    MULTI_TRAIN_SAVE_PATH : str = "foundation_training_predictions"
    GT_SAVE_DIR : str = "ground_truth"

    IMG_SIZE : tuple = (2600, 2600)


    # LOSS TERMS
    SOFT_DICE_LOSS_WEIGHT : float = 0.25 
    ROAD_FOCAL_LOSS_WEIGHT :float = 0.75 

    BCE_LOSS_WEIGHT : float = 0.75
    BUILDING_FOCAL_WEIGHT : float = 0.
    BUILDING_JACCARD_WEIGHT : float = 0.25

    ROAD_LOSS_WEIGHT : float = 0.5
    BUILDING_LOSS_WEIGHT : float= 0.5

    # AUGMENTATION 
    NORMALIZE : float = True
    TRAIN_CROP : tuple = (650, 650)
    VALIDATION_CROP : tuple = (1024,1024)
    P_FLIPS : float = 0

    # LR SCHEDULER
    PATIENCE : int = 8
    FACTOR : float = 0.5

    # FINAL EVAL LOOP
    FINAL_EVAL_LOOP : bool = True
    SAVE_FIG : bool = None
    SAVE_PRED : bool = None

    # CUDNN FLAGS
    CUDNN_BENCHMARK : bool = False
    CUDNN_ENABLED : bool = False

    # OPTIMIZER PARAMS
    OPTIMIZER : str = 'adamw'
    LR : float = 1e-4
    MOMENTUM : float = 0.0

    # MULTI_MODEL
    TEST_CSV : str = ""
    MULTI_MODEL : bool = False
    MULTI_MODEL_EVAL : bool = False
    SAVE_TRAINING_PREDS : bool = False

    # BAGGING
    BAGGING : bool = False
    BAGGING_RATIO : float = 0.8
    
    def __post_init__(self):
        if self.OPTIMIZER == 'sgd':
            self.LR = 1e-2
            self.MOMENTUM = 0.9
        else:
            self.LR = 1e-4
        
        if not self.TEST_CSV:
            self.TEST_CSV = self.VALIDATION_LOOP_CSV

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
        self.MULTI_SAVE_PATH = os.path.join(root_dir, self.MULTI_SAVE_PATH)
        self.MULTI_TRAIN_SAVE_PATH = os.path.join(root_dir, self.MULTI_TRAIN_SAVE_PATH)
        self.GT_SAVE_DIR = os.path.join(root_dir, self.GT_SAVE_DIR)
        




# def get_multi_config(num):
#     VAL_EVERY_N_EPOCH=1
#     NUM_EPOCHS=120
#     IMG_SIZE=(2600, 2600)
#     TRAIN_CROP=(1024, 1024)
#     VALIDATION_CROP=(1024, 1024)
#     FINAL_EVAL_LOOP=False
    
#     root_dir = "foundation_preds"
#     SAVE_DIR="foundation_save"
#     MULTI_SAVE_PATH = "foundation_validation_preds"
#     MULTI_TRAIN_SAVE_PATH = "foundation_train_preds"
#     MULTI_MODEL = True
#     MULTI_MODEL_EVAL = True
    
#     TEST_CSV = "areas_of_interest/sn8_data_val.csv"
#     if num==1:
#         config = FoundationConfig(
#                 VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
#                 NUM_EPOCHS=NUM_EPOCHS,
#                 SAVE_DIR=SAVE_DIR,
#                 IMG_SIZE=IMG_SIZE,
#                 TRAIN_CROP=TRAIN_CROP,
#                 VALIDATION_CROP=VALIDATION_CROP,
#                 FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
#                 MULTI_SAVE_PATH=MULTI_SAVE_PATH,
#                 MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
#                 MULTI_MODEL=MULTI_MODEL,
#                 MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
#                 TEST_CSV=TEST_CSV,
                
                
#                 BATCH_SIZE=4,
#                 VAL_BATCH_SIZE=4,
#                 RUN_NAME="efficientnet-b3_X2",
#                 MODEL="efficientnet-b3",
#                 GPU=num-1
#                 )

#     if num==2:
#         config = FoundationConfig(
#                 VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
#                 NUM_EPOCHS=NUM_EPOCHS,
#                 SAVE_DIR=SAVE_DIR,
#                 IMG_SIZE=IMG_SIZE,
#                 TRAIN_CROP=TRAIN_CROP,
#                 VALIDATION_CROP=VALIDATION_CROP,
#                 FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
#                 MULTI_SAVE_PATH=MULTI_SAVE_PATH,
#                 MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
#                 MULTI_MODEL=MULTI_MODEL,
#                 MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
#                 TEST_CSV=TEST_CSV,


#                 BATCH_SIZE=4,
#                 VAL_BATCH_SIZE=4,
#                 RUN_NAME="efficientnet-b3_JACC",
#                 MODEL="efficientnet-b3",
#                 BUILDING_JACCARD_WEIGHT=0.25,
#                 BCE_LOSS_WEIGHT=0.75,
#                 GPU=num-1,
#                 )
    
#     if num==3:
#         config = FoundationConfig(
#                 VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
#                 NUM_EPOCHS=NUM_EPOCHS,
#                 SAVE_DIR=SAVE_DIR,
#                 IMG_SIZE=IMG_SIZE,
#                 TRAIN_CROP=TRAIN_CROP,
#                 VALIDATION_CROP=VALIDATION_CROP,
#                 FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
#                 MULTI_SAVE_PATH=MULTI_SAVE_PATH,
#                 MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
#                 MULTI_MODEL=MULTI_MODEL,
#                 MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
#                 TEST_CSV=TEST_CSV,


#                 BATCH_SIZE=3,
#                 VAL_BATCH_SIZE=4,
#                 RUN_NAME="efficientnet-b4_X2",
#                 MODEL="efficientnet-b4",
#                 GPU=num-1,
#                 )

#     if num==4:
#         config = FoundationConfig(
#                 VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
#                 NUM_EPOCHS=NUM_EPOCHS,
#                 SAVE_DIR=SAVE_DIR,
#                 IMG_SIZE=IMG_SIZE,
#                 TRAIN_CROP=TRAIN_CROP,
#                 VALIDATION_CROP=VALIDATION_CROP,
#                 FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
#                 MULTI_SAVE_PATH=MULTI_SAVE_PATH,
#                 MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
#                 MULTI_MODEL=MULTI_MODEL,
#                 MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
#                 TEST_CSV=TEST_CSV,
#                 SAVE_TRAINING_PREDS=True,


#                 BATCH_SIZE=3,
#                 VAL_BATCH_SIZE=4,
#                 RUN_NAME="efficientnet-b4_X2_JACC",
#                 MODEL="efficientnet-b4",
#                 BUILDING_JACCARD_WEIGHT=0.25,
#                 BCE_LOSS_WEIGHT=0.75,
#                 GPU=num-1,
#                 )
    
#     config.add_root_dir(root_dir)
#     return config

def get_multi_config(num):
    VAL_EVERY_N_EPOCH=1
    NUM_EPOCHS=65
    IMG_SIZE=(2600, 2600)
    TRAIN_CROP=(1024, 1024)
    VALIDATION_CROP=(1024, 1024)
    FINAL_EVAL_LOOP=False
    
    root_dir = "foundation_preds80"
    SAVE_DIR="foundation_save"
    MULTI_SAVE_PATH = "foundation_validation_preds"
    MULTI_TRAIN_SAVE_PATH = "foundation_train_preds"
    GT_SAVE_DIR = "ground_truth"
    MULTI_MODEL = True
    MULTI_MODEL_EVAL = True
    
    BAGGING = True
    BAGGING_RATIO = 0.8
    
    TEST_CSV = "areas_of_interest/sn8_data_val.csv"
    if num==1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                
                GT_SAVE_DIR=GT_SAVE_DIR,
                
                
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
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                GT_SAVE_DIR=GT_SAVE_DIR,


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
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                GT_SAVE_DIR=GT_SAVE_DIR,


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
                VAL_EVERY_N_EPOCH=VAL_EVERY_N_EPOCH,
                NUM_EPOCHS=NUM_EPOCHS,
                SAVE_DIR=SAVE_DIR,
                IMG_SIZE=IMG_SIZE,
                TRAIN_CROP=TRAIN_CROP,
                VALIDATION_CROP=VALIDATION_CROP,
                FINAL_EVAL_LOOP=FINAL_EVAL_LOOP,
                MULTI_SAVE_PATH=MULTI_SAVE_PATH,
                MULTI_TRAIN_SAVE_PATH=MULTI_TRAIN_SAVE_PATH,
                MULTI_MODEL=MULTI_MODEL,
                MULTI_MODEL_EVAL=MULTI_MODEL_EVAL,
                TEST_CSV=TEST_CSV,
                SAVE_TRAINING_PREDS=True,
                GT_SAVE_DIR=GT_SAVE_DIR,


                BATCH_SIZE=3,
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
    


