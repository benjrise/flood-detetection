from dataclasses import dataclass
import torch

@dataclass
class FoundationConfig:
    #LR : float
    
    DEBUG : bool = True
    TRAIN_CSV : str = "areas_of_interest/sn8_data_train.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv"

    MIXED_PRECISION : bool = True
    VAL_EVERY_N_EPOCH : int = 3
    NUM_EPOCHS : int = 1
    SAVE_DIR : str = "debug"

    GPU : int = 0
    BATCH_SIZE : int = 8
    VAL_BATCH_SIZE : int = 1

    RUN_NAME : str = "HRNET_X2_FULL_BS1"
    
    MODEL : str = "efficientnet-b3"
    PRETRAIN : bool = True

    IMG_SIZE : tuple = (2600, 2600)

    SOFT_DICE_LOSS_WEIGHT : float = 0.25 
    ROAD_FOCAL_LOSS_WEIGHT :float = 0.75 

    BCE_LOSS_WEIGHT : float = 1.
    BUILDING_FOCAL_WEIGHT : float = 0.
    BUILDING_JACCARD_WEIGHT : float = 0.

    ROAD_LOSS_WEIGHT : float = 0.5
    BUILDING_LOSS_WEIGHT : float= 0.5


    NORMALIZE : float = True
    TRAIN_CROP : tuple = (650, 650)
    VALIDATION_CROP : tuple = None
    P_FLIPS : float = 0

    PATIENCE : int = 8
    FACTOR : float = 0.5

    PLOT_EVERY : int = 10


    # FINAL EVAL LOOP
    FINAL_EVAL_LOOP : bool = True
    SAVE_FIG : bool = True
    SAVE_PRED : bool = None


    # CUDNN FLAGS
    CUDNN_BENCHMARK : bool = False
    CUDNN_ENABLED : bool = False

    OPTIMIZER : str = 'adamw'
    #MOMENTUM : float = 0
    

    def get_optimizer(self, model):
        eps = 1e-7
        WEIGHT_DECAY = 0

        if self.OPTIMIZER == 'adamw':
            LR = 1e-4
            return torch.optim.AdamW(model.parameters(), 
                                    lr=LR, 
                                    eps=eps,
                                    weight_decay=WEIGHT_DECAY)
        if self.OPTIMIZER == 'adam':
            LR = 1e-4
            return torch.optim.AdamW(model.parameters(), 
                                    lr=LR, 
                                    eps=eps,
                                    weight_decay=WEIGHT_DECAY)
        if self.OPTIMIZER == 'sgd':
            LR = 1e-2
            MOMENTUM = 0.9
            return torch.optim.SGD(model.parameters(), 
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY)



def get_config1(num):
    if num == 1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
                SAVE_DIR="upsample_experiments",
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
    if num==1:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="hrnet_JAC",
                MODEL="hrnet",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),

                BUILDING_JACCARD_WEIGHT=0.25,
                BCE_LOSS_WEIGHT=0.75
                )

    elif num == 2:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
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
    elif num == 3:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
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
    

    elif num == 4:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX2",
                MODEL="hrnet",
                IMG_SIZE=(2600,2600),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),

        )
    elif num == 5:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRENT_UPSAMPLEX3",
                MODEL="hrnet",
                IMG_SIZE=(3900,3900),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
        )

    if num==6:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="HRNET",
                MODEL="hrnet",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                )

    if num==7:
        config = FoundationConfig(
                VAL_EVERY_N_EPOCH=1,
                NUM_EPOCHS=120,
                SAVE_DIR="upsample_experiments",
                BATCH_SIZE=8,
                VAL_BATCH_SIZE=1,
                RUN_NAME="efficientnet-b3",
                MODEL="efficientnet-b3",
                IMG_SIZE=(1300,1300),
                TRAIN_CROP=(512,512),
                VALIDATION_CROP=(1024, 1024),
                )


    if num == 8:
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments",
                    BATCH_SIZE=8,
                    VAL_BATCH_SIZE=1,
                    RUN_NAME="efficinetb4",
                    MODEL="efficientnet-b4",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(512,512),
                    VALIDATION_CROP=(1024, 1024),
            
                    )
    if num == 9: 
        config = FoundationConfig(
                    VAL_EVERY_N_EPOCH=1,
                    NUM_EPOCHS=120,
                    SAVE_DIR="upsample_experiments",
                    BATCH_SIZE=8,
                    VAL_BATCH_SIZE=1,
                    RUN_NAME="efficinetb5",
                    MODEL="efficientnet-b5",
                    IMG_SIZE=(1300,1300),
                    TRAIN_CROP=(512,512),
                    VALIDATION_CROP=(1024, 1024),
                    )
    return config
    


@dataclass
class FloodConfig:
    MIXED_PRECISION : bool = True

    TRAIN_CSV : str = "areas_of_interest/sn8_data_train.csv"
    VAL_CSV : str = "areas_of_interest/sn8_data_val.csv"
    VALIDATION_LOOP_CSV : str = "areas_of_interest/sn8_data_val.csv"

    SAVE_CSV : str = "flood_debug/"
    MODEL_NAME : str = "resnet34"
    LR : float = 1e-4
    BATCH_SIZE : int = 8
    N_EPOCHS : int = 30

    IMG_SIZE : tuple = (2600, 2600)

    VALID_CROP = (1300, 1300)

    GPU : int = 0

    SAVE_FIG : bool = True
    SAVE_PRED : bool = False

