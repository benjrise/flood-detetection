 
import os
import argparse
from datetime import date, datetime
from tqdm.auto import tqdm
from models.efficientnet.efficient_unet import EfficientNet_Unet, EfficientNetSeperate

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import SN8Dataset
from core.losses import ComputeLoss
import models.pytorch_zoo.unet as unet
from models.other.unet import UNet
from utils.utils import get_transforms, train_validation_file_split
from config_foundation import FoundationConfig, get_config, get_multi_config, get_multi_config2, get_multi_config3
from models.hrnet.hrnet import HighResolutionNet, get_seg_model
from models.hrnet.hr_config import get_hrnet_config
from utils.utils import get_prediction_fig, plot_buildings, check_dir_exists
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# def average_gradients(model):
#     size = float(dist.get_world_size())
#     for param in model.parameters():
#         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
#         param.grad.data /= size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                         type=int,)
    parser.add_argument("-v",
                        type=int,
                        )
    parser.add_argument("--slurm",
                        action="store_true")
    args = parser.parse_args()
    return args

def log_to_tensorboard(writer, metrics, n_iter):
    for key, value in metrics.items():
        writer.add_scalar(key, value, n_iter)

def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)



models = {
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet': UNet,
}


if __name__ == "__main__":
    
    args = parse_args()

    if args.config:
        if args.v == 1:
            config = get_multi_config(args.config)
        if args.v == 2:
            config = get_multi_config2(args.config)
        if args.v == 3:
            config = get_multi_config3(args.config)
        else:
            config = get_multi_config(args.config)
        
    else:
        config = FoundationConfig()
    
    model_name = config.MODEL
    initial_lr = 1e-4
    batch_size = config.BATCH_SIZE
    n_epochs = config.NUM_EPOCHS
    run_name = config.RUN_NAME

    now = datetime.now() 
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))
    
    img_size = config.IMG_SIZE

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = config.CUDNN_ENABLED
    run_name = config.RUN_NAME

    SEED=12
    torch.manual_seed(SEED)

    save_dir = config.SAVE_DIR
    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    save_dir = os.path.join(save_dir, f"{run_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")
    os.makedirs(save_dir, mode=0o777, exist_ok=True)
    
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")

    writer = SummaryWriter(save_dir)
    print(f"RUN: {run_name}")
    train_transforms, validation_transforms =\
         get_transforms(crop=config.TRAIN_CROP, 
                        normalize=config.NORMALIZE,
                        p_random_flips=config.P_FLIPS, 
                        center_crop=config.VALIDATION_CROP)
    
    train_files, val_files = train_validation_file_split(config.GPU,
                                ["preimg","building","roadspeed"], 
                                config.TRAIN_CSV)

    train_dataset = SN8Dataset(train_files,
                                data_to_load=["preimg","building","roadspeed"],
                                img_size=img_size,
                                transforms=train_transforms)
    # if config.BAGGING:
    #     possible_indexes = np.arrange(len(train_dataset))
    #     indexes = np.random.choice(range(len(train_dataset)), int(len(train_dataset)*config.BAGGING_RATIO))
    #     val_indexes = possible_indexes[possible_indexes != indexes]
    #     train_dataset = torch.utils.data.Subset(train_dataset, indexes)


    train_dataloader = torch.utils.data.DataLoader(train_dataset,
             num_workers=4, 
            batch_size=batch_size, shuffle=True,
            )
    val_dataset = SN8Dataset(val_files,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size,
                            transforms=validation_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                num_workers=4, batch_size=config.VAL_BATCH_SIZE, 
                                )

    if model_name == "unet":
        model = UNet(3, [1,8], bilinear=True)
    elif model_name == "hrnet":
        model_config = get_hrnet_config("models/hrnet/hr_config.yml")
        model = get_seg_model(model_config, pretrain=config.PRETRAIN)
    elif model_name == "seperate":
        model = EfficientNetSeperate("efficientnet-b4")
    
    elif model_name[:13] == "efficientnet-":
        model = EfficientNet_Unet(name=model_name, 
                                pretrained=config.PRETRAIN,
                                mode="foundation")
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3)
    
    device = torch.device(f"cuda:{config.GPU}")
    model = model.to(device)
    optimizer = config.get_optimizer(model)

    
    if config.NEW_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                patience=config.PATIENCE, factor=config.FACTOR, 
                                eps=1e-7, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)

    best_loss = np.inf
    loss_fn = ComputeLoss(
            config.BUILDING_LOSS_WEIGHT,
            config.ROAD_LOSS_WEIGHT,
            config.BCE_LOSS_WEIGHT,
            config.SOFT_DICE_LOSS_WEIGHT,
            config.ROAD_FOCAL_LOSS_WEIGHT,
            config.BUILDING_JACCARD_WEIGHT,
            config.BUILDING_FOCAL_WEIGHT,
            config.BUILDING_SOFT_DICE
    )

    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")

        ### Training ##
        model.train()
        train_metrics = {}
        with tqdm(enumerate(train_dataloader), unit="batch", total=len(train_dataloader)) as tepoch:
            for i, data in tepoch:
                optimizer.zero_grad()

                preimg, postimg, building, road, roadspeed, flood = data

                preimg = preimg.to(device).float()
                roadspeed = roadspeed.to(device).float()
                building = building.to(device).float()

                with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                    building_pred, road_pred = model(preimg)
                    loss = loss_fn(building_pred, road_pred, building, roadspeed)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                for key, value in loss_fn.stash_metrics.items():
                    if key  not in train_metrics:
                        train_metrics[key] = value
                    else:
                        train_metrics[key] += value
                
                tepoch.set_postfix(loss=loss.item())
                
                n_iter = epoch * len(train_dataloader) + i
                writer.add_scalar("training_loss_step", loss, n_iter)

        for key in train_metrics.keys():
            train_metrics[key] /= len(train_dataloader)
        
        train_metrics["lr"] = optimizer.param_groups[0]['lr']
        # print(train_metric)
        log_to_tensorboard(writer, train_metrics, epoch)
        train_metrics.clear()
        if epoch % config.VAL_EVERY_N_EPOCH == 0:
            # validation
            model.eval()
            val_metrics = {}
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_dataloader)):
                    preimg, postimg, building, road, roadspeed, flood = data
                    preimg = preimg.to(device)
                    roadspeed = roadspeed.to(device)
                    building = building.to(device)

                    with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                        building_pred, road_pred = model(preimg)
                        loss = loss_fn.forward(building_pred, road_pred,
                                            building, roadspeed)
                        for key, value in loss_fn.stash_metrics.items():
                                key = f"valid_{key}"
                                if key not in val_metrics:
                                    val_metrics[key] = value
                                else:
                                    val_metrics[key] += value
                                    
                    out_loss = val_metrics["valid_loss"]
                    print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(out_loss*1.0/(i+1))}", end="\r")

            for key in val_metrics.keys():
                val_metrics[key] /= len(val_dataloader)
            
            if config.NEW_SCHEDULER:
                scheduler.step()
                print("step")
            else:
                scheduler.step(val_metrics["valid_loss"])
                
            log_to_tensorboard(writer, val_metrics, epoch)
            save_model_checkpoint(model, checkpoint_model_path)

            epoch_val_loss = val_metrics["valid_loss"]
            
            if epoch_val_loss < best_loss:
                print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
                best_loss = epoch_val_loss
                save_model_checkpoint(model, best_model_path)

        
    
   

    if config.MULTI_MODEL:
        from foundation_eval import save_model_predictions
        save_model_predictions(config, model, save_dir)
    
    if config.FINAL_EVAL_LOOP:
        from foundation_eval import foundation_final_eval_loop
        foundation_final_eval_loop(config, model, save_dir)
        

    
    