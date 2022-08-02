import csv
import os
import argparse
from datetime import date, datetime
from tqdm.auto import tqdm
from models.efficientnet.efficient_unet import EfficientNet_Unet

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import SN8Dataset
from core.losses import ComputeLoss, focal, jaccard_loss, soft_dice_loss
import models.pytorch_zoo.unet as unet
from models.other.unet import UNet
from utils.utils import get_transforms
from config import FoundationConfig, get_config
from models.hrnet.hrnet import HighResolutionNet, get_seg_model
from models.hrnet.hr_config import get_hrnet_config
from utils.utils import get_prediction_fig, plot_buildings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                         type=int,)
    args = parser.parse_args()
    return args
    
def write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv):
    epoch_dict = {"epoch":epoch}
    merged_metrics = {**epoch_dict, **train_metrics, **val_metrics}
    with open(training_log_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(merged_metrics)

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
    
    # TODO ADD DEBUG FLAG TO SAVE TO RANDOM DIRECTORY AND ONLY RUN A FEW SAMPLES
    args = parse_args()
    if args.config:
        config = get_config(args.config)
    else:
        config = FoundationConfig()
    model_name = config.MODEL
    initial_lr = 1e-4
    batch_size = config.BATCH_SIZE
    n_epochs = config.NUM_EPOCHS
    gpu = config.GPU
    run_name = config.RUN_NAME

    now = datetime.now() 
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))
    
    img_size = config.IMG_SIZE

    soft_dice_loss_weight = config.SOFT_DICE_LOSS_WEIGHT # road loss
    focal_loss_weight = config.ROAD_FOCAL_LOSS_WEIGHT # road loss
    
    bce_weight = config.BCE_LOSS_WEIGHT
    jaccard_weight = config.BUILDING_JACCARD_WEIGHT
    road_loss_weight = config.ROAD_LOSS_WEIGHT
    building_loss_weight = config.BUILDING_LOSS_WEIGHT
    b_focal_weight = config.BUILDING_FOCAL_WEIGHT

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = config.CUDNN_ENABLED
    run_name = config.RUN_NAME

    # In case of multi-gpu can select differnt gpus by setting gpu = 0,1,2,3
    device = torch.device(f'cuda:{gpu}') 
    SEED=12
    torch.manual_seed(SEED)
    save_dir = config.SAVE_DIR
    assert(os.path.exists(save_dir))
    save_dir = os.path.join(save_dir, f"{run_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.chmod(save_dir, 0o777)
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")


    # init the training log
    # with open(training_log_csv, 'w', newline='') as csvfile:
    #     fieldnames = ['epoch', 'lr', 'train_tot_loss', 'train_bce', 'train_dice', 'train_focal', 'train_road_loss',
    #                                  'val_tot_loss', 'val_bce', 'val_dice', 'val_focal', 'val_road_loss']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    
    writer = SummaryWriter(save_dir)
    print(f"RUN: {run_name}")
    train_transforms, validation_transforms =\
         get_transforms(crop=config.TRAIN_CROP, 
                        normalize=config.NORMALIZE,
                        p_random_flips=config.P_FLIPS, 
                        center_crop=config.VALIDATION_CROP)
    
    train_dataset = SN8Dataset(config.TRAIN_CSV,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size,
                            transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
             shuffle=True, num_workers=4, 
             batch_size=batch_size,
             )
    val_dataset = SN8Dataset(config.VAL_CSV,
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
    elif model_name[:13] == "efficientnet-":
        model = EfficientNet_Unet(name=model_name, pretrained=config.PRETRAIN)
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3)
    
    optimizer = config.get_optimizer(model)

    model.to(device)
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
            config.BUILDING_FOCAL_WEIGHT
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

                out_loss = train_metrics["loss"]
                tepoch.set_postfix(loss=out_loss*1.0/(i+1))
                
                n_iter = epoch * len(train_dataloader) + i
                writer.add_scalar("training_loss_step", loss, n_iter)

                # if i == 0 and epoch % config.PLOT_EVERY == 0:
                #         for idx, (image, pred_buildings, pred_roads, gt_buildings, gt_roads)in enumerate(zip(preimg, building_pred, y_pred, building, roadspeed)):
                #             image = image.cpu()
                #             pred_buildings = torch.sigmoid(pred_buildings).cpu().detach()
                #             pred_roads = pred_roads.cpu().detach()
                #             gt_buildings = gt_buildings.cpu().detach()
                #             gt_roads = gt_roads.cpu().detach()
                #             predictions = [pred_buildings, pred_roads]
                #             gt = [gt_buildings, gt_roads]
                #             # fig = get_prediction_fig(image, gt,  predictions)
                #             fig = plot_buildings(gt_buildings, pred_buildings)
                #             writer.add_figure(f"EPOCH {epoch} - TRAINING-ITERATION {idx}", fig, global_step=epoch)

        for key in train_metrics.keys():
            train_metrics[key] /= len(train_dataloader)
        
        train_metrics["lr"] = optimizer.param_groups[0]['lr']
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
                                if key not in val_metrics:
                                    val_metrics[key] = value
                                else:
                                    val_metrics[key] += value
                    
                    # if i == 0 and epoch % config.PLOT_EVERY == 0:
                    #     for idx, (image, pred_buildings, pred_roads, gt_buildings, gt_roads)in enumerate(zip(preimg, building_pred, road_pred, building, roadspeed)):
                    #         image = image.cpu().numpy()
                    #         pred_buildings = pred_buildings.cpu().numpy()
                    #         pred_roads = pred_roads.cpu().numpy()
                    #         gt_buildings = gt_buildings.cpu().numpy()
                    #         gt_roads = gt_roads.cpu().numpy()
                    #         predictions = [pred_buildings, pred_roads]
                    #         gt = [gt_buildings, gt_roads]
                    #         fig = get_prediction_fig(image, gt,  predictions)
                    #         writer.add_figure(f"EPOCH {epoch} - ITERATION {idx}", fig, global_step=epoch)
                    
                    out_loss = val_metrics["loss"]
                    print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(out_loss*1.0/(i+1))}", end="\r")


            for key in val_metrics.keys():
                val_metrics[key] /= len(val_dataloader)

            # write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)
            log_to_tensorboard(writer, val_metrics, epoch)
            scheduler.step(val_metrics["loss"])

            save_model_checkpoint(model, checkpoint_model_path)

            epoch_val_loss = val_metrics["loss"]
            if epoch_val_loss < best_loss:
                print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
                best_loss = epoch_val_loss
                save_model_checkpoint(model, best_model_path)
    
    if config.FINAL_EVAL_LOOP:
        from foundation_eval import foundation_final_eval_loop
        foundation_final_eval_loop(config, model, val_dataset, save_dir)