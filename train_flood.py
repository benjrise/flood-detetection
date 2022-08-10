import csv
import os
import argparse
import datetime
from datetime import datetime
from flood_eval import flood_final_eval_loop

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from datasets.datasets import SN8Dataset
import models.pytorch_zoo.unet as unet
from models.other.unet import UNetSiamese
from models.other.siamunetdif import SiamUnet_diff
from models.other.siamnestedunet import SNUNet_ECAM
from models.efficientnet.efficient_unet import EfficientNet_Unet, EfficientNet_Unet_Double
from utils.utils import get_transforms
from config import FloodConfig
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--save_dir",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    
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
        writer.add_scalar(key, value,global_step=n_iter)
        
def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)
        
def save_best_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)

models = {
    'resnet34_siamese': unet.Resnet34_siamese_upsample,
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet_siamese':UNetSiamese,
    'unet_siamese_dif':SiamUnet_diff,
    'nestedunet_siamese':SNUNet_ECAM
}

if __name__ ==  "__main__":
    # args = parse_args()
    config = FloodConfig()
    train_csv = config.TRAIN_CSV
    val_csv = config.VAL_CSV
    save_dir = config.SAVE_CSV
    model_name = config.MODEL_NAME
    initial_lr = config.LR
    batch_size = config.BATCH_SIZE
    n_epochs = config.N_EPOCHS
    gpu = config.GPU

    print(model_name)
    
    now = datetime.now() 
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device(f"cuda:{gpu}")

    soft_dice_loss_weight = 0.25
    focal_loss_weight = 0.75
    num_classes=5
    class_weights = None

    road_loss_weight = 0.5
    building_loss_weight = 0.5

    img_size = config.IMG_SIZE

    SEED=12
    torch.manual_seed(SEED)
    
    assert(os.path.exists(save_dir))
    save_dir = os.path.join(save_dir, f"{model_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.chmod(save_dir, 0o777)
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")

    # init the training log
    with open(training_log_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'lr', 'train_tot_loss',
                                     'val_tot_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    writer = SummaryWriter()
    
    train_transforms, validation_transforms = get_transforms(crop=config.TRAIN_CROP,
                                                    center_crop=config.VALID_CROP)
    train_dataset = SN8Dataset(train_csv,
                            data_to_load=["preimg","postimg","flood"],
                            img_size=img_size,
                            transforms=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batch_size)
    val_dataset = SN8Dataset(val_csv,
                            data_to_load=["preimg","postimg","flood"],
                            img_size=img_size,
                            transforms=validation_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

    # model = models["resnet34"](num_classes=5, num_channels=6)
    if config.SIAMESE:
        if model_name == "unet_siamese":
            model = UNetSiamese(3, num_classes, bilinear=True)
        elif model_name[:13] == "efficientnet-":
            model = EfficientNet_Unet_Double(name=model_name,
                                    pretrained=config.PRETRAIN, 
                                    in_channels=3, 
                                    num_classes=5,
                                    )
    
    else:
        if model_name[:13] == "efficientnet-":
            model = EfficientNet_Unet(name=model_name,
                                    pretrained=config.PRETRAIN, 
                                    in_channels=6, 
                                    num_classes=5,
                                    mode="flood")
        else:
            model = models[model_name](num_classes=num_classes, 
                                        num_channels=6,
                                        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)

    if class_weights is None:
        celoss = nn.CrossEntropyLoss()
    else:
        celoss = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = np.inf
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")

        ### Training ##
        model.train()
        train_loss_val = 0
        train_focal_loss = 0
        train_soft_dice_loss = 0
        train_bce_loss = 0
        train_road_loss = 0
        train_building_loss = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            preimg, postimg, building, road, roadspeed, flood = data

            preimg = preimg.to(device).float()
            postimg = postimg.to(device).float()
            if not config.SIAMESE:
                combinedimg = torch.cat((preimg, postimg), dim=1)

            flood = flood.numpy()
            flood_shape = flood.shape
            flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
            flood = np.argmax(flood, axis = 1) # this is needed for cross-entropy loss. 

            flood = torch.tensor(flood).to(device)
            
            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                if not config.SIAMESE:
                    flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
                else:
                    flood_pred = model(preimg, postimg) # this is for siamese resnet34 with stacked preimg+postimg input

                #y_pred = F.sigmoid(flood_pred)
                #focal_l = focal(y_pred, flood)
                #dice_soft_l = soft_dice_loss(y_pred, flood)
                #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
                loss = celoss(flood_pred, flood.long())

                train_loss_val+=loss
            #train_focal_loss += focal_l
            #train_soft_dice_loss += dice_soft_l
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"    {str(np.round(i/len(train_dataloader)*100,2))}%: TRAIN LOSS: {(train_loss_val*1.0/(i+1)).item()}", end="\r")
            n_iter = epoch * len(train_dataloader) + i
            writer.add_scalar("training_loss_step", train_loss_val, n_iter)
            
        
        print()
        train_tot_loss = (train_loss_val*1.0/len(train_dataloader)).item()
        #train_tot_focal = (train_focal_loss*1.0/len(train_dataloader)).item()
        #train_tot_dice = (train_soft_dice_loss*1.0/len(train_dataloader)).item()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_metrics = {"lr": current_lr, "train_tot_loss": train_tot_loss}
        
        log_to_tensorboard(writer, train_metrics, epoch)

        # validation
        model.eval()
        val_loss_val = 0
        val_focal_loss = 0
        val_soft_dice_loss = 0
        val_bce_loss = 0
        val_road_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                preimg, postimg, building, road, roadspeed, flood = data

                preimg = preimg.to(device).float()
                postimg = postimg.to(device).float()
                if not config.SIAMESE:
                    combinedimg = torch.cat((preimg, postimg), dim=1)

                flood = flood.numpy()
                flood_shape = flood.shape
                flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
                flood = np.argmax(flood, axis = 1) # for crossentropy
                
                #temp = np.zeros(shape=(flood_shape[0],6,flood_shape[2],flood_shape[3]))
                #temp[:,:4] = flood
                #temp[:,4] = np.max(flood[:,:2], axis=1)
                #temp[:,5] = np.max(flood[:,2:], axis=1)
                #flood = temp

                flood = torch.tensor(flood).to(device)

                with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                    
                    if not config.SIAMESE:
                        flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
                    else:
                        flood_pred = model(preimg, postimg) # this is for siamese resnet34 with stacked preimg+postimg input


                    #y_pred = F.sigmoid(flood_pred)
                    #focal_l = focal(y_pred, flood)
                    #dice_soft_l = soft_dice_loss(y_pred, flood)
                    #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)

                    loss = celoss(flood_pred, flood.long())

                    #val_focal_loss += focal_l
                    #val_soft_dice_loss += dice_soft_l
                    val_loss_val += loss

                print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(val_loss_val*1.0/(i+1)).item()}", end="\r")

        print()        
        val_tot_loss = (val_loss_val*1.0/len(val_dataloader)).item()
        #val_tot_focal = (val_focal_loss*1.0/len(val_dataloader)).item()
        #val_tot_dice = (val_soft_dice_loss*1.0/len(val_dataloader)).item()
        val_metrics = {"val_tot_loss":val_tot_loss}

        write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)
        log_to_tensorboard(writer, val_metrics, epoch)
   

        save_model_checkpoint(model, checkpoint_model_path)

        epoch_val_loss = val_metrics["val_tot_loss"]
        if epoch_val_loss < best_loss:
            print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
            best_loss = epoch_val_loss
            save_best_model(model, best_model_path)

    flood_final_eval_loop(config, model, val_dataset, save_dir)