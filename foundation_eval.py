import os
import argparse
from config_foundation import FoundationConfig, get_multi_config
import csv
from dataclasses import asdict
from models.efficientnet.efficient_unet import EfficientNet_Unet
import time

from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn

import models.pytorch_zoo.unet as unet
from models.other.unet import UNet
from datasets.datasets import SN8Dataset, SN8TestDataset
from utils.utils import get_transforms, train_validation_file_split, write_geotiff, check_dir_exists
from models.hrnet.hrnet import HighResolutionNet, get_seg_model
from models.hrnet.hr_config import get_hrnet_config
from utils.validation_utils import validate_patches 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=int)
    
    args = parser.parse_args()
    return args


def make_prediction_png_roads_buildings(image, gts, predictions, save_figure_filename):
    bldg_gt = gts[0][0]
    road_gt = gts[1]
    bldg_pred = predictions[0][0]
    road_pred = predictions[1]
    
    # seperate the binary road preds and speed preds
    binary_road_pred = road_pred[-1]
    binary_road_gt = road_gt[-1]
    
    speed_pred = np.argmax(road_pred[:-1], axis=0)
    speed_gt = np.argmax(road_gt[:-1], axis=0) 
    
    roadspeed_shape = road_pred.shape
    tempspeed = np.zeros(shape=(roadspeed_shape[0]+1,roadspeed_shape[1],roadspeed_shape[2]))
    tempspeed[1:] = road_pred
    road_pred = tempspeed
    road_pred = np.argmax(road_pred, axis=0)
    
    combined_pred = np.zeros(shape=bldg_pred.shape, dtype=np.uint8)
    combined_pred = np.where(bldg_pred==1, 1, combined_pred)
    combined_pred = np.where(binary_road_pred==1, 2, combined_pred)
    
    combined_gt = np.zeros(shape=bldg_gt.shape, dtype=np.uint8)
    combined_gt = np.where(bldg_gt==1, 1, combined_gt)
    combined_gt = np.where(binary_road_gt==1, 2, combined_gt)
    
    
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    
    grid = [[raw_im, combined_gt, combined_pred, speed_gt, speed_pred]]
    
    nrows = len(grid)
    ncols = len(grid[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*4))
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[col]
            ax.axis('off')
            if row==0 and col==0:
                ax.imshow(grid[row][col])
            elif row==0 and col in [3,4]:
                combined_mask_cmap = colors.ListedColormap(['black', 'green', 'blue', 'red',
                                                            'purple', 'orange', 'yellow', 'brown',
                                                            'pink'])
                ax.imshow(grid[row][col], cmap=combined_mask_cmap, interpolation='nearest', origin='upper',
                                  norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8], combined_mask_cmap.N))
            if row==0 and col in [1,2]:
                combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue'])
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3], combined_mask_cmap.N))
            # if row==1 and col == 1:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_gt==0, np.nan, combined_gt)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            # if row==1 and col == 2:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_pred==0, np.nan, combined_pred)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename)
    plt.close(fig)
    plt.close('all')

models = {
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'efficientnet-b'
    'unet':UNet,
    'hrnet' : HighResolutionNet
}

def save_model_predictions(config : FoundationConfig, 
                        model,  
                        save_dir):
    """
    Saves model predictions as numpy array, which is to be read in by ensemble function
    """
    model_path = os.path.join(save_dir, "best_model.pth")
    in_csv = config.TEST_CSV
    predictions_dir = config.PATH_SAVE_PREDS
    model_folder = config.RUN_NAME

    check_dir_exists(predictions_dir)
    prediction_path = os.path.join(predictions_dir, model_folder)
    check_dir_exists(prediction_path)
    if config.MULTI_MODEL_EVAL:
        ground_truth_path = config.PATH_GT_SAVE
        check_dir_exists(ground_truth_path)       

    gpu = config.GPU
    img_size = config.IMG_SIZE
    train_transforms, valid_transforms = get_transforms(crop=None)
    
    print(config.TESTING)
    if not config.TESTING:
        _, val_files = train_validation_file_split(gpu, 
                                    data_to_load=["preimg","building","roadspeed"],
                                    csv_path="areas_of_interest/sn8_full_data.csv")
        
        val_dataset = SN8Dataset(val_files, 
                                data_to_load=["preimg","building","roadspeed"],  
                                img_size=config.IMG_SIZE,
                                transforms=valid_transforms)
    else:
        
        val_dataset = SN8TestDataset(in_csv, 
                                data_to_load=["preimg"],  
                                img_size=config.IMG_SIZE,
                                transforms=valid_transforms)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{gpu}'))
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(zip(val_dataloader)):
            
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            save_path = os.path.join(prediction_path, os.path.basename(current_image_filename.replace(".tif", ".npy")))
            # if os.path.exists(save_path):
            #     print("skipping")
            #     continue
            preimg, postimg, building, road, roadspeed, flood = data[0]

            preimg.to(device)
            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                building_pred, roadspeed_pred = validate_patches(preimg, model, device,
                                                        patch_size=(1024, 1024), bs=2, 
                                                        overlap_height_ratio=0.2,
                                                        overlap_width_ratio=0.2)
            
            building_prediction = building_pred.cpu().numpy()[0] # (C,H,W)
            road_prediction = roadspeed_pred.cpu().numpy()[0] # (C,H,W)
           

            save_path = os.path.join(prediction_path, os.path.basename(current_image_filename.replace(".tif", ".npy")))
            with open(save_path, "wb") as f:
                np.savez(f, building_prediction=building_prediction, 
                        road_prediction=road_prediction, tif_path=current_image_filename)

            
            if config.MULTI_MODEL_EVAL:
                gt_building = building.cpu().numpy()[0][0] # index so building gt is (H, W)
                gt_roadspeed = roadspeed.cpu().numpy()[0] # index so we have (C,H,W)
                with open(os.path.join(ground_truth_path, os.path.basename(current_image_filename.replace(".tif", ".npy"))), "wb") as f:
                    np.savez(f, gt_building=gt_building, gt_roadspeed=gt_roadspeed)
            
    
    if config.SAVE_TRAINING_PREDS:
        out_dir = config.PATH_SAVE_TRAIN_PREDS
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            os.chmod(out_dir, 0o777)

        model_folder = config.RUN_NAME
        
        prediction_path = os.path.join(out_dir, model_folder)
        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)
            os.chmod(prediction_path, 0o777)    

        train_dataset = SN8Dataset(
            csv_filename=config.TRAIN_CSV,
            data_to_load=["preimg","building","roadspeed"],  
            transforms=train_transforms,
            img_size=img_size
        )
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(zip(train_dataloader)):
                current_image_filename = train_dataset.get_image_filename(i)
                print("evaluating: ", i, os.path.basename(current_image_filename))
                preimg, postimg, building, road, roadspeed, flood = data[0]
                preimg.to(device)
                with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                    building_pred, roadspeed_pred = validate_patches(preimg, 
                                                    model, device,
                                                    patch_size=(1024, 1024), bs=2, 
                                                    overlap_height_ratio=0.2,
                                                    overlap_width_ratio=0.2)
                                                

                building_prediction = building_pred.cpu().numpy() # (1,H,W) 
                road_prediction = roadspeed_pred.cpu().numpy() #  (1,H,W)
                save_path = os.path.join(prediction_path, os.path.basename(current_image_filename.replace(".tif", ".npy")))
                with open(save_path, "wb") as f:
                    np.savez(f, building_prediction=building_prediction, 
                        road_prediction=road_prediction, 
                        tif_path=current_image_filename)


def foundation_final_eval_loop(config : FoundationConfig, 
                            model, 
                            save_dir):

    model_path = os.path.join(save_dir, "best_model.pth")
    print(model_path)
    in_csv = config.VALIDATION_LOOP_CSV
    
    if config.SAVE_FIG:
        save_fig_dir = os.path.join(save_dir, "final_validation_figs")
        if not os.path.exists(save_fig_dir):
            os.mkdir(save_fig_dir)
            os.chmod(save_fig_dir, 0o777)
    else:
        save_fig_dir = None
    
    if config.SAVE_PRED:
        save_preds_dir = os.path.join(save_dir, "final_validation_preds")
        if not os.path.exists(save_preds_dir):
            os.mkdir(save_preds_dir)
            os.chmod(save_preds_dir, 0o777)
    else:
        save_preds_dir = None
    
    gpu = config.GPU
    img_size = config.IMG_SIZE
    _, valid_transforms = get_transforms()
    _, val_files = train_validation_file_split(gpu, 
                                    data_to_load=["preimg","building","roadspeed"],
                                    csv_path="areas_of_interest/sn8_full_data.csv")
        
    val_dataset = SN8Dataset(val_files, 
                            data_to_load=["preimg","building","roadspeed"],  
                            img_size=config.IMG_SIZE,
                            transforms=valid_transforms)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    predictions = np.zeros((2, 8, img_size[0], img_size[1]))
    gts = np.zeros((2, 8, img_size[0], img_size[1]))
    running_tp = [0,0] 
    running_fp = [0,0]
    running_fn = [0,0]
    running_union = [0,0]

    filenames = [[], []]
    precisions = [[], []]
    recalls = [[], []]
    f1s = [[], []]
    ious = [[], []]
    positives = [[], []]

    model.eval()
    val_loss_val = 0
    with torch.no_grad():

        for i, data in enumerate(zip(val_dataloader)):
            
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            #### TODO REMOVE
            
            preimg, postimg, building, road, roadspeed, flood = data[0]
            # gt_preimg, gt_postimg, gt_building, gt_road, gt_roadspeed, gt_flood = data[1]
            #preimg = preimg.to(device).float()
            # roadspeed = roadspeed.to(device).float()
            # building = building.to(device).float()
            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                building_pred, roadspeed_pred = validate_patches(preimg, model, device,
                                                        patch_size=(1024, 1024), bs=2, 
                                                        overlap_height_ratio=0.2,
                                                        overlap_width_ratio=0.2)
                
                roadspeed_pred = torch.sigmoid(roadspeed_pred)
                building_pred = torch.sigmoid(building_pred)
                
            preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
            
            gt_building = building.cpu().numpy()[0][0] # index so building gt is (H, W)
            gt_roadspeed = roadspeed.cpu().numpy()[0] # index so we have (C,H,W)
            
            
            building_prediction = building_pred.cpu().numpy()[0][0] # index so shape is (H,W) for buildings
            road_prediction = roadspeed_pred.cpu().numpy()[0] # index so we have (C,H,W)

            building_prediction = np.rint(building_prediction).astype(int)
            roadspeed_prediction = np.rint(road_prediction).astype(int)
            

            gts[0,0] = gt_building
            gts[1,:] = gt_roadspeed
            predictions[0,0] = building_prediction
            predictions[1,:] = roadspeed_prediction
            
      

            ### save prediction
            if save_preds_dir is not None:
                road_pred_arr = (road_prediction * 255).astype(np.uint8) # to be compatible with the SN5 eval and road speed prediction, need to mult by 255
                ds = gdal.Open(current_image_filename)
                geotran = ds.GetGeoTransform()
                xmin, xres, rowrot, ymax, colrot, yres = geotran
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(ds.GetProjectionRef())
                ds = None
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_roadspeedpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, road_pred_arr)
                            
                building_pred_arr = np.array([(building_prediction * 255).astype(np.uint8)])
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_buildingpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, building_pred_arr)
            
            for j in range(len(gts)): # iterate through the building and road gt
                prediction = predictions[j]
                gt = gts[j]
                if j == 1: # it's roadspeed, so get binary pred and gt for metrics
                    prediction = prediction[-1]
                    gt = gt[-1]
                
                tp = np.rint(prediction * gt)
                fp = np.rint(prediction - tp)
                fn = np.rint(gt - tp)
                union = np.rint(np.sum(prediction + gt - tp))

                iou = np.sum(tp) / np.sum((prediction + gt - tp + 0.00001))
                tp = np.sum(tp).astype(int)
                fp = np.sum(fp).astype(int)
                fn = np.sum(fn).astype(int)

                running_tp[j]+=tp
                running_fp[j]+=fp
                running_fn[j]+=fn
                running_union[j]+=union

                #acc = np.sum(np.where(prediction == gt, 1, 0)) / (gt.shape[0] * gt.shape[1])
                precision = tp / (tp + fp + 0.00001)
                recall = tp / (tp + fn + 0.00001)
                f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
                precisions[j].append(precision)
                recalls[j].append(recall)
                f1s[j].append(f1)
                ious[j].append(iou)
            
                current_image_filename = val_dataset.files[i]["preimg"]
                filenames[j].append(current_image_filename)
                if np.sum(gt) < 1:
                    positives[j].append("n")
                else:
                    positives[j].append("y") 

            if save_fig_dir is not None:
                #if save_preds_dir is not None: # for some reason, seg fault when doing both of these. maybe file saving or something is interfering. so sleep for a little
                #    time.sleep(2) 
                save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                make_prediction_png_roads_buildings(preimg, gts, predictions, save_figure_filename)

    print()
    data = ["building", "road"]
    out =  {"building" : {}, "road" : {}}
    for i in range(len(running_tp)):
        print(f"final metrics for: {data[i]}")
        precision = running_tp[i] / (running_tp[i] + running_fp[i] + 0.00001)
        recall = running_tp[i] / (running_tp[i] + running_fn[i] + 0.00001)
        f1 = 2 * (precision * recall)  / (precision + recall + 0.00001)
        iou = running_tp[i]  / (running_union[i] + 0.00001)
        out[data[i]]["precision"] = precision
        out[data[i]]["recall"] = recall
        out[data[i]]["f1"] = f1
        out[data[i]]["iou"] = iou

        print("final running evaluation score: ")
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("iou: ", iou)
        print()
    
    final_out_csv = os.path.join(save_dir, "final_out.csv")
    fields = ["data_type", "precision", "recall", "f1", "iou"]
    with open(final_out_csv, "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k in out:
            w.writerow({field: out[k].get(field) or k for field in fields})

    config_path = os.path.join(save_dir, "config.csv")
    out_config = asdict(config)
    w = csv.writer(open(config_path, "w"))
    for key, val in out_config.items():
        w.writerow([key, val])





if __name__ == "__main__":
    args = parse_args() 
    if args.config:
        config = get_multi_config(args.config)
    model = EfficientNet_Unet(name=config.MODEL, pretrained=False)
    config.TEST_CSV = "areas_of_interest/test_data.csv"
    config.PATH_SAVE_PREDS = "foundation_pres_hold/test_model_preds"
    config.SAVE_TRAINING_PREDS = False
    config.MULTI_MODEL_EVAL = False
    config.TESTING = True
    if args.config == 1:
        save_dir = "foundation_pres_hold/foundation_save/efficientnet-b3_JACC_lr1.00e-04_bs4_10-08-2022-12-05"
        model = EfficientNet_Unet(name="efficientnet-b3", pretrained=False)
        save_model_predictions(config, model, save_dir)
        
    if args.config == 2:
        save_dir = "foundation_pres_hold/foundation_save/efficientnet-b3_X2_lr1.00e-04_bs4_10-08-2022-12-05"
        model = EfficientNet_Unet(name="efficientnet-b3", pretrained=False)
        save_model_predictions(config, model, save_dir)

    if args.config == 3:
        save_dir = "foundation_pres_hold/foundation_save/efficientnet-b4_X2_JACC_lr1.00e-04_bs3_10-08-2022-12-05"
        model = EfficientNet_Unet(name="efficientnet-b4", pretrained=False)
        save_model_predictions(config, model, save_dir)

    if args.config == 4:
        model = EfficientNet_Unet(name="efficientnet-b4", pretrained=False)
        save_dir = "foundation_pres_hold/foundation_save/efficientnet-b4_X2_lr1.00e-04_bs3_10-08-2022-12-05"
        save_model_predictions(config, model, save_dir)


    