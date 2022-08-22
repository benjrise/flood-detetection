from typing import Tuple
from cv2 import transform
from osgeo import gdal
from albumentations import (Compose, 
                            Normalize, 
                            RandomCrop,
                            HorizontalFlip,
                            VerticalFlip,
                            CenterCrop,
                            OneOf)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import csv
import copy

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        #outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None

def get_transforms(crop=(512, 512), normalize=True, p_random_flips=0, center_crop=None, random_crop=False) -> Tuple[list, list]:
    train_transforms = list()
    validation_tranforms = list()
    if crop is not None:
        train_transforms.append(RandomCrop(*crop, always_apply=True))
    if normalize:
        train_transforms.append(Normalize())
        validation_tranforms.append(Normalize())
    if p_random_flips:
        flips = OneOf([HorizontalFlip(p=p_random_flips),
                        VerticalFlip(p=p_random_flips)], p=0.5)
        train_transforms.append(flips)
    if center_crop:
        assert type(center_crop) == tuple
        validation_tranforms.append(CenterCrop(*center_crop, always_apply=True))

    return train_transforms, validation_tranforms

def get_prediction_fig(image, gts, predictions):
    
    bldg_gt = gts[0][0]
    road_gt = gts[1]
    bldg_pred = predictions[0][0]
    road_pred = predictions[1]
    # print("bldg gt shape: ", bldg_gt.shape)
    # print("road gt shape: ", road_gt.shape)
    # print("bldg pred shape: ", bldg_pred.shape)
    # print("road pred shape: ", road_pred.shape)
    
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
    return fig  

def plot_buildings(gt_buildings, prediction_buildings):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(gt_buildings.permute(1,2,0))
    axs[1].imshow(prediction_buildings.permute(1,2,0))
    return fig

def check_dir_exists(dir):
    """
    Checks if path to dir exists, if not creates the dir
    """
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
            os.chmod(dir, 0o777)
            print(f"Directory: {dir} created.")
            return
        except:
            pass
    print(f"Directory: {dir} exists.")
    return



def train_validation_file_split(val_fold_id, data_to_load, csv_path="areas_of_interest/sn8_full_data.csv"):
    train_files = []
    val_files = []
    all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
    dict_template = {}
    for i in all_data_types:
        dict_template[i] = None

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            in_data = copy.copy(dict_template)
            for j in data_to_load:
                if j == "training_preds":
                    continue
                if j == "fold_id":
                    continue
                in_data[j]=row[j]
            if int(row["fold_id"]) == val_fold_id:
                val_files.append(in_data)
            else:
                train_files.append(in_data)
    return train_files, val_files
