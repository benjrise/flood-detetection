"""
Loosley based on Selim_sef spacenet 6
"""
import argparse
import csv
from genericpath import exists
from operator import gt
import os
from multiprocessing.pool import Pool
import cv2

import numpy as np
import torch
from osgeo import gdal, osr
from tqdm import tqdm

from config_foundation import get_multi_config, get_multi_config2, get_multi_config3
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from utils.utils import check_dir_exists, write_geotiff
import matplotlib.pyplot as plt

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def downsample(image, channels, output_size=(1300,1300), input_size=(2600,2600)):
    input_size = input_size[0]
    output_size = output_size[0]
    bin_size = input_size // output_size
    image = image.reshape((channels, output_size, bin_size,
                                        output_size, bin_size)).max(4).max(2)
    return image


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--predictions_dir",
#                         type=str,
#                         default="predictions")
#     parser.add_argument("--ground_truth_dir",
#                         type=str,
#                         default="ground_truth")
#     parser.add_argument("--out_dir",
#                         type=str,
#                         default="foundation_predictions")
#     parser.add_argument("--train_predictions_dir",
#                         type=str)
#     parser.add_argument("--train_out")
#     args = parser.parse_args()
#     return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int)
    args = parser.parse_args()
    return args

def watershed_mod(mask, thresh_h=0.6, thresh_l=0.4, distance=5, conn=2, watershed_line=True):
    mask0 = mask > thresh_h
    local_maxi = peak_local_max(mask, indices=False, footprint=np.ones((distance*2+1, distance*2+1)),
                            labels=(mask>thresh_l))
    local_maxi[mask0] = True
    seed_msk = ndi.label(local_maxi)[0]

    mask = watershed(-mask, seed_msk, mask=(mask > thresh_l), watershed_line=watershed_line)
    mask = measure.label(mask, connectivity=conn, background=0).astype('uint8')   
    mask[mask>0] = 1
    return mask

def average_strategy(images):
    return np.average(images, axis=0)

def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.


def ensemble_image(args_list):
    image, dirs, out_dir, gt_dir = args_list
    save_preds_dir = out_dir
    
    building_predictions = []
    road_predictions = []
    current_image_filename = ''
    for dir in dirs:
        path = os.path.join(dir, image)
        with open(path, "rb") as f:
            in_prediction = np.load(f)
            building_predictions.append(in_prediction["building_prediction"])
            road_predictions.append(in_prediction["road_prediction"])
            current_image_filename = str(in_prediction["tif_path"])
            output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_roadspeedpred.tif")))
            if os.path.exists(output_tif):
                return [[],[],]


    road_predictions = np.array(road_predictions)
    road_prediction = average_strategy(road_predictions)
    building_predictions = np.array(building_predictions)
    building_prediction = average_strategy(building_predictions)
    
    building_prediction = torch.sigmoid(torch.from_numpy(building_prediction)).numpy()[0]
    road_prediction = torch.sigmoid(torch.from_numpy(road_prediction)).numpy()


    building_prediction = watershed_mod(building_prediction, 
                    thresh_l=0.4, thresh_h=0.6)
    building_prediction = np.rint(building_prediction).astype(int)
    roadspeed_prediction = np.rint(road_prediction).astype(int)

    predictions = np.zeros((2, 8, building_prediction.shape[0], building_prediction.shape[1]))
    predictions[0,0] = building_prediction
    predictions[1,:] = roadspeed_prediction

    building_prediction = downsample(building_prediction, channels=1)[0]
    roadspeed_prediction = downsample(road_prediction, channels=8)

    if save_preds_dir is not None:
        # road_pred_arr = (road_prediction * 255).astype(np.uint8) # to be compatible with the SN5 eval and road speed prediction, need to mult by 255
        road_pred_arr = (roadspeed_prediction * 255).astype(np.uint8)
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
    
    metrics = [[], []]
    if gt_dir is not None:
        with open(os.path.join(gt_dir, image), "rb") as f:
            gts = np.load(f)
            gt_building = gts["gt_building"]
            gt_roadspeed = gts["gt_roadspeed"]
        
        gts = np.zeros((2, 8, gt_building.shape[-2], gt_building.shape[-1]))
        gts[0,0] = gt_building
        gts[1,:] = gt_roadspeed
        
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
            tp = np.sum(tp).astype(int)
            fp = np.sum(fp).astype(int)
            fn = np.sum(fn).astype(int)

            metrics[j].append(union)
            metrics[j].append(tp)
            metrics[j].append(fp)
            metrics[j].append(fn)
        
    return  metrics  

def ensemble_train_image(args_list):
    image, dirs, out_dir = args_list
    building_predictions = []
    road_predictions = []
    current_image_filename = ''
    
    for dir in dirs:
        path = os.path.join(dir, image)
        with open(path, "rb") as f:
            in_prediction = np.load(f)
            building_predictions.append(in_prediction["building_prediction"])
            road_predictions.append(in_prediction["road_prediction"])
            current_image_filename = str(in_prediction["tif_path"])
    

    road_predictions = np.array(road_predictions)
    road_prediction = average_strategy(road_predictions)
    building_predictions = np.array(building_predictions)
    building_prediction = average_strategy(building_predictions)
    
    building_prediction = torch.sigmoid(torch.from_numpy(building_prediction)).numpy()[0]
    road_prediction = torch.sigmoid(torch.from_numpy(road_prediction)).numpy()

    building_prediction = np.rint(building_prediction).astype(int)
    roadspeed_prediction = np.rint(road_prediction).astype(int)

    out = os.path.join(out_dir, os.path.basename(current_image_filename).replace(".tif","npy"))
    with open(out, "wb") as f:
        np.savez(f, building_prediction=building_prediction, 
                    roadspeed_prediction=roadspeed_prediction[-1])
    # make_prediction_png_roads_buildings(preimg, gts, predictions, save_figure_filename)

def ensemble(predictions_dir, out_dir, gt_dir=None):
    dirs = [os.path.join(predictions_dir, d) for d in os.listdir(predictions_dir)]
    n_threads = 16
    images = os.listdir(dirs[0])
    images = [image for image in os.listdir(dirs[0]) if image[-4:] != ".csv"]

    args_list = []
    for image in images:
        args_list.append((image, dirs, out_dir, gt_dir))

    running_tp = [0,0] 
    running_fp = [0,0]
    running_fn = [0,0]
    running_union = [0,0]

    with Pool(n_threads) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for metrics in pool.imap_unordered(ensemble_image, args_list):
                if gt_dir is not None:
                    for idx, (union, tp, fp, fn) in enumerate(metrics):
                        running_tp[idx] += tp
                        running_fp[idx] += fp
                        running_fn[idx] += fn
                        running_union[idx] += union
                pbar.update()
    if gt_dir is not None:
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
        
            final_out_csv = os.path.join(out_dir, f"final_out.csv")
            fields = ["data_type", 
                    "precision", 
                    "recall", 
                    "f1", 
                    "iou",]
            with open(final_out_csv, "w", newline='') as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                for k in out:
                    w.writerow({field: out[k].get(field) or k for field in fields})
            

    # if config.SAVE_TRAINING_PREDS:
    #     out_dir = config.TRAINING_SAVE_DIR
    #     check_dir_exists(out_dir)
    #     dirs = [os.path.join(config.MULTI_TRAIN_SAVE_PATH, d) for d in os.listdir(config.MULTI_TRAIN_SAVE_PATH)]
    #     n_threads = 16
    #     args_list = []
    #     images = os.listdir(dirs[0])

    #     for image in images:
    #         args_list.append((image, dirs, out_dir))
        
    #     with Pool(n_threads) as pool:
    #         with tqdm(total=len(args_list)) as pbar:
    #             for _ in pool.imap_unordered(ensemble_train_image, args_list):
    #                 pbar.update()
        

if __name__ == "__main__":
    args = parse_args()
    if args.v == 1:
        config = get_multi_config(1)
    if args.v == 2:
        config = get_multi_config2(1)
    if args.v == 3:
        config = get_multi_config3(1)
    else:
        config = get_multi_config(1)

    # config.PATH_SAVE_PREDS = "foundation_preds/train_out"
    # config.PATH_SAVE_FINAL_PREDS = "foundation_preds/test_out"
    out_dir = config.PATH_SAVE_FINAL_PREDS
    predictions_dir = config.PATH_SAVE_PREDS

    if config.TESTING:
        gt_dir=None
    else:
        gt_dir = config.PATH_GT_SAVE
    
    os.makedirs(out_dir, 0o777, exist_ok=True)
    ensemble(predictions_dir=predictions_dir, out_dir=out_dir, gt_dir=gt_dir)

    