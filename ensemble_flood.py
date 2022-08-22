from multiprocessing.pool import Pool
import os
import csv
from config_flood import get_multi_flood_config
from models.efficientnet.efficient_unet import EfficientNet_Unet_Double
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr
from utils.utils import check_dir_exists, write_geotiff
import torch
import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir")
    parser.add_argument("--ground_truth_dir",
                        type=str,
                        )
    parser.add_argument("--out_dir",
                        type=str,
                        )
    args = parser.parse_args()
    return args

def downsample(image, channels, output_size=(1300,1300), input_size=(2600,2600)):
    input_size = input_size[0]
    output_size = output_size[0]
    bin_size = input_size // output_size
    image = image.reshape((channels, output_size, bin_size,
                                        output_size, bin_size)).max(4).max(2)
    return image

def average_strategy(image):
    return np.average(image, axis=0)

def ensemble_flood(args):
    image, dirs, out_dir, gt_dir = args
    
    flood_predictions = []
    current_image_filename = ''

    for dir in dirs:
        path = os.path.join(dir, image)
        with open(path, "rb") as f:
            in_prediction = np.load(f)
            flood_prediction = in_prediction["flood_prediction"]
            flood_predictions.append(flood_prediction)
            current_image_filename = str(in_prediction["tif_path"])
    
    flood_prediction = average_strategy(np.array(flood_predictions))
    flood_prediction = torch.nn.functional.softmax(
            torch.from_numpy(flood_prediction), dim=1).cpu().numpy()[0]
    flood_prediction = np.argmax(flood_prediction, axis=0)
    down_flood = downsample(flood_prediction, channels=1)[0]

    if out_dir is not None:
        ds = gdal.Open(current_image_filename)
        geotran = ds.GetGeoTransform()
        xmin, xres, rowrot, ymax, colrot, yres = geotran
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjectionRef())
        ds = None
        output_tif = os.path.join(out_dir, os.path.basename(current_image_filename.replace(".tif","_floodpred.tif")))
        nrows, ncols = down_flood.shape
        write_geotiff(output_tif, ncols, nrows,
                    xmin, xres, ymax, yres,
                    raster_srs, [down_flood])
    
    metrics = [[], [], [], []]
    print(gt_dir)
    if gt_dir is not None:
        print("here")
        with open(os.path.join(gt_dir, image), "rb") as f:
            gt_flood = np.load(f) 

        for j in range(4): # there are 4 classes
            gt = np.where(gt_flood==(j+1), 1, 0) # +1 because classes start at 1. background is 0
            prediction = np.where(flood_prediction==(j+1), 1, 0)
            
            #gts[i] = gt_flood
            #predictions[i] = flood_prediction

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
    
    return metrics



if __name__ == "__main__":
    
    #args = parse_args()
   
    config = get_multi_flood_config(1)
    
        
    predictions_dir = config.MULTI_SAVE_PATH
    if config.GT_SAVE:
        gt_dir = config.GT_SAVE_PATH
    else:
        gt_dir = None
    out_dir = config.FINAL_SAVE_PATH
    os.makedirs(out_dir, 0o777, exist_ok=True)
    
    
    dirs = [os.path.join(predictions_dir, d) for d in os.listdir(predictions_dir)]
    n_threads = 16
    images = os.listdir(dirs[0])
    args_list = []
    for image in images:
        args_list.append((image, dirs, out_dir, gt_dir))

    running_tp = [0, 0, 0, 0]
    running_fp = [0, 0, 0, 0]
    running_fn = [0, 0, 0, 0]
    running_union = [0, 0, 0, 0]

    with Pool(n_threads) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for metrics in pool.imap_unordered(ensemble_flood, args_list):
                if gt_dir is not None:
                    for idx, (union, tp, fp, fn) in enumerate(metrics):
                        running_tp[idx] += tp
                        running_fp[idx] += fp
                        running_fn[idx] += fn
                        running_union[idx] += union
                pbar.update()
    
    if gt_dir is not None:
        data = ["non-flooded building", "flooded building", "non-flooded road", "flooded road"]
        out = {data_type : {}  for data_type in data}
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

        final_out_csv = os.path.join(out_dir, "final_out.csv")
        fields = ["data_type", "precision", "recall", "f1", "iou"]
        with open(final_out_csv, "w", newline='') as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for k in out:
                w.writerow({field: out[k].get(field) or k for field in fields})

