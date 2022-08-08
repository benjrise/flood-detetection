"""
Loosley based on Selim_sef spacenet 6
"""
from multiprocessing.pool import Pool
import os
import csv
from traceback import print_tb
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr
from utils.utils import write_geotiff
import torch


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


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
    
    if gt_dir is not None:
        with open(os.path.join(gt_dir, image), "rb") as f:
            gts = np.load(f)
            # gt_building = gts[0,0]
            # gt_roadspeed = gts[1,:]

    road_predictions = np.array(road_predictions)
    road_prediction = average_strategy(road_predictions)
    building_predictions = np.array(building_predictions)
    building_prediction = average_strategy(building_predictions)
    
    building_prediction = torch.sigmoid(torch.from_numpy(building_prediction)).numpy()[0]
    road_prediction = torch.sigmoid(torch.from_numpy(road_prediction)).numpy()

    building_prediction = np.rint(building_prediction).astype(int)
    roadspeed_prediction = np.rint(road_prediction).astype(int)

    predictions = np.zeros((2, 8, building_prediction.shape[0], building_prediction.shape[1]))
    predictions[0,0] = building_prediction
    predictions[1,:] = roadspeed_prediction
            
    if save_preds_dir is not None:
        # THIS MAY NOT WORK CHECK
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

    # if save_fig_dir is not None:
        #if save_preds_dir is not None: # for some reason, seg fault when doing both of these. maybe file saving or something is interfering. so sleep for a little
        #    time.sleep(2) 
        # save_figure_filename = os.path.join("predictions/", os.path.basename(current_image_filename)[:-4]+"_pred.png")
        # make_prediction_png_roads_buildings(preimg, gts, predictions, save_figure_filename)

    return  metrics  


if __name__ == "__main__":
    predictions_dir = "predictions/"
    dirs = [os.path.join(predictions_dir, d) for d in os.listdir(predictions_dir)]
    n_threads = 16
    
    images = os.listdir(dirs[0])
    out_dir = "foundation_predictions/"
    gt_dir = "ground_truth/"
    args_list = []

    # print("here")
    for image in images:
        args_list.append((image, dirs, out_dir, gt_dir))
    

    running_tp = [0,0] 
    running_fp = [0,0]
    running_fn = [0,0]
    running_union = [0,0]

    with Pool(n_threads) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for metrics in pool.imap_unordered(ensemble_image, args_list):

                for idx, (union, tp, fp, fn) in enumerate(metrics):
                    running_tp[idx] += tp
                    running_fp[idx] += fp
                    running_fn[idx] += fn
                    running_union[idx] += union
                
                pbar.update()


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
    
    final_out_csv = os.path.join(out_dir, "final_out.csv")
    fields = ["data_type", "precision", "recall", "f1", "iou"]
    with open(final_out_csv, "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k in out:
            w.writerow({field: out[k].get(field) or k for field in fields})