import csv
import copy
from typing import List, Tuple
import cv2
import os

from osgeo import gdal
import numpy as np
import torch
from torch.utils.data import Dataset

from albumentations import RandomCrop, Compose, Normalize, Resize

class SN8Dataset(Dataset):
    def __init__(self,
                 csv_filename: str,
                 data_to_load: List[str] = ["preimg","postimg","building","road","roadspeed","flood"],
                 img_size: Tuple[int, int] = (1300,1300),
                 transforms=None,
                 ):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images
        
        Parameters:
        ------------
        csv_filename (str): absolute filepath to the csv to load images from. the csv should have columns: preimg, postimg, building, road, roadspeed, flood.
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv. 
        img_size (tuple): the size of the input pre-event image in number of pixels before any augmentation occurs.
        
        """
        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
        
        self.img_size = img_size
        if data_to_load[0] != "preimg" and data_to_load[0] != "postimg":
            raise ValueError(f"First value in the data_to_load list must be preimg or postimg: recieved {data_to_load[0]}")
        self.data_to_load = data_to_load
        
        self.files = []

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None
        
        with open(csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    if j == "training_preds":
                        continue
                    in_data[j]=row[j]
                self.files.append(in_data)
        
        if transforms:
           img_size != (1300,1300) 
           self.transforms = self.get_transforms(transforms, data_to_load, img_size != (1300,1300))
        else:
            self.transforms = None
            
        print("loaded", len(self.files), "image filepaths")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_dict = self.files[index]

        returned_data = {}
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                if i == "postimg":
                    ds = self.get_warped_ds(data_dict["postimg"])
                else:
                    ds = gdal.Open(filepath)
                image = ds.ReadAsArray()
                ds = None
                if len(image.shape)==2: 
                    image = np.expand_dims(image, axis=0)  

                # up or down sample image, unless postimg, which is handelled by 
                # get_warped_ds   
                
                # if self.img_size != (1300, 1300) and i != "postimg":
                #     #image = resize(image, (self.img_size[0], self.img_size[1]))
                #     image = cv2.resize(image, (1300, 1300))
                
                # H W C -> C H W
                image = image.transpose(1, 2, 0)
                returned_data[i] = image
                

            else:
                returned_data[i] = None

        if self.transforms is not None:
            returned_data = self._transform(returned_data)
        
        for key in self.all_data_types: 
            if returned_data[key] is not None:
                returned_data[key] = torch.from_numpy(returned_data[key]).float().permute(2,0,1)
            else:
                # Necessary as default collate_fn dosen't handle None
                returned_data[key] = 0

        out = (returned_data["preimg"], 
                returned_data["postimg"],
                returned_data["building"],
                returned_data["road"],
                returned_data["roadspeed"],
                returned_data["flood"])            
        
        return out

    def get_image_filename(self, index: int) -> str:
        """ return pre-event image absolute filepath at index """
        data_dict = self.files[index]
        return data_dict["preimg"]

    def get_warped_ds(self, post_image_filename: str) -> gdal.Dataset:
        """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks 
        
        SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
        In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
        the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
        ds = gdal.Warp("", post_image_filename,
                       format='MEM', width=self.img_size[1], height=self.img_size[0],
                       resampleAlg=gdal.GRIORA_Bilinear,
                       outputType=gdal.GDT_Byte)
        return ds

    def get_transforms(self, transforms : list, data_to_load : dict, resize=False):
        additional_targets = {}
        if resize:
            transforms.insert(0, Resize(self.img_size[0], self.img_size[1]))
        for data_type in data_to_load[1:]:
            if data_type == 'preimg' or data_type == "postimg":
                additional_targets[data_type] = "image"
                continue
            additional_targets[data_type] = "mask"
        transforms = Compose(transforms, additional_targets=additional_targets)
        return transforms

    def _transform(self, data : dict):
        transform_data = {}
        transform_data["image"] = data[self.data_to_load[0]]
        for key in self.data_to_load[1:]:
            transform_data[key] = data[key]
        transform_data = self.transforms(**transform_data)
        data[self.data_to_load[0]] = transform_data["image"]
        for key in self.data_to_load[1:]:
            data[key] = transform_data[key]    
        return data


class SN8FloodDataset(SN8Dataset):

    def __init__(self, 
                csv_filename: str, 
                data_to_load: List[str] = ["preimg", "postimg", "building", "road", "roadspeed", "flood", "training_preds"],
                img_size: Tuple[int, int] = (1300, 1300), 
                transforms=None,
                training_preds_dir=None):
        
        super().__init__(csv_filename, data_to_load, img_size, transforms)
        self.all_data_types = ["preimg", 
                                "postimg", 
                                "building", 
                                "road", 
                                "roadspeed", 
                                "flood", 
                                "training_preds"]
        if "training_preds" in data_to_load:
            assert training_preds_dir is not None
        self.training_preds_dir = training_preds_dir 


    def __getitem__(self, index):
        data_dict = self.files[index]
        returned_data = {}
        
        for i in self.all_data_types:
            if i == "training_preds":
                if i not in self.data_to_load:
                    returned_data[i] = None
                    continue
                
                image_name = self.get_image_filename(index)
                training_path = os.path.join(self.training_preds_dir, os.path.basename(image_name).replace(".tif", ".npy"))
                with open(training_path, "rb") as f:
                    training_prediction = np.load(f)
                    training_prediction = np.concatenate(
                        (training_prediction["building_prediction"], training_prediction["road_prediction"])
                        , axis=0)
                    returned_data[i] = training_prediction.transpose(1, 2, 0)
                    continue
            
            filepath = data_dict[i]
            if filepath:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                if i == "postimg":
                    ds = self.get_warped_ds(data_dict["postimg"])
                else:
                    ds = gdal.Open(filepath)
                image = ds.ReadAsArray()
                ds = None
                if len(image.shape)==2: 
                    image = np.expand_dims(image, axis=0)  

                # up or down sample image, unless postimg, which is handelled by 
                # get_warped_ds   
                
                # if self.img_size != (1300, 1300) and i != "postimg":
                #     #image = resize(image, (self.img_size[0], self.img_size[1]))
                #     image = cv2.resize(image, (1300, 1300))
                
                # C H W ->  H W C
                image = image.transpose(1, 2, 0)
                returned_data[i] = image
            
            else:
                returned_data[i] = None

        if self.transforms is not None:
            returned_data = self._transform(returned_data)
        
        for key in self.all_data_types: 
            if returned_data[key] is not None:
                returned_data[key] = torch.from_numpy(returned_data[key]).float().permute(2,0,1)
            else:
                # Necessary as default collate_fn dosen't handle None
                returned_data[key] = 0

        out = (returned_data["preimg"], 
                returned_data["postimg"],
                returned_data["building"],
                returned_data["road"],
                returned_data["roadspeed"],
                returned_data["flood"],
                returned_data["training_preds"])            
        
        return out



if __name__ == "__main__":

                            
    
    transforms = [RandomCrop(512, 512, always_apply=True), Normalize()]

    # train_dataset = SN8Dataset("areas_of_interest/sn8_data_train.csv",
    #                         data_to_load=["preimg","postimg","flood", "road"], # ,"building", "road", "roadspeed", "flood"],
    #                         img_size=(1300, 1300),
    #                         transforms=transforms,
    #                         )
    

    train_dataset = SN8FloodDataset("areas_of_interest/sn8_data_val.csv",
                                data_to_load=["preimg",
                                            "postimg",
                                            "flood", 
                                            "road"], 
                                img_size=(1300, 1300),
                                transforms=transforms,
                                training_preds_dir="foundation_predictions/efficientnet-b3_JACC")

    print(train_dataset[30])
    


    # x = transforms(image=None)


    
    # import matplotlib.pyplot as plt

    # for x, image in enumerate(train_dataset):
            
    #     plt.imshow(image["image"])
    #     plt.savefig(f"testfiles/test{x}-pre.png")
    #     plt.imshow(image["postimg"])
    #     plt.savefig(f"testfiles/test{x}-post.png")
    #     plt.show()

    