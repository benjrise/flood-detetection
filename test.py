
from http.client import CannotSendRequest
import torch
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Rectangle
from pyparsing import TokenConverter
from datasets.datasets import SN8Dataset, return_batched_patches
from utils.utils import get_transforms
import albumentations as A
from models.hrnet.hr_config import get_hrnet_config
from models.hrnet.hrnet import HighResolutionNet, get_seg_model
from models.efficientnet.efficient_unet import EfficientNet_Unet



train_transforms, val_transforms = get_transforms()

val_dataset = SN8Dataset("areas_of_interest/sn8_data_val.csv",
                        data_to_load=["preimg","building","roadspeed"],
                        img_size=(2600, 2600),
                        transforms=val_transforms
                        )


config = get_hrnet_config("models/hrnet/hr_config.yml")
model = get_seg_model(config)
model.load_state_dict(torch.load("upsample_experiments/HRENT_UPSAMPLEX2_lr1.00e-04_bs8_01-08-2022-19-55/best_model.pth"))
image =  val_dataset[0][0].unsqueeze(0)
out_image = torch.zeros_like(image)
mask = val_dataset[0][4].unsqueeze(0)
images_batched, mask,  patches_order = return_batched_patches(image, [mask], patch_size=(1300, 1300), bs=1)

print(patches_order)

buildings = []
roads = []
for batch in images_batched:
    print("here")
    print(batch.shape)
    pred_building, pred_road = model(batch)
    buildings.append(pred_building)
    roads.append(pred_road)

buildings = torch.cat(buildings)
roads = torch.cat(roads) 

buildings_out = torch.zeros(1, buildings.shape[1], 2600, 2600)
ones = torch.zeros_like(buildings_out)
print(buildings.shape)
print(roads.shape)

for building_patch, patch in zip(buildings, patches_order):
    buildings_out[:,:, patch[0]:patch[2], patch[1]:patch[3]] += building_patch
    ones[:,:, patch[0]:patch[2], patch[1]:patch[3]] += 1

buildings_out= buildings_out/ones
full_images = torch.cat(images_batched)
print(ones)
#print(mask[end])
for image, patch in zip (full_images, patches_order):
    out_image[:, :, patch[0]:patch[2], patch[1]:patch[3]] = image


# fig, axes = plt.subplots()
plt.imshow(buildings_out.squeeze(0).permute(1, 2 , 0).detach().numpy()) 
#axes[1].imshow(mask[0].squeeze(0).permute(1, 2 , 0).detach().numpy())
plt.savefig("reconstruct")

plt.imshow(ones.squeeze(0).permute(1, 2 , 0).detach().numpy()) 
plt.savefig("ones")

#x = [torch.rand(1, 3, 512, 512), torch.rand(1, 3, 512, 512)]
# y = torch.cat(x)
# print(y.shape)
print([[] for _ in range(3)])


# def calculate_slice_bboxes(
#     image_height: int,
#     image_width: int,
#     slice_height: int = 512,
#     slice_width: int = 512,
#     overlap_height_ratio: float = 0.2,
#     overlap_width_ratio: float = 0.2,
# ):
#     """
#     Given the height and width of an image, calculates how to divide the image into
#     overlapping slices according to the height and width provided. These slices are returned
#     as bounding boxes in xyxy format.
#     :param image_height: Height of the original image.
#     :param image_width: Width of the original image.
#     :param slice_height: Height of each slice
#     :param slice_width: Width of each slice
#     :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
#     :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
#     :return: a list of bounding boxes in xyxy format
#     """

#     slice_bboxes = []
#     y_max = y_min = 0
#     y_overlap = int(overlap_height_ratio * slice_height)
#     x_overlap = int(overlap_width_ratio * slice_width)
#     while y_max < image_height:
#         x_min = x_max = 0
#         y_max = y_min + slice_height
#         while x_max < image_width:
#             x_max = x_min + slice_width
#             if y_max > image_height or x_max > image_width:
#                 xmax = min(image_width, x_max)
#                 ymax = min(image_height, y_max)
#                 xmin = max(0, xmax - slice_width)
#                 ymin = max(0, ymax - slice_height)
#                 slice_bboxes.append([xmin, ymin, xmax, ymax])
#             else:
#                 slice_bboxes.append([x_min, y_min, x_max, y_max])
#             x_min = x_max - x_overlap
#         y_min = y_max - y_overlap
#     return slice_bboxes

# print(image.shape)
# out = calculate_slice_bboxes(2600, 2600 , slice_width=650, slice_height=650, overlap_height_ratio=0., overlap_width_ratio=0.)
# fig, ax = plt.subplots(1, 2)

# ax[0].imshow(image.permute(1, 2, 0))
# for rect in out:
#     x = rect[0]
#     y = rect[1]
#     width = rect[2]-rect[0]
#     height = rect[3]-rect[1]
#     patch = Rectangle((x, y), width, height, linewidth=1,edgecolor='r',facecolor='none')
#     ax[0].add_patch(patch)

# print(out[0])
# crop_transform = A.Crop(*out[3])

# image = image.permute(1, 2, 0).numpy()

# tranformed = crop_transform(image=image)
# ax[1].imshow(tranformed["image"])


plt.savefig("crops.png")
