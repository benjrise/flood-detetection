import torch
from torchvision.transforms.functional import crop

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def get_patches(img_h, img_w,  split_height, split_width, overlap_y=0.2, overlap_x=0.2):
    X_points = start_points(img_w, split_width, overlap_y)
    Y_points = start_points(img_h, split_height, overlap_x)
    patches = []
    for i in Y_points:
        for j in X_points:
            x1 = j
            y1 = i
            x2 = j+split_width
            y2 = i+split_height
            patches.append((y1, x1, y2, x2))
    return patches

def return_batched_patches(image, 
                        patch_size, 
                        bs=4, 
                        overlap_height_ratio : float = 0.2, 
                        overlap_width_ratio: float = 0.2):

    image_h = image.shape[2]
    image_w = image.shape[3]
    patches = get_patches(image_h, 
                        image_w, 
                        patch_size[0],
                        patch_size[1],
                        overlap_height_ratio,
                        overlap_width_ratio)
    patch_order = []
    image_slices = []
    for patch in patches:
        y, x = patch[0], patch[1]
        h = patch[2]-patch[0]
        w = patch[3]-patch[1]
        crop_slice = crop(image, y, x, h, w)
        image_slices.append(crop_slice)
        patch_order.append(patch)

    image_batches = []
    for i in range(-(len(image_slices)//-bs)):
        start = i*bs
        end = i*bs+bs
        if end > len(image_slices):
            end = len(image_slices)
        img_sec = image_slices[start:end]
        image_batches.append(torch.cat(img_sec))
    return image_batches, patch_order

def validate_patches(image, 
                    model, 
                    device, 
                    patch_size, 
                    bs=4,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2):
    
    """
    Patch_size 1024, overlap=0.2, seems to work well
    """
    
    img_shape =(image.shape[-2], image.shape[-1])
    images_batched, patches_order = return_batched_patches(image, 
                                                patch_size, bs=bs,
                                                overlap_height_ratio=overlap_height_ratio,
                                                overlap_width_ratio=overlap_width_ratio)
    buildings = []
    roads = []
    model.to(device)
    for batch in images_batched:
            batch = batch.to(device)
            pred_building, pred_road = model(batch)
            buildings.append(pred_building.cpu())
            roads.append(pred_road.cpu())
    
    buildings = torch.cat(buildings)
    roads = torch.cat(roads)
    buildings_out = torch.zeros(1, buildings.shape[1], *img_shape)
    roads_out = torch.zeros(1, roads.shape[1], *img_shape)
    ones_b = torch.zeros_like(buildings_out)  
    ones_r = torch.zeros_like(roads_out)

    for building_patch, roads_patch, patch in zip(buildings, roads, patches_order):
        buildings_out[:, :, patch[0]:patch[2], patch[1]:patch[3]] += building_patch
        roads_out[:, :, patch[0]:patch[2], patch[1]:patch[3]] += roads_patch
        ones_b[:,:, patch[0]:patch[2], patch[1]:patch[3]] += 1
        ones_r[:, :, patch[0]:patch[2], patch[1]:patch[3]] += 1
    
    buildings_out = buildings_out/ones_b
    roads_out = roads_out/ones_r
    return buildings_out, roads_out

def validate_flood_patches(image,
                    postimg,
                    model, 
                    device, 
                    patch_size, 
                    bs=4,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2):
    
    """
    Patch_size 1024, overlap=0.2, seems to work well
    """
    
    img_shape =(image.shape[-2], image.shape[-1])
    images_batched, patches_order = return_batched_patches(image, 
                                                patch_size, bs=bs,
                                                overlap_height_ratio=overlap_height_ratio,
                                                overlap_width_ratio=overlap_width_ratio)
    postimg_batched, _ = return_batched_patches(postimg,
                    patch_size, bs=bs,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio)
    
    flood_patches = []
    model.to(device)
    for pre_batch, post_batch in zip(images_batched, postimg_batched):
            pre_batch = pre_batch.to(device)
            post_batch = post_batch.to(device)
            flood_pred = model(pre_batch, post_batch)
            flood_patches.append(flood_pred.cpu())
    
    flood_patches = torch.cat(flood_patches)
    flood_out = torch.zeros(1, flood_patches.shape[1], *img_shape)
    ones = torch.zeros_like(flood_out)  

    for flood_patch, patch in zip(flood_patches, patches_order):
        flood_out[:, :, patch[0]:patch[2], patch[1]:patch[3]] += flood_patch
        ones[:,:, patch[0]:patch[2], patch[1]:patch[3]] += 1
    
    
    flood_out /= ones
    return flood_out