""" loss functions. also found in cresi repo """
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets, weights = 1.0):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        weights = weights.contiguous()

        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        weights = weights.contiguous().view(-1)[non_ignored]

        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)

        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return ((-(1. - pt) ** self.gamma * torch.log(pt)) * weights).mean()



# Note that eps must be less than 1e-8 when using FP16
def soft_dice_loss(outputs, targets, per_image=False):
    '''
    From cannab sn4
    '''
    
    batch_size = outputs.size()
    # batch_size = outputs.size()[0]
    eps = 1e-7
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def focal(outputs, targets, gamma=2,  ignore_index=255):
    '''From cannab sn4'''
    outputs = outputs.contiguous()
    targets = targets.contiguous()
    eps = 1e-6
    non_ignored = targets.view(-1) != ignore_index
    targets = targets.view(-1)[non_ignored].float()
    outputs = outputs.contiguous().view(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1. - eps)
    targets = torch.clamp(targets, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()


def jaccard_loss(outputs, targets, eps=1e-6):
    jaccard_target = (targets == 1).float()
    jaccard_output = torch.sigmoid(outputs)
    intersection = (jaccard_output * jaccard_target).sum()
    # We count the interesection twice here so have to remove it in 
    # Jaccard calc
    union_plus_intersection = jaccard_output.sum() + jaccard_target.sum()
    jaccard_score = (
        (intersection + eps) / 
        (union_plus_intersection - intersection + eps)
        )
    jaccard_loss =  (1. - jaccard_score)
    return jaccard_loss

def jaccard_loss_multi_class(outputs, targets, eps=1e-6):
    jaccard_target = (targets == 1).float()
    jaccard_output = F.softmax(outputs, dim=1)
    intersection = (jaccard_output * jaccard_target).sum()
    # We count the interesection twice here so have to remove it in 
    # Jaccard calc
    union_plus_intersection = jaccard_output.sum() + jaccard_target.sum()
    jaccard_score = (
        (intersection + eps) / 
        (union_plus_intersection - intersection + eps)
        )
    jaccard_loss =  (1. - jaccard_score)
    return jaccard_loss



class ComputeLoss(nn.Module):
    def __init__(self, 
                building_weight=0.5,
                road_weight=0.5,
                bce_loss=0., 
                road_dice_loss=0., 
                road_focal_loss=0.,
                building_jaccard=0., 
                building_focal=0.,
                building_dice=0.) -> None:
        super().__init__()
        self.bce_loss = bce_loss
        self.road_dice_loss = road_dice_loss
        self.road_focal_loss = road_focal_loss
        self.building_jaccard = building_jaccard
        self.building_focal = building_focal
        self.building_dice = building_dice

        self.building_weight = building_weight
        self.road_weight = road_weight
        self.stash_metrics = {}

    def forward(self, out_buildings, out_roads, targets_buildings, targets_roads): 
        building_loss = 0
        road_loss = 0
        out_roads = torch.sigmoid(out_roads)

        if self.bce_loss:
            bce_loss = \
                    self.bce_loss * F.binary_cross_entropy_with_logits(out_buildings, targets_buildings)
            building_loss +=  bce_loss
            self.stash_metrics["bce_loss"] = bce_loss.item()

        if self.road_dice_loss:
            road_dice_loss =\
                 self.road_dice_loss *  soft_dice_loss(out_roads, targets_roads)
            road_loss += road_dice_loss
            self.stash_metrics["road_dice_load"] = road_dice_loss.item()
        
        if self.road_focal_loss:
            # road_focal_loss = self.road_focal_loss * focal(out_roads, targets_roads)
            road_focal_loss = self.road_focal_loss * focal(out_roads, 
                                                    targets_roads).mean()
            road_loss +=  road_focal_loss
            self.stash_metrics["road_focal_loss"] = road_focal_loss.item()
        
        if self.building_focal:
            building_focal_loss = self.building_focal * focal(out_buildings, targets_buildings)
            building_loss +=  building_focal_loss
            self.stash_metrics["building_focal_loss"] = building_focal_loss.item()
        
        if self.building_jaccard:
            building_jaccard_loss = self.building_jaccard * jaccard_loss(out_buildings, targets_buildings)
            building_loss +=  building_jaccard_loss
            self.stash_metrics["building_jaccard_loss"] = building_jaccard_loss.item()

        if self.building_dice:
            building_dice_loss = self.building_dice * soft_dice_loss(out_buildings, targets_buildings)
            building_loss +=  building_dice_loss
            self.stash_metrics["building_dice_loss"] = building_dice_loss.item()

        loss = self.building_weight * building_loss + self.road_weight * road_loss
        self.stash_metrics["loss"] = loss.item()
        return loss

