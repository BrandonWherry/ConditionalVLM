"""
Module that contains all useful regional explainability algorithms and functions
"""
#Imports

from skimage import segmentation
from pytorch_grad_cam import XGradCAM, GradCAM, FullGrad, GradCAMPlusPlus, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from multiprocessing import Pool

import torchvision
from torchvision import models as tvmodels
from torchsummary import summary

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision.models as torchvisionmodels

import os
import numpy as np
import cv2
import argparse
import sys

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools
import more_itertools

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from captum.attr import LayerGradCam
from captum.attr import visualization
from PIL import Image
import shutil

import numpy as np
from dask_image.imread import imread
from dask_image import ndfilters, ndmorph, ndmeasure
import matplotlib.pyplot as plt
from dask_image import ndmeasure

from operator import itemgetter
from time import perf_counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_grayscale_grad_cam(image, SMU_class_index, model):
    input_tensor = image.to(device)
    targets = [ClassifierOutputTarget(SMU_class_index)]
    #target_layers = [model.layer4[-1]]
    target_layers = [model.layer2]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return(grayscale_cam)

def segmentation_info(image, num_segments, compactness):
    img_np = image.detach().cpu().squeeze().numpy()
    segments_slic = slic(img_np, n_segments = num_segments, compactness=compactness,
                     start_label=1)
    num_segments = len(np.unique(segments_slic))
    list_unique_regions = np.unique(segments_slic)
    segment_pixel_num_list = []
    total_pixels = 0
    for i in (list_unique_regions):
        num_pixels = np.count_nonzero(segments_slic == i)
        segment_pixel_num_list.append(num_pixels)
        total_pixels += num_pixels
    
    
    information_for_each_segment = []
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(segment_pixel_num_list[i-1])
        image_list.append(total_pixels)
        information_for_each_segment.append(image_list)

    return(information_for_each_segment, segments_slic, num_segments)


# I want to get the average attribution score for each segment
def cam_processor_for_segments(grayscale_cam_output, segments_slic):
    list_unique_regions = np.unique(segments_slic)
    region_attr_score = []
    final_region_attr_score = []
    num_pixels_in_region_list = []
    
    for i in (list_unique_regions):
        row_counter = 0
        column_counter = 0
        region_attr_score = []
        num_pixels_in_region = 0
        for row in grayscale_cam_output:
            for cell in row:
                current_score = grayscale_cam_output[row_counter, column_counter]
                current_region = segments_slic[row_counter, column_counter]
                if current_region == i:
                    region_attr_score.append(current_score)
                    num_pixels_in_region += 1
                column_counter +=1
            row_counter += 1
            column_counter = 0
        avg_score = np.mean(region_attr_score)
        final_region_attr_score.append(avg_score)
        num_pixels_in_region_list.append(num_pixels_in_region)
    
    unique_region_info = []
    n = 0
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(final_region_attr_score[n])
        image_list.append(num_pixels_in_region_list[n])
        image_list.append(np.sum(num_pixels_in_region_list))
        unique_region_info.append(image_list)
        n += 1
    return(unique_region_info)


def get_feature_masks(image, attributions, segments_slic):
    segments_slic_1 = segments_slic
    features = []
    for i in attributions:
        feature = np.where(i==segments_slic_1, 1, 0)
        features.append(feature)
        
    return(features)


def attribution_ranker(cam_processor_for_segments_output, num_top_attr):
    ranked_images = sorted(cam_processor_for_segments_output, key=itemgetter(1), reverse=True)
    top_ranked_features = []
    for i in range(num_top_attr):
        top_ranked_features.append(ranked_images[i][0])
        
    return top_ranked_features



def image_rankings(get_image_versions):
    #for idx in iterative_Grad_CAM_counterfactual_masking_output
    ranked_images = sorted(get_image_versions, key=itemgetter(3))
    
    return ranked_images

def blur_image_from_attribution(image, attribution_map):
    # attribution map is the attributions after being passed through the attribution processor
    # image is a tensor
    # will output the blurred image based on the attribution map
    
    
    #average_img = image.squeeze().cpu().permute(1, 2, 0).numpy()
    #avg = np.average(average_img)
    #blurred_img = cv2.GaussianBlur(image.squeeze().cpu().permute(1, 2, 0).numpy(), (181, 181), 0)
    avg = np.float32(-1)
    #avg_img = np.where(average_img > 9999, average_img, avg)
    
    #attribution_map = attribution_map.detach().squeeze().cpu().numpy()
    
    mask = [attribution_map]
    mask = np.array(mask).squeeze()

    out = np.where(mask==np.array([0]), image.squeeze().cpu().numpy(), avg)

    return torch.tensor(out).unsqueeze(0).unsqueeze(0)



def segmentation_info(segment_mask):
    num_segments = len(np.unique(segment_mask))
    list_unique_regions = np.unique(segment_mask)
    segment_pixel_num_list = []
    total_pixels = 0
    for i in (list_unique_regions):
        num_pixels = np.count_nonzero(segment_mask == i)
        segment_pixel_num_list.append(num_pixels)
        total_pixels += num_pixels
    
    
    information_for_each_segment = []
    n = 0
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(segment_pixel_num_list[n])
        image_list.append(total_pixels)
        information_for_each_segment.append(image_list)
        n +=1

    return(information_for_each_segment, segment_mask, num_segments)


def segmentation_info_slic(image, num_segments, compactness):
    img_np = image.detach().cpu().squeeze().numpy()
    segments_slic = slic(img_np, n_segments = num_segments, compactness=compactness,
                     start_label=1)
    num_segments = len(np.unique(segments_slic))
    list_unique_regions = np.unique(segments_slic)
    segment_pixel_num_list = []
    total_pixels = 0
    for i in (list_unique_regions):
        num_pixels = np.count_nonzero(segments_slic == i)
        segment_pixel_num_list.append(num_pixels)
        total_pixels += num_pixels
    
    
    information_for_each_segment = []
    for i in (list_unique_regions):
        image_list = []
        image_list.append(i)
        image_list.append(segment_pixel_num_list[i-1])
        image_list.append(total_pixels)
        information_for_each_segment.append(image_list)

    return(information_for_each_segment, segments_slic, num_segments)


# def segmentation_info_felzenszwalb(image, scale, sigma, min_size):
#     img_np = image.detach().cpu().squeeze().numpy()
#     segments_felz = felzenszwalb(img_np, scale=scale, sigma=sigma, min_size=min_size)
#     num_segments = len(np.unique(segments_slic))
#     list_unique_regions = np.unique(segments_slic)
#     segment_pixel_num_list = []
#     total_pixels = 0
#     for i in (list_unique_regions):
#         num_pixels = np.count_nonzero(segments_slic == i)
#         segment_pixel_num_list.append(num_pixels)
#         total_pixels += num_pixels
    
    
#     information_for_each_segment = []
#     for i in (list_unique_regions):
#         image_list = []
#         image_list.append(i)
#         image_list.append(segment_pixel_num_list[i-1])
#         image_list.append(total_pixels)
#         information_for_each_segment.append(image_list)

#     return(information_for_each_segment, segments_slic, num_segments)


def softmax_score(num_total_pixels, num_obf_pixels, model, image, SMU_class_index):
    #image = good_img_transform(image)
    image = image
    logits = model(image).cpu()
    #print(logits)
    probs = F.softmax(logits, dim=1)
    probs = probs.detach().cpu()
    probs = probs.tolist()[0]
    probs = probs[SMU_class_index]
    return probs


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def region_explainability(image, segment_mask: np.array, top_n_start: int, model: torch.nn.Module, 
                          SMU_class_index, threshold: float, 
                          top_n_stop: int, MAX_BATCH_SZ: int = 16,
                          PRUNE_HEURISTIC: int = 3):
    if not(next(model.parameters()).is_cuda):
        print('Model is not on GPU')
        return -1
    # Get attribution map
    explainability_mask = get_grayscale_grad_cam(image, SMU_class_index, model)
    # Get segment mask
    #seg = segmentation_info_slic(image = image, num_segments = 25, compactness = 1)
    seg = segmentation_info(segment_mask = segment_mask)
    # Calculate average attribution in each superpixel
    avg_attr_scores = cam_processor_for_segments(grayscale_cam_output = explainability_mask, 
                                                 segments_slic = seg[1])
    # Sort the regions by average attribution, make num_top_attr = the number of segments in the image
    top_attrs = attribution_ranker(cam_processor_for_segments_output = avg_attr_scores, 
                                   num_top_attr = seg[2])
    features_1 = get_feature_masks(image = image, attributions = top_attrs, 
                                   segments_slic = seg[1])
    # features_1 gives us a sorted list of feature masks. 
    # Element at position 0 is the top attribution region mask
    top_n = top_n_start
    score = 1000
    prob = 1
    
    sm1 = softmax(model(image.to(device)).cpu().detach().numpy()).squeeze()
    sm_idx1 = np.unravel_index(np.argmax(sm1), sm1.shape)[0]
    prediction = sm_idx1 
    confidence = sm1[sm_idx1]

    powerset_list = [0] 
    total_images_analyzed = 0
    searches_in_current_depth = 0
    best_masked_image = None
    start = perf_counter()
    # print('Starting with region search depth:', top_n)
    features_list = features_1[0:top_n]
    features_nums = list(range(len(features_list)))
    powerset_list = list(more_itertools.powerset(features_nums))
    powerset_list = [ele for ele in powerset_list if len(ele) != 0]
    unique_image_info = []
    #num_pixels_changed holds the count of the number of pixels that are obfuscated
    num_pixels_changed = []
    #total_attr_list I think gives us the label of the regions that are being obfuscated
    total_attr_list = []
    #scores holds the score given to the image with regions obfuscated
    scores = []
    
    # The top_n_stop can't be greater than the number of features
    if len(features_1) < top_n_stop:
        top_n_stop = len(features_1)
    
    while True:
        
        # getting all combinations of features as a list, based on their index
        if searches_in_current_depth == len(powerset_list):
            top_n += 1
            # PRUNE_HEURISTIC += 1
            searches_in_current_depth = 0
            features_list = features_1[0:top_n]
            features_nums = list(range(len(features_list)))
            powerset_list = list(more_itertools.powerset(features_nums))
            powerset_list = [ele for ele in powerset_list if features_nums[-1] in ele]
            # powerset_list = [ele for ele in powerset_list if len(ele) > PRUNE_HEURISTIC]
            print('Number of images analyzed so far:', total_images_analyzed)
            print('Increasing search depth to', top_n, 'regions\n')        
        
        should_use_max_batch_size = MAX_BATCH_SZ <= len(powerset_list) - searches_in_current_depth
        if should_use_max_batch_size:
            batch_size = MAX_BATCH_SZ
        else:
            batch_size = len(powerset_list) - searches_in_current_depth
        
        image_tensor_batch = torch.zeros(batch_size, 1, 28, 28).to(device)
        total_attribution = list()
        num_changes = list()
        total_num_pixels = list()
        
        for num in range(batch_size):
            total_attribution.append(np.zeros((28, 28)))
            # num_changes.append(np.count_nonzero(total_attribution[-1]))
            total_num_pixels.append(total_attribution[-1].size)
            for i in range(len(powerset_list[searches_in_current_depth])):
                total_attribution[num] += features_list[powerset_list[searches_in_current_depth][i]]
            num_changes.append(np.count_nonzero(total_attribution[-1]))
            obfuscated_image = blur_image_from_attribution(image = image,
                               attribution_map = total_attribution[num]).to(device)
            image_tensor_batch[num] = obfuscated_image.detach().clone().squeeze(0)
            searches_in_current_depth += 1
            
        np_output = model(image_tensor_batch).cpu().detach().numpy()
        sm2 = np.apply_along_axis(softmax, 1, np_output)
        sm_idx2 = np.unravel_index(np.argmax(sm2), sm2.shape)
        img_index = sm_idx2[0]
        cf_prediction = sm_idx2[1]
        cf_confidence = sm2[sm_idx2]
        total_images_analyzed += batch_size
        if (prediction != cf_prediction) and (cf_confidence > threshold):
            unique_image_info.append(image_tensor_batch[img_index])
            unique_image_info.append(num_changes[img_index])
            unique_image_info.append(total_num_pixels[img_index])
            unique_image_info.append([confidence, cf_confidence])
            unique_image_info.append([prediction, cf_prediction])
            unique_image_info.append(total_attr_list)
            unique_image_info.append(top_n)
            unique_image_info.append(avg_attr_scores)
            unique_image_info.append(total_images_analyzed)
            print('Counterfactual found at depth:', top_n, 'regions')
            print('Total Number of Counterfactuals tested:', total_images_analyzed)
            end = perf_counter() - start
            print(f'Total Search time: {end:.2f}')
            break
            
        if top_n == top_n_stop and searches_in_current_depth == len(powerset_list):
            print('Counterfactual not found up to depth (including):', top_n, 'regions')
            print('Total Number of Counterfactuals tested:', total_images_analyzed)
            end = perf_counter() - start
            print(f'Total Search time: {end:.2f}')
            return -1
        
    return unique_image_info
