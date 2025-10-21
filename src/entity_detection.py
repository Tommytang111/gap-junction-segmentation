"""
Gap Junction Entity Detection Module
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""

import numpy as np
import cc3d
from cc3d import connected_components
import torch
from tqdm import tqdm

class GapJunctionEntityDetector: 
    def __init__(self, threshold = 0.5, min_size = 8, connectivity = 8): 
        self.threshold = threshold ##threshold to binarize pixels
        self.min_size = min_size ##minimum number of voxels to be considered a gap junction entity
        self.connectivity = connectivity ##number of neighbours cc3d library is checking 

    def extract_entities_2d(self, img): 
        """
        Extract connected component entities from a 2D image. 
        
        Args:
            img: 2D numpy array (H, W) - can be binary, uint8 (0-255), or probabilities (0-1)
            
        Returns:
            position_list: list of the positions of any labelled entities
            num_entities: Number of detected entities (excluding background)
        """

        position_list = []

        if img is None:
            raise ValueError("Input image is None")


        if img.dtype == np.float32 or img.dtype == np.float64:
            ## numpy array has probability values for each pixel
            binary_img = (img >= self.threshold).astype(np.uint8)
        elif img.dtype == np.uint8:
            if img.max() > 1: ##img contains pixel values of either 0 or 255
                binary_img = (img >= 127).astype(np.uint8)
            else: ##img contains pixel values of either 0 or 1
                binary_img = img.astype(np.uint8)
        else:
            binary_img = (img > 0).astype(np.uint8)
        
        labelled_img = connected_components(
            binary_img,
            connectivity = 8
        )

        num_entities = np.max(binary_img)

        for pred_id in range(1, num_entities): 
            rows, cols = np.where(labelled_img == pred_id)
            pixel_positions = np.array(list(zip(rows, cols))) ##makes a list of the position of each pixel with that label
            position_list.append(pixel_positions)

        return position_list, num_entities
    
    def entity_metrics_2d(self, pred_img, gt_img):
        matched_entities = []
        iou = 0

        pred_positions, pred_entities = GapJunctionEntityDetector.extract_entities_2d(pred_img)
        gt_positions, gt_entities = GapJunctionEntityDetector.extract_entities_2d(gt_img)

        pred_set = set(map(tuple, pred_positions)) ## make each vector in prediction position array into a tuple 
        ## then create a set of these tuples
        gt_set = set(map(tuple, gt_positions)) 

        for entity in range(1, pred_entities + 1):
            union = len(pred_set & gt_set)
            intersection = len(pred_set | gt_set)
            if union > 0:
                iou = intersection / union
            
            if iou >= self.threshold:
                matched_entities.append(entity)
        
        tp = len(matched_entities)
        fp = pred_entities - tp
        fn = gt_entities - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_dict = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return matched_entities, metrics_dict
        


        


        

    