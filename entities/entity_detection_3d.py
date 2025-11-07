"""
Gap Junction Entity Detection Module
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""

import numpy as np
import cc3d
import torch
from tqdm import tqdm


class GapJunctionEntityDetector3D:
    """
    Detects gap junction entities from 3D volumes using connected components analysis.
    
    Uses 26-connectivity by default, which considers voxels as neighbors if they touch
    in any direction (including diagonals) - appropriate for biological structures.
    """
    
    def __init__(self, threshold=0.5, min_size=30, connectivity=26):
        """
        Initialize the entity detector.
        
        Args:
            threshold: Probability threshold to binarize predictions (0-1 for probabilities, 
                      or 0-255 for uint8 images). Default 0.5 for probabilities.
            min_size: Minimum number of voxels for a valid entity (filters noise). Default 5.
            connectivity: 6, 18, or 26 for 3D connected components. Default 26.
                         - 6: Face neighbors only
                         - 18: Face and edge neighbors
                         - 26: All neighbors including diagonals (recommended)
        """
        self.threshold = threshold
        self.min_size = min_size
        self.connectivity = connectivity
    
    def extract_entities_3d(self, volume): 
        """
        Extract connected component entities from a 3D volume.
        
        Args:
            volume: 3D numpy array (D, H, W) - can be binary, uint8 (0-255), or probabilities (0-1)
            
        Returns:
            labeled_volume: 3D array where each entity has a unique integer label (0 = background)
            num_entities: Number of detected entities (excluding background)
        """
        # Handle different input types
        if volume.dtype == np.float32 or volume.dtype == np.float64:
            # Probability values (0-1)
            binary_volume = (volume >= self.threshold).astype(np.uint8)
        elif volume.dtype == np.uint8:
            # Uint8 values (0-255) -- though i'm pretty sure the output predictions are all in this form
            if volume.max() > 1:
                binary_volume = (volume > 127).astype(np.uint8)
            else:
                binary_volume = volume.astype(np.uint8)
        else:
            # Assume already binary
            binary_volume = (volume > 0).astype(np.uint8)
            
        # Filter out small components (likely noise)
        if self.min_size > 1:
            filtered_volume = cc3d.dust(labeled_volume, threshold=self.min_size, connectivity=self.connectivity, in_place=False)
        
        # Run 3D connected components with specified connectivity
        labeled_volume, num_entities = cc3d.connected_components(
            filtered_volume, 
            connectivity=self.connectivity,
            return_N=True
        )

        voxel_list = []

        for entity_id in range(1, num_entities + 1):
            rows, cols, depths = np.where(labeled_volume == entity_id)
            voxel_positions = list(zip(rows, cols, depths))
            voxel_list.append(voxel_positions)
        
        return labeled_volume, voxel_list, num_entities
    
    def calculate_iou(self, entity1_voxels, entity2_voxels):
        """
        Calculate IoU between two entities represented as lists of voxel positions.
        """

        set1 = set(entity1_voxels)
        set2 = set(entity2_voxels)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
        
        return intersection / union

    
    def calculate_metrics_3d(self, pred_volume, gt_volume, iou_threshold=0.5):
        """
        Calculate F1 score for entity detection using IoU-based matching.
        
        Matches predicted entities to ground truth entities based on Intersection over Union (IoU).
        An entity is considered correctly detected if IoU >= iou_threshold.
        
        Args:
            pred_volume: 3D predicted volume (probabilities, uint8, or binary)
            gt_volume: 3D ground truth volume (probabilities, uint8, or binary)
            iou_threshold: Minimum IoU to consider a match (default 0.5)
            
        Returns:
            f1: F1 score
            precision: Precision (TP / (TP + FP))
            recall: Recall (TP / (TP + FN))
            metrics_dict: Dictionary with detailed metrics including TP, FP, FN, counts
        """
        # Extract entities from both volumes
        pred_labeled, pred_voxel_list, num_pred = self.extract_entities_3d(pred_volume)
        gt_labeled, gt_voxel_list, num_gt = self.extract_entities_3d(gt_volume)
        
        # Handle edge cases
        if num_pred == 0 and num_gt == 0:
            return 1.0, 1.0, 1.0, {
                'tp': 0, 'fp': 0, 'fn': 0,
                'num_pred': 0, 'num_gt': 0
            }
        
        if num_pred == 0:
            return 0.0, 0.0, 0.0, {
                'tp': 0, 'fp': 0, 'fn': num_gt,
                'num_pred': 0, 'num_gt': num_gt
            }
        
        if num_gt == 0:
            return 0.0, 0.0, 0.0, {
                'tp': 0, 'fp': num_pred, 'fn': 0,
                'num_pred': num_pred, 'num_gt': 0
            }
        
        matched_pred_indices = []
        matched_gt_indices = []
        shared_voxels = []  # Store actual voxel positions of matched entities
        shared_entities = 0
        
        for pred_idx, pred_entity_voxels in enumerate(pred_voxel_list):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_entity_voxels in enumerate(gt_voxel_list):
                if gt_idx in matched_gt_indices:
                    continue
                
                iou = self.calculate_iou(pred_entity_voxels, gt_entity_voxels)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # If best IoU exceeds threshold, consider it a match
            if best_iou >= self.iou_threshold:
                matched_pred_indices.append(pred_idx)
                matched_gt_indices.append(best_gt_idx)

                # Add the positions of this matched entity
                shared_voxels.extend(pred_entity_voxels)
                num_shared_entities += 1
        
        # Calculate metrics
        tp = len(matched_pred_indices)  # True positives: correctly detected entities
        fp = num_pred - tp      # False positives: predicted but no match
        fn = num_gt - tp        # False negatives: ground truth but not detected
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_dict = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        return shared_voxels, metrics_dict, num_shared_entities
