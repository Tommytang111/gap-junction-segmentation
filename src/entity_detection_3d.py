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
        
        # Run 3D connected components with specified connectivity
        labeled_volume = cc3d.connected_components(
            binary_volume, 
            connectivity=26 # Using 26-connectivity for 3D
        )
        
        # Filter out small components (likely noise)
        if self.min_size > 1:
            labeled_volume = self._filter_small_components_3d(labeled_volume)
        
        # Count entities (excluding background which is 0)
        num_entities = np.max(labeled_volume)
        
        return labeled_volume, num_entities
    
    def _filter_small_components_3d(self, labeled_volume):
        """
        Remove connected components smaller than min_size.
        
        Args:
            labeled_volume: 3D array with labeled connected components
            
        Returns:
            filtered_volume: 3D array with small components removed and labels renumbered
        """
        unique_labels, counts = np.unique(labeled_volume, return_counts=True)
        
        # Keep labels that meet size threshold (excluding background 0)
        valid_labels = unique_labels[(counts >= self.min_size) & (unique_labels != 0)]
        
        # Create filtered volume with renumbered labels
        filtered_volume = np.zeros_like(labeled_volume)
        for new_label, old_label in enumerate(valid_labels, start=1):
            filtered_volume[labeled_volume == old_label] = new_label
        
        return filtered_volume
    
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
        pred_labeled, num_pred = self.extract_entities(pred_volume)
        gt_labeled, num_gt = self.extract_entities(gt_volume)
        
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
        
        # Match predicted entities to ground truth using IoU
        matched_pred = set()
        matched_gt = set()
        
        for pred_id in range(1, num_pred + 1):
            pred_mask = (pred_labeled == pred_id)
            best_iou = 0
            best_gt_id = None
            
            # Find best matching ground truth entity
            for gt_id in range(1, num_gt + 1):
                if gt_id in matched_gt:
                    continue
                
                gt_mask = (gt_labeled == gt_id)
                
                # Calculate IoU
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_id = gt_id
            
            # If we found a match above threshold, mark both as matched
            if best_iou >= iou_threshold and best_gt_id is not None:
                matched_pred.add(pred_id)
                matched_gt.add(best_gt_id)
        
        # Calculate metrics
        tp = len(matched_pred)  # True positives: correctly detected entities
        fp = num_pred - tp      # False positives: predicted but no match
        fn = num_gt - tp        # False negatives: ground truth but not detected
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_dict = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_pred': num_pred,
            'num_gt': num_gt,
            'iou_threshold': iou_threshold
        }
        
        return metrics_dict