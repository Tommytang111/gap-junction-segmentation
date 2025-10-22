"""
Gap Junction Entity Detection Module
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""

import numpy as np
import cc3d  # Using cc3d for connected components

class GapJunctionEntityDetector2D:
    def __init__(self, threshold=0.5, iou_threshold=0.001, connectivity=8, min_size=30):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.connectivity = connectivity  # 6, 18, or 26 for 3D; 4 or 8 for 2D
        self.min_size = min_size  # Minimum size to consider a 3d component valid
    
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
        
        # Convert to binary image
        if img.dtype == np.float32 or img.dtype == np.float64:
            # numpy array has probability values for each pixel
            binary_img = (img >= self.threshold).astype(np.uint8)
        elif img.dtype == np.uint8:
            if img.max() > 1:  # img contains pixel values of either 0 or 255
                binary_img = (img >= 127).astype(np.uint8)
            else:  # img contains pixel values of either 0 or 1
                binary_img = img.astype(np.uint8)
        else:
            binary_img = (img > 0).astype(np.uint8)
        
        # Use cc3d connected_components
        labelled_img, num_entities = cc3d.connected_components(
            binary_img,
            connectivity=8
            return_N=True
        )
                
        for entity_id in range(1, num_entities + 1):  # Start from 1 to skip background (0)
            rows, cols = np.where(labelled_img == entity_id)
            pixel_positions = list(zip(rows, cols))  # List of tuples for this entity
            position_list.append(pixel_positions)

        gj_pixels = np.sum(binary_img)
        
        return labelled_img, position_list, num_entities
    
    def calculate_iou(self, entity1_positions, entity2_positions):
        """
        Calculate IoU between two entities represented as lists of pixel positions.
        """
        set1 = set(entity1_positions)
        set2 = set(entity2_positions)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def entity_metrics_2d(self, pred_img, gt_img):
        """
        Calculate entity-based metrics for 2D segmentation.
        Returns both the matched entity indices and their actual positions.
        """
        matched_pred_indices = []
        matched_gt_indices = []
        shared_positions = []  # Store actual positions of matched entities
        shared_entities = 0
        
        pred_labelled, pred_positions, pred_entities = self.extract_entities_2d(pred_img)
        gt_labelled gt_positions, gt_entities = self.extract_entities_2d(gt_img)
        
        # Compare each predicted entity with each GT entity
        for pred_idx, pred_entity_positions in enumerate(pred_positions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_entity_positions in enumerate(gt_positions):
                iou = self.calculate_iou(pred_entity_positions, gt_entity_positions)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If best IoU exceeds threshold, consider it a match
            if best_iou >= self.iou_threshold:
                matched_pred_indices.append(pred_idx)
                # Add the positions of this matched entity
                shared_positions.extend(pred_entity_positions)
                shared_entities += 1
                if best_gt_idx not in matched_gt_indices:
                    matched_gt_indices.append(best_gt_idx)
        
        tp = len(matched_pred_indices)  # True positives: predicted entities that matched
        fp = pred_entities - tp  # False positives: predicted entities that didn't match
        fn = gt_entities - len(matched_gt_indices)  # False negatives: GT entities that weren't matched
        
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
        
        # Return both indices and actual positions
        return shared_positions, metrics_dict, shared_entities
    
    def visualize_matches(self, pred_img, gt_img):
        """
        Create visualization showing matched entities.
        """
        # Get matched entities
        matched_pred, metrics, shared_entities_num = self.entity_metrics_2d(pred_img, gt_img)
        
        # Create image showing only matched predictions
        shared_img = np.zeros_like(pred_img)
        
        # Get the labeled image
        if pred_img.dtype == np.float32 or pred_img.dtype == np.float64:
            binary_pred = (pred_img >= self.threshold).astype(np.uint8)
        elif pred_img.dtype == np.uint8:
            if pred_img.max() > 1:
                binary_pred = (pred_img >= 127).astype(np.uint8)
            else:
                binary_pred = pred_img.astype(np.uint8)
        else:
            binary_pred = (pred_img > 0).astype(np.uint8)
        
        labelled_pred = cc3d.connected_components(binary_pred, connectivity=8)
        
        # Only show matched entities
        for idx in matched_pred:
            entity_id = idx + 1  # Entity IDs start from 1
            shared_img[labelled_pred == entity_id] = 1
        
        return shared_img, metrics
    