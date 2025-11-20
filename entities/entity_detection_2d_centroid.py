"""
Gap Junction Entity Detection Module With Centroid Based Matching
Detects and evaluates individual gap junction entities using 3D connected components.
Tommy Tang & Kirpa Chandok
October 2025
"""

import numpy as np
import cc3d  # Using cc3d for connected components
from scipy.optimize import linear_sum_assignment

class GapJunctionEntityDetector2DCentroid: 
    def __init__(self, threshold=0.5, iou_threshold=0.001, connectivity=8, min_size=25):
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
        
        # Ensure we have a 2D image
        if len(img.shape) > 2:
            # If there's a batch dimension or extra dimensions, squeeze or select first
            if img.shape[0] == 1:
                img = img.squeeze(0)
            else:
                raise ValueError(f"Expected 2D image, got shape {img.shape}")
        
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

        ## binary_img = cc3d.dust(img=binary_img, threshold=self.min_size, connectivity=8)
        
        # Use cc3d connected_components
        labelled_img, num_entities = cc3d.connected_components(
            binary_img,
            connectivity=8,
            return_N=True
        )
        
        for entity_id in range(1, num_entities + 1):  # Start from 1 to skip background (0)
            rows, cols = np.where(labelled_img == entity_id)
            pixel_positions = list(zip(rows, cols))  # List of tuples for this entity
            position_list.append(pixel_positions)
        
        return labelled_img, position_list, num_entities

    def get_centroid(self, entity_positions):
        """
        Calculate the centroid of an entity given its pixel positions.
        
        Args:
            entity_positions: List of (row, col) tuples for the entity
            
        Returns:
            centroid: (row, col) tuple representing the centroid
        """
        centroid = np.mean(entity_positions, axis=0)
        return tuple(centroid)

    def calculate_centroid_distance(self, entity1_positions, entity2_positions):
        """
        Calculate the Euclidean distance between the centroids of two entities.
        
        Args:
            entity1_positions: List of (row, col) tuples for entity 1
            entity2_positions: List of (row, col) tuples for entity 2
            
        Returns:
            distance: Euclidean distance between centroids
        """
        centroid1 = np.mean(entity1_positions, axis=0)
        centroid2 = np.mean(entity2_positions, axis=0)
        distance = np.linalg.norm(centroid1 - centroid2)
        return distance
    
    def match_entities(self, pred_img, gt_img, max_distance=50):
        """
        Match predicted entities to ground truth entities based on centroid distances.
        
        Args:
            pred_img: 2D numpy array of predicted segmentation
            gt_img: 2D numpy array of ground truth segmentation
            max_distance: Maximum allowed distance for a valid match
            
        Returns:
            matches: List of tuples (pred_entity_index, gt_entity_index)
            matched_pred: List of predicted entities that were matched
            matched_gt: List of ground truth entities that were matched
        """
        _, pred_entities, num_pred = self.extract_entities_2d(pred_img)
        _, gt_entities, num_gt = self.extract_entities_2d(gt_img)
        
        if num_pred == 0 or num_gt == 0:
            return [], [], []
        
        pred_centroids = np.array([self.get_centroid(entity) for entity in pred_entities])
        gt_centroids = np.array([self.get_centroid(entity) for entity in gt_entities])
        cost_matrix = np.linalg.norm(pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=-1)
        
        # Use Hungarian algorithm for optimal one-to-one matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        matched_pred = []
        matched_gt = []
        
        # Filter matches by distance threshold
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            if cost_matrix[pred_idx, gt_idx] <= max_distance:
                matches.append((pred_idx, gt_idx))
                matched_pred.append(pred_entities[pred_idx])
                matched_gt.append(gt_entities[gt_idx])
        
        return matches, matched_pred, matched_gt
    
    def entity_metrics_2d(self, pred_img, gt_img):
        """
        Calculate entity-based metrics for 2D segmentation using centroid-based matching.
        
        Args:
            pred_img: 2D numpy array of predicted segmentation
            gt_img: 2D numpy array of ground truth segmentation
            
        Returns:
            f1: F1 score
            precision: Precision (TP / (TP + FP))
            recall: Recall (TP / (TP + FN))
            metrics_dict: Dictionary with detailed metrics including TP, FP, FN, counts
        """
        _, _, num_pred = self.extract_entities_2d(pred_img)
        _, _, num_gt = self.extract_entities_2d(gt_img)
        
        matches, _, _ = self.match_entities(pred_img, gt_img)
        tp = len(matches)
        fp = num_pred - tp
        fn = num_gt - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_dict = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_pred': num_pred,
            'num_gt': num_gt
        }
        
        return f1, precision, recall, metrics_dict
    
    def entity_metrics_2d_batch(self, pred_imgs, gt_imgs):
        """
        Calculate entity-based metrics for a batch of 2D segmentation images.
        
        Args:
            pred_imgs: List of 2D numpy arrays of predicted segmentations
            gt_imgs: List of 2D numpy arrays of ground truth segmentations
            
        Returns:
            avg_f1: Average F1 score across the batch
            avg_precision: Average precision across the batch
            avg_recall: Average recall across the batch
            batch_metrics: List of metrics dictionaries for each image pair
        """
        if len(pred_imgs) != len(gt_imgs):
            raise ValueError("Number of predicted images and ground truth images must be the same.")
        
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        batch_metrics = []
        
        for pred_img, gt_img in zip(pred_imgs, gt_imgs):
            f1, precision, recall, metrics_dict = self.entity_metrics_2d(pred_img, gt_img)
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            batch_metrics.append(metrics_dict)
        
        n = len(pred_imgs)
        avg_f1 = total_f1 / n
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        
        return avg_f1, avg_precision, avg_recall, batch_metrics
    
    def compute_iou(self, entity1_positions, entity2_positions):
        """
        Compute Intersection over Union (IoU) between two entities.
        
        Args:
            entity1_positions: List of (row, col) tuples for entity 1
            entity2_positions: List of (row, col) tuples for entity 2
            
        Returns:
            iou: Intersection over Union value
        """
        set1 = set(entity1_positions)
        set2 = set(entity2_positions)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou