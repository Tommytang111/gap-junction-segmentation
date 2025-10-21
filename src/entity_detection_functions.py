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


class GapJunctionEntityDetector:
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
    
    def extract_entities(self, volume): 
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
            connectivity=self.connectivity
        )
        
        # Filter out small components (likely noise)
        if self.min_size > 1:
            labeled_volume = self._filter_small_components(labeled_volume)
        
        # Count entities (excluding background which is 0)
        num_entities = np.max(labeled_volume)
        
        return labeled_volume, num_entities
    
    def _filter_small_components(self, labeled_volume):
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
    
    def calculate_entity_f1(self, pred_volume, gt_volume, iou_threshold=0.5):
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
        
        return f1, precision, recall, metrics_dict
    
    def get_entity_statistics(self, labeled_volume):
        """
        Calculate statistics about detected entities.
        
        Args:
            labeled_volume: 3D array with labeled entities
            
        Returns:
            stats_dict: Dictionary with size statistics for entities
        """
        unique_labels, counts = np.unique(labeled_volume[labeled_volume > 0], return_counts=True)
        
        if len(counts) == 0:
            return {
                'num_entities': 0,
                'mean_size': 0,
                'median_size': 0,
                'min_size': 0,
                'max_size': 0,
                'std_size': 0
            }
        
        return {
            'num_entities': len(counts),
            'mean_size': float(np.mean(counts)),
            'median_size': float(np.median(counts)),
            'min_size': int(np.min(counts)),
            'max_size': int(np.max(counts)),
            'std_size': float(np.std(counts))
        }


def evaluate_entity_metrics_batch(model, dataloader, entity_detector, device='cuda', three=False):
    """
    Evaluate entity-level metrics on a dataset (for training loop).
    
    This function is designed to work with existing training/validation dataloaders.
    It processes batches and calculates entity metrics for each volume.
    
    Args:
        model: Trained PyTorch model
        dataloader: PyTorch DataLoader (train/val/test)
        entity_detector: Instance of GapJunctionEntityDetector
        device: Device to run on ('cuda' or 'cpu')
        three: Boolean indicating if using 3D model (True) or 2D model (False)
        
    Returns:
        metrics_dict: Dictionary with averaged entity metrics
    """
    model.eval()
    
    all_f1_scores = []
    all_precisions = []
    all_recalls = []
    all_num_pred = []
    all_num_gt = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Entity Evaluation"):
            # Unpack batch (handles both 2-tuple and 3-tuple returns)
            if len(batch_data) == 3:
                X, y, _ = batch_data  # Has file paths
            else:
                X, y = batch_data  # No file paths
            
            X, y = X.to(device), y.to(device)
            
            # Get model predictions
            pred = model(X)
            pred_probs = torch.sigmoid(pred)
            
            # Process each volume in the batch
            batch_size = pred_probs.shape[0]
            for i in range(batch_size):
                # Extract single volume and convert to numpy
                if three:
                    # For 3D: shape is (1, depth, H, W) -> (depth, H, W)
                    pred_volume = pred_probs[i, 0].cpu().numpy()
                    gt_volume = y[i].cpu().numpy()
                else:
                    # For 2D: treat as single slice with depth 1 -> (1, H, W)
                    pred_volume = pred_probs[i, 0].cpu().numpy()[np.newaxis, ...]
                    gt_volume = y[i].cpu().numpy()[np.newaxis, ...]
                
                # Calculate entity F1
                f1, precision, recall, metrics = entity_detector.calculate_entity_f1(
                    pred_volume, 
                    gt_volume
                )
                
                all_f1_scores.append(f1)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_num_pred.append(metrics['num_pred'])
                all_num_gt.append(metrics['num_gt'])
    
    # Calculate mean metrics
    return {
        'entity_f1': np.mean(all_f1_scores) if all_f1_scores else 0.0,
        'entity_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'entity_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'avg_num_pred': np.mean(all_num_pred) if all_num_pred else 0.0,
        'avg_num_gt': np.mean(all_num_gt) if all_num_gt else 0.0,
        'total_samples': len(all_f1_scores)
    }


def evaluate_full_volume_entities(volume, gt_volume=None, threshold=127, min_size=10, 
                                   connectivity=26, save_path=None):
    """
    Evaluate entities in a full assembled 3D volume (for inference pipeline).
    
    This function is designed for use after volume assembly in inference pipeline.
    
    Args:
        volume: 3D numpy array of predictions (typically uint8, 0-255)
        gt_volume: Optional 3D numpy array of ground truth (for F1 calculation)
        threshold: Threshold for binarization (default 127 for uint8)
        min_size: Minimum entity size in voxels (default 10)
        connectivity: Connectivity for connected components (default 26)
        save_path: Optional path to save labeled volume
        
    Returns:
        results_dict: Dictionary with entity counts, statistics, and optional F1 metrics
    """
    print("\n" + "="*60)
    print("Calculating Entity-Level Metrics")
    print("="*60)
    
    # Initialize entity detector
    entity_detector = GapJunctionEntityDetector(
        threshold=threshold,
        min_size=min_size,
        connectivity=connectivity
    )
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume range: [{volume.min()}, {volume.max()}]")
    
    # Extract entities from prediction
    print("\nExtracting gap junction entities from predictions...")
    labeled_volume, num_entities = entity_detector.extract_entities(volume)
    
    print(f"✓ Found {num_entities} gap junction entities in volume")
    
    # Get entity statistics
    stats = entity_detector.get_entity_statistics(labeled_volume)
    
    print(f"\nEntity Size Statistics:")
    print(f"  Number of entities: {stats['num_entities']}")
    print(f"  Mean size: {stats['mean_size']:.1f} voxels")
    print(f"  Median size: {stats['median_size']:.1f} voxels")
    print(f"  Min size: {stats['min_size']} voxels")
    print(f"  Max size: {stats['max_size']} voxels")
    print(f"  Std dev: {stats['std_size']:.1f} voxels")
    
    results = {
        'num_entities': num_entities,
        'labeled_volume': labeled_volume,
        'statistics': stats
    }
    
    # Save labeled volume if path provided
    if save_path is not None:
        np.save(save_path, labeled_volume)
        print(f"✓ Saved labeled volume to {save_path}")
    
    # If ground truth provided, calculate F1 score
    if gt_volume is not None:
        print(f"\nCalculating entity-level F1 score against ground truth...")
        
        if gt_volume.shape != volume.shape:
            print(f"WARNING: Shape mismatch! GT: {gt_volume.shape}, Pred: {volume.shape}")
            print("Cannot calculate F1 score.")
        else:
            f1, precision, recall, metrics = entity_detector.calculate_entity_f1(
                volume, 
                gt_volume,
                iou_threshold=0.5
            )
            
            print(f"\n{'='*60}")
            print("Entity Detection Performance")
            print(f"{'='*60}")
            print(f"  F1 Score:           {f1:.4f}")
            print(f"  Precision:          {precision:.4f}")
            print(f"  Recall:             {recall:.4f}")
            print(f"  True Positives:     {metrics['tp']}")
            print(f"  False Positives:    {metrics['fp']}")
            print(f"  False Negatives:    {metrics['fn']}")
            print(f"  Predicted Entities: {metrics['num_pred']}")
            print(f"  GT Entities:        {metrics['num_gt']}")
            print(f"{'='*60}")
            
            results.update({
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'metrics': metrics
            })
    
    return results


# Quick test function
def test_entity_detection():
    """
    Quick test to verify entity detection is working.
    Run this to check installation before using in training/inference.
    """
    print("Testing entity detection...")
    print("="*60)
    
    # Test 1: Small 3D volume (quick sanity check)
    print("\nTest 1: Small 3D volume (10×50×50)")
    test_volume_small = np.zeros((10, 50, 50), dtype=np.uint8)
    test_volume_small[2:5, 10:15, 10:15] = 255
    test_volume_small[6:8, 30:35, 30:35] = 255
    
    detector = GapJunctionEntityDetector(threshold=127, min_size=5, connectivity=26)
    labeled, num_entities = detector.extract_entities(test_volume_small)
    
    print(f"  Found {num_entities} entities (expected 2)")
    assert num_entities == 2, f"Expected 2 entities, found {num_entities}"
    print("  ✓ Test 1 passed!")
    
    # Test 2: 10 slices of 512×512 images (realistic size)
    print("\nTest 2: Realistic volume (10×512×512)")
    test_volume_realistic = np.zeros((10, 512, 512), dtype=np.uint8)
    
    # Create 3 gap junctions in different locations
    # Gap junction 1: Top-left region, slices 1-3
    test_volume_realistic[1:4, 50:80, 50:80] = 255
    
    # Gap junction 2: Center region, slices 4-6
    test_volume_realistic[4:7, 220:260, 220:260] = 255
    
    # Gap junction 3: Bottom-right region, slices 7-9
    test_volume_realistic[7:10, 400:440, 400:440] = 255
    
    labeled, num_entities = detector.extract_entities(test_volume_realistic)
    
    print(f"  Volume shape: {test_volume_realistic.shape}")
    print(f"  Found {num_entities} entities (expected 3)")
    assert num_entities == 3, f"Expected 3 entities, found {num_entities}"
    print("  ✓ Test 2 passed!")
    
    # Get statistics
    stats = detector.get_entity_statistics(labeled)
    print(f"\n  Entity statistics:")
    print(f"    Mean size: {stats['mean_size']:.1f} voxels")
    print(f"    Size range: {stats['min_size']} - {stats['max_size']} voxels")
    
    # Test 3: Entity F1 calculation
    print("\nTest 3: F1 Score calculation")
    # Create ground truth (same as prediction for perfect score)
    gt_volume = test_volume_realistic.copy()
    
    f1, precision, recall, metrics = detector.calculate_entity_f1(
        test_volume_realistic, 
        gt_volume
    )
    
    print(f"  Perfect prediction test:")
    print(f"    F1: {f1:.3f} (expected 1.0)")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    assert f1 == 1.0, f"Expected F1=1.0 for perfect prediction, got {f1}"
    print("  ✓ Test 3 passed!")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("Entity detection is working correctly with realistic image sizes.")
    
    return True


if __name__ == "__main__":
    # Run test when module is executed directly
    test_entity_detection()
    print("\n✓ Entity detection module is working correctly!")
