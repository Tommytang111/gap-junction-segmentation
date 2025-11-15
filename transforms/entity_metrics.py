#Script to calculate entity metrics between predictions and point ground truths
#Tommy Tang
#Nov 13, 2025

import numpy as np
from skimage.measure import block_reduce
import cc3d
import time

def calculate_entity_metrics(preds_path, points_path, nerve_ring_mask_path):
    """
    Calculate entity-level precision, recall, and F1 score.
    
    Treats connected components in predictions as entities, and evaluates:
    - TP: entities containing at least one ground-truth point
    - FP: entities containing no points
    - FN: ground-truth points not falling within any entity
    
    Parameters
    ----------
    preds_path : str
        Path to binary prediction volume (.npy)
    points_path : str
        Path to binary point annotation volume (.npy)
    nerve_ring_mask_path : str
        Path to binary nerve ring mask (.npy)
    
    Returns
    -------
    f1, precision, recall, tp, fp, fn : float, float, float, int, int, int
    """
    #Load volumes
    preds = np.load(preds_path).astype(bool)
    points = np.load(points_path).astype(bool)
    nr_mask = np.load(nerve_ring_mask_path).astype(bool)
    
    #If all shapes are the same proceed, else raise error
    if preds.shape != points.shape or preds.shape != nr_mask.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, points {points.shape}, mask {nr_mask.shape}")
    
    #Get predictions only in nerve ring
    nr_preds = preds & nr_mask
    
    #Transform predictions to entities
    nr_preds_entities, max_entities = cc3d.connected_components(nr_preds, connectivity=26, return_N=True)
    
    #Create list of point coordinates
    points_list = np.argwhere(points == 255)

    #Get entity labels at point locations
    points_coords = np.argwhere(points)
    if len(points_coords) == 0:
        print("WARNING: No points found in ground truth")
        return 0.0, 0.0, 0.0, 0, max_entities, 0
    
    entity_labels_at_points = nr_preds_entities[points_coords[:, 0], 
                                                  points_coords[:, 1], 
                                                  points_coords[:, 2]]
    
    #TP: Entities that contain at least one point
    tp_entities = set(entity_labels_at_points[entity_labels_at_points > 0])
    tp = len(tp_entities)

    #FP: Entities that don't contain any points
    all_entities = set(range(1, max_entities + 1))
    fp_entities = all_entities - tp_entities
    fp = len(fp_entities)

    #FN: Points that don't fall within any entity (in background)
    fn = np.sum(entity_labels_at_points == 0)

    #Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall, tp, fp, fn

if __name__ == "__main__":
    #SEM Dauer 1
    f1, precision, recall, tp, fp, fn = calculate_entity_metrics(preds_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_1_s000-850/volume_block_downsampled4x.npy",
                                                                 points_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs_block_downsampled4x.npy",
                                                                 nerve_ring_mask_path="/home/tommy111/scratch/Neurons/SEM_dauer_1_NR_mask_downsampled4x.npy")
    print("SEM Dauer 1")
    print(f"TP (entities with points): {tp}")
    print(f"FP (entities without points): {fp}")
    print(f"FN (points not in any entity): {fn}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    #SEM Dauer 2
    f1, precision, recall, tp, fp, fn = calculate_entity_metrics(preds_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_2_s000-972/volume_block_downsampled4x.npy",
                                                                 points_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs_block_downsampled4x.npy",
                                                                 nerve_ring_mask_path="/home/tommy111/scratch/Neurons/SEM_dauer_2_NR_mask_downsampled4x.npy")
    print("SEM Dauer 2")
    print(f"TP (entities with points): {tp}")
    print(f"FP (entities without points): {fp}")
    print(f"FN (points not in any entity): {fn}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    #SEM Adult
    f1, precision, recall, tp, fp, fn = calculate_entity_metrics(preds_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled8x.npy",
                                                                 points_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled8x.npy",
                                                                 nerve_ring_mask_path="/home/tommy111/scratch/Neurons/SEM_adult_NR_mask_downsampled8x.npy")
    print("SEM Adult")
    print(f"TP (entities with points): {tp}")
    print(f"FP (entities without points): {fp}")
    print(f"FN (points not in any entity): {fn}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")