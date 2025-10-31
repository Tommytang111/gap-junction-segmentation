#Transform GJ points to nearby entities if they exist within a certain radius
#October 21, 2025
#Tommy Tang

#Libraries
import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
import gc

def transform_points_to_nearby_entities(preds_path:str, points_path:str, radius:int=10, dust_size:int=6):
    """
    Transform a points array into an entity array by keeping only the closest entity to a point within a specified radius. 
    This function performs a connected component analysis to identify entities, then computes a distance transform to find 
    the closest Euclidian distance to every existing point.

    This loads two 3D .npy volumes with shape (Z, Y, X):
    - preds_path: predicted binary mask (non-zero = foreground).
    - points_path: binary point annotations (non-zero = point locations).

    Processing pipeline:
    - Downsample both volumes in-plane by a factor of 2 using max pooling
      (block_reduce(..., block_size=(1, 2, 2))) to reduce memory and noise.
    - Remove small speckles from the predictions with cc3d.dust(threshold=dust_size, connectivity=26).
    - Label remaining connected components (entities) with 26-connectivity (uint32 labels).
    - Compute a Euclidean distance transform from the point mask.
    - Keep only entity labels that intersect voxels within `radius` voxels of any point.

    Parameters
    - preds_path (str): Path to .npy file of the predicted binary volume.
    - points_path (str): Path to .npy file of the point-annotation volume.
    - radius (int, default=15): Neighborhood radius in voxels in the downsampled grid
      (after the 2× downsampling in X and Y). Adjust accordingly if you need a radius
      in the original resolution.
    - dust_size (int, default=6): Minimum size of connected components to keep in preds.

    Returns
    - filtered_entity_array (np.ndarray): uint32 labeled volume (same shape as the
      downsampled inputs) where only entities near points are retained; others set to 0.
    - num_entities (int): Total number of connected components before filtering.

    Notes
    - Inputs are cast to uint8; any non-zero value is treated as foreground/point.
    - Uses 26-connectivity for 3D components.
    - np.load errors (e.g., missing files) will propagate.
    """
    #Relevant paths
    points = np.load(points_path).astype(np.uint8)
    preds = np.load(preds_path).astype(np.uint8)

    #Optional downsampling depending on RAM constraints for faster processing
    points = block_reduce(points, block_size=(1, 2, 2), func=np.max)
    preds = block_reduce(preds, block_size=(1, 2, 2), func=np.max)
    print("Loaded and downsampled preds and points by a factor of 2 in x and y.")

    #Convert preds to entity array
    preds_filtered = cc3d.dust(preds, threshold=dust_size, connectivity=26, in_place=False)
    entities, num_entities = cc3d.connected_components(preds_filtered, connectivity=26, return_N=True, out_dtype=np.uint32)
    print("Entity array computed.")

    # Convert points to boolean array
    points_bool = (points > 0)

    # Distance transform to distance array using inverse (bitwise NOT) of points mask
    dist_to_points = distance_transform_edt(~points_bool).astype(np.float32)
    print("Distance transform to points computed.")

    # Create boolean mask of voxels within the specified radius of any point
    near_points = dist_to_points <= radius

    # Keep only entities that overlap with near_points
    # Step 1: Get intersection of near_points and entity voxels (using boolean AND) and keep only those labels
    keep_labels = np.unique(entities[near_points & (entities > 0)])
    
    # Step 2: Filter entity array to keep only labels found in Step 1
    filtered_entity_array = np.where(np.isin(entities, keep_labels), entities, 0)
    print("Filtered entity array computed.")

    return filtered_entity_array, num_entities

def move_points_to_gap_junctions():
    """
    Transforms points in a point array such that each point is moved to the nearest predicted/real gap junction entity.
    """
    
if __name__ == "__main__":
    point_entities, num_entities = transform_points_to_nearby_entities("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_1_s000-850/volume.npy",
                                    "/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs.npy",
                                    radius=7
                                    )
    #Save point entities to scratch
    np.save("/home/tommy111/scratch/sem_dauer_1_GJs_entities.npy", point_entities)
    print(f'Saved point entities to /home/tommy111/scratch/sem_dauer_1_GJs_entities.npy')
    print(f'Number of entities found: {num_entities}')

    #Downsample and save transformed points for 3D visualization comparison
    downsampled_point_entities = block_reduce(point_entities, block_size=(1, 4, 4), func=np.max)
    np.save("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs_entities_downsampled.npy", downsampled_point_entities)
    print(f'Saved downsampled point entities to /home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs_entities_downsampled.npy')
    
    #Garbage collection
    del point_entities, num_entities, downsampled_point_entities
    gc.collect()
    
    ####SECOND SET OF POINTS####
    point_entities, num_entities = transform_points_to_nearby_entities("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_2_s000-972/volume.npy",
                                    "/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs.npy",
                                    radius=7
                                    )
    #Save point entities to scratch
    np.save("/home/tommy111/scratch/sem_dauer_2_GJs_entities.npy", point_entities)
    print(f'Saved point entities to /home/tommy111/scratch/sem_dauer_2_GJs_entities.npy')
    print(f'Number of entities found: {num_entities}')

    #Downsample and save transformed points for 3D visualization comparison
    downsampled_point_entities = block_reduce(point_entities, block_size=(1, 4, 4), func=np.max)
    np.save("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs_entities_downsampled.npy", downsampled_point_entities)
    print(f'Saved downsampled point entities to /home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs_entities_downsampled.npy')
    print('Second set finished!')