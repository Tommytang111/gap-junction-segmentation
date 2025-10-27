#Transform GJ points to nearby entities if they exist within a certain radius
#October 21, 2025
#Tommy Tang

#Libraries
import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
import gc

def transform_points_to_nearby_entities(preds_path:str, points_path:str, radius:int=15):
    """
    """
    #Relevant paths
    points = np.load(points_path).astype(np.uint8)
    preds = np.load(preds_path).astype(np.uint8)

    points = block_reduce(points, block_size=(1, 2, 2), func=np.max)
    preds = block_reduce(preds, block_size=(1, 2, 2), func=np.max)
    print("Loaded and downsampled preds and points by a factor of 2 in x and y.")

    #Convert preds to entity array
    preds_filtered = cc3d.dust(preds, threshold=6, connectivity=26, in_place=False)
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