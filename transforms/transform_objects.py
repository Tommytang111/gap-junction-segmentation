"""
A class to transform volumes, masks, points and other objects for 2D/3D visualization or quantatitive analysis. 
Tommy Tang
Last Updated: Dec 16th, 2025
"""

#Libraries
import numpy as np
import pandas as pd
import cv2
import os
import gc
from pathlib import Path
from time import time
from skimage.measure import block_reduce
from skimage.morphology import remove_small_holes, disk
from scipy.ndimage import binary_dilation, generate_binary_structure, distance_transform_edt, binary_fill_holes, binary_closing
import cc3d
from src.utils import check_output_directory

def calculate_entity_metrics(preds:str|np.ndarray, points:str|np.ndarray, nerve_ring_mask:str|np.ndarray=None, verbose:bool=True) -> tuple[float, float, float, int, int, int]:
    """
    Compute detection metrics (F1, precision, recall) for predicted 3D entities against point annotations.

    The function:
    1. Loads predicted binary volume (preds_path) and ground-truth point volume (points_path).
    2. Optionally masks predictions to a nerve ring mask (nerve_ring_mask_path).
    3. Labels connected components (entities) in the masked prediction volume (26-connectivity).
    4. Maps each ground-truth point to the entity label at its coordinate.
    5. Derives counts:
        - TP: number of distinct entity labels that contain ≥1 point.
        - FP: number of entity labels with zero points.
        - FN: number of points falling in background (label 0).
    6. Calculates precision, recall, F1.

    Parameters
    ----------
    preds : str | np.ndarray
        Path to .npy file or .npy file containing binary predictions (foreground >0).
    points : str | np.ndarray
        Path to .npy file or .npy file containing point annotations (nonzero voxels are points).
    nerve_ring_mask : str | np.ndarray | None, default None
        Optional .npy mask to restrict evaluation region. If provided, must match shape.
    verbose : bool, default True
        If True, prints TP/FP/FN and all other returned values.

    Returns
    -------
    tuple[float, float, float, int, int, int]
        (f1, precision, recall, tp, fp, fn)

    Notes
    -----
    - All input volumes must share identical shape.
    - Connected components use 26-connectivity (3D).
    - If no points exist, returns zeros (f1=precision=recall=0, tp=fp=fn=0).
    - Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 harmonic mean.
    """
    #Load volumes
    preds = np.load(preds).astype(bool) if isinstance(preds, str) else preds
    points = np.load(points).astype(bool) if isinstance(points, str) else points
    if isinstance(nerve_ring_mask, str):
        nr_mask = np.load(nerve_ring_mask).astype(bool) 
    elif isinstance(nerve_ring_mask, np.ndarray):
        nr_mask = nerve_ring_mask
    else:
        nr_mask = None
        
    #If all shapes are the same proceed, else raise error
    if nr_mask is not None:
        if preds.shape != points.shape or preds.shape != nr_mask.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, points {points.shape}, mask {nr_mask.shape}")
    else:
        if preds.shape != points.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, points {points.shape}")
    
    #Get predictions only in nerve ring
    nr_preds = preds & nr_mask if nr_mask is not None else preds
    
    #Transform predictions to entities
    nr_preds_entities, max_entities = cc3d.connected_components(nr_preds, connectivity=26, return_N=True)

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
    
    #Verbose output
    if verbose:
        print(f"TP (entities with points): {tp}")
        print(f"FP (entities without points): {fp}")
        print(f"FN (points not in any entity): {fn}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    return f1, precision, recall, tp, fp, fn
    
def downsample(array:str|np.ndarray, block_size:tuple[int,...], save:bool=True, save_path:str=None) -> np.ndarray:
    """
    Downsample a NumPy volume (e.g., image stack, mask, segmentation) by applying
    skimage.measure.block_reduce with a max aggregation.

    Parameters
    ----------
    array : str | np.ndarray
        Path to a .npy file or a .npy file containing the source array to downsample.
    block_size : tuple[int, ...]
        Block size per dimension; length must equal the array ndim.
        Example: (1, 4, 4) keeps the first axis (e.g., slices) and downsamples
        spatial axes by 4× using max pooling.
    save : bool, default True
        If True, write the downsampled array to save_path.
    save_path : str | None
        Output .npy path. Required when save is True.

    Returns
    -------
    np.ndarray
        The downsampled array.

    Notes
    -----
    Uses np.max within each block (preserves foreground in binary/label volumes).
    Change func in block_reduce for different aggregation (e.g., np.mean).
    """
    if isinstance(array, str):
        #Load array
        array = np.load(array)
    else:
        array = array
    
    #Downsample
    downsampled_array = block_reduce(array, block_size=block_size, func=np.max)
    
    #Save downsampled volume
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, downsampled_array)
        print(f"Array successfully downsampled and saved as {save_path}.")
    
    return downsampled_array

def enlarge(array:str, binary_structure:tuple[int,int]|None=None, iterations:int=1, save:bool=True, save_path:str=None, to_uint8:bool=True) -> np.ndarray:
    """
    Morphologically enlarge (dilate) a binary or label volume.

    Parameters
    ----------
    array : str | np.ndarray
        Path to a .npy file or a .npy file containing the source array. Expected shape (Z, Y, X) or (Y, X).
    binary_structure : tuple[int, int] | None, default None
        (rank, connectivity) passed to scipy.ndimage.generate_binary_structure.
        If None, uses a 2D (rank=2, connectivity=2) structuring element broadcast along Z.
        For true 3D dilation, pass (3, 3).
    iterations : int, default 1
        Number of dilation iterations. Each iteration expands boundaries once.
    save : bool, default True
        If True, writes the enlarged volume to save_path.
    save_path : str | None
        Output .npy path. Required when save is True.
    to_uint8 : bool, default True
        Convert boolean result to uint8 (0/255). If False, return bool array.

    Returns
    -------
    np.ndarray
        Enlarged volume (uint8 with {0,255} if to_uint8 else bool).

    Notes
    -----
    - Input is binarized (values > 0 become True).
    - Default structure applies 2D dilation independently per slice (no inter-slice connectivity).
    - For 3D connectivity (link slices), supply structure_params=(3, 1 or 2).
    - Dilation can dramatically grow thin structures; tune iterations.
    """
    if isinstance(array, str):
        #Load array
        array = np.load(array)
    else:
        array = array
    
    #Structuring element
    if binary_structure is None:
        base = generate_binary_structure(2, 2)
        structure = base[None, :, :]
    else:
        structure = generate_binary_structure(binary_structure[0], binary_structure[1])
    
    #Dilate/enlarge
    array_enlarged = binary_dilation(array, structure=structure, iterations=iterations)
    
    #Convert to uint8 if specified
    array_enlarged = array_enlarged.astype(np.uint8) * 255 if to_uint8 else array_enlarged
    
    #Save enlarged volume
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, array_enlarged)
        print(f"Array successfully enlarged and saved as {save_path}.")
    
    return array_enlarged

def filter_labels(img:str|np.ndarray, labels_to_keep:list[int], save:bool=True, save_path:str=None) -> np.ndarray:
    """
    Keep only specified label IDs in a labeled mask; set all other pixels to 0.

    Parameters
    ----------
    img : str | np.ndarray
        Image, or path to the input mask/image file (typically single-channel label image).
    labels_to_keep : list[int]
        Label values to retain. Pixels whose value is not in this list are set to 0.
    save : bool, default True
        If True, writes the filtered mask to save_path using cv2.imwrite.
    save_path : str | None
        Destination path for the filtered image. Required when save=True.

    Returns
    -------
    np.ndarray
        Filtered mask with the same shape and dtype as the input (in memory). Pixels not in
        labels_to_keep are zeroed.

    Notes
    -----
    - Expects a single-channel label mask. If the source is multi-channel, convert to a single-channel
        label image before using this function.
    - When saving, ensure the dtype is supported by cv2.imwrite (e.g., uint8 or uint16); otherwise,
        cast accordingly before saving.
    """
    if isinstance(img, str):
        #Read img/mask
        mask = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    else:
        mask=img
    
    #Filter mask
    mask_filtered = np.where(np.isin(mask, labels_to_keep), mask, 0)
    
    #Save mask
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        cv2.imwrite(save_path, mask_filtered)
        print(f"Filtered mask successfully saved as {save_path}.")
        
    return mask_filtered
        
def generate_mask(volume:np.ndarray, dilation_radius:int=25, min_hole_size:int=100, save:bool=True, save_path:str=None) -> np.ndarray:
    """
    Generate an enclosing binary ROI mask from a labeled/segmented 3D volume,
    **computed independently per Z-slice (2D per section)**.

    This is intended for cases where you want a per-section nerve-ring/ROI envelope and
    do *not* want morphology to connect structures across adjacent slices.

    Processing steps (per z-slice)
    ------------------------------
    1) Binarize: foreground = (volume[z] > 0).
    2) 2D morphological closing with a disk structuring element of radius `dilation_radius`.
       (bridges small gaps and smooths boundaries in-plane only)
    3) Fill small holes using `skimage.morphology.remove_small_holes` with threshold
       `min_hole_size` (in pixels; per-slice).
    4) Fill remaining enclosed holes using `scipy.ndimage.binary_fill_holes` (per-slice).
    5) Convert boolean mask to uint8 (0/255). Optionally save as a .npy file.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (typically shape (Z, Y, X)). Any value > 0 is treated as foreground.
    dilation_radius : int, default 25
        Radius (in pixels) of the 2D disk structuring element used for closing (per slice).
    min_hole_size : int, default 100
        Minimum hole size (in pixels) to fill during the `remove_small_holes` step (per slice).
    save : bool, default True
        If True and `save_path` is provided, saves the resulting mask as a .npy file.
    save_path : str | None, default None
        Output path for saving the mask when `save` is True.

    Returns
    -------
    np.ndarray
        A uint8 mask with the same shape as `volume`, where background is 0 and ROI is 255.

    Notes
    -----
    - This function intentionally does not use 3D structuring elements (e.g., `ball`).
    - If `volume` contains no foreground (all zeros), the returned mask will be all zeros.
    """
    # Binarize once (bool saves memory and is what ndimage expects)
    vol_bool = (volume > 0)

    # 2D structuring element (per-slice)
    struct_elem_2d = disk(dilation_radius)

    out = np.zeros_like(vol_bool, dtype=bool)

    for z in range(vol_bool.shape[0]):
        sl = vol_bool[z]

        # 2D closing (no inter-slice connectivity)
        sl = binary_closing(sl, structure=struct_elem_2d)

        # Fill holes per-slice
        sl = remove_small_holes(sl, area_threshold=min_hole_size)
        sl = binary_fill_holes(sl)

        out[z] = sl

    final_mask = (out.astype(np.uint8) * 255)

    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, final_mask)
        print(f"Volume successfully saved as {save_path}.")

    return final_mask
        
def json_to_volume(json_path:str, volume_shape:tuple[int,int,int], voxel_size:tuple[int,int,int], point_value:int=255, save:bool=True, save_path:str=None) -> np.ndarray:
    """
    Create a sparse 3D volume from 3D point coordinates stored in a JSON file.

    Parameters
    ----------
    json_path : str
        Path to a JSON file containing points. Expected format is a mapping where
        each value is a 3-element list [x, y, z] (same physical units as voxel_size).
        The file is read with pandas.read_json(..., orient='index').
    volume_shape : tuple[int, int, int]
        Output array shape as (Z, Y, X).
    voxel_size : tuple[int, int, int]
        Physical voxel size (dz, dy, dx). Each coordinate is divided by the corresponding
        voxel size and floored (//) to obtain integer indices.
    point_value : int, default 255
        Value to assign at each point voxel.
    save : bool, default True
        If True, saves the resulting array to save_path as a .npy file.
    save_path : str | None
        Destination .npy path. Required when save=True.

    Returns
    -------
    np.ndarray
        A uint8 volume of shape (Z, Y, X) with zeros everywhere except at point
        locations set to point_value.

    Notes
    -----
    - Axis/order: indices are computed as (z_idx, y_idx, x_idx) from input [x, y, z].
    - Points falling outside volume_shape are skipped.
    - Multiple points mapping to the same voxel simply set the same value.
    - Uses floor division; if rounding is preferred, pre-process coordinates accordingly.
    """
    #Read json
    points = pd.read_json(json_path, orient='index')
    points.rename(columns={0: "points"}, inplace=True)
    
    #Create point volume
    point_vol = np.zeros(volume_shape, dtype=np.uint8)  # Z, Y, X
    
    #Assign a value in the volume for each point
    count = 0
    for point in points['points']:
        z_idx = int(point[2] // voxel_size[0])
        y_idx = int(point[1] // voxel_size[1])
        x_idx = int(point[0] // voxel_size[2])
        if z_idx < volume_shape[0] and y_idx < volume_shape[1] and x_idx < volume_shape[2]:
            point_vol[z_idx, y_idx, x_idx] = point_value
            count += 1
    #Save point volume
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, point_vol)
        print(f"Volume successfully saved as {save_path}.")

    print(f'# of Points assigned to volume: {count}/{len(points["points"])}')
    return point_vol

def move_points_to_junctions(preds:str|np.ndarray, points:str|np.ndarray, max_distance:int=35, three:bool=False, save:bool=True, save_path:str=None) -> tuple[np.ndarray, int, int]:
    """Snap point annotations onto the nearest predicted junction voxel.

    This function takes a binary/label prediction volume (gap junctions) and a sparse point
    volume (annotations) and relocates each point to the nearest predicted foreground voxel,
    provided the nearest-foreground distance is below ``max_distance`` (in voxels). The
    output is a new sparse uint8 volume containing only the relocated points (255).

    Two modes are supported via ``three``:
    - ``three=True``: a full 3D Euclidean distance transform is computed once over the entire
      volume and points may move across slices.
    - ``three=False`` (default): distance transforms are computed per Z-slice (2D), so points
      only move within their original slice.

    Parameters
    ----------
    preds : str | np.ndarray
        Predicted junction volume or path to a ``.npy`` file. Any value ``> 0`` is treated as
        junction foreground.
    points : str | np.ndarray
        Point annotation volume or path to a ``.npy`` file. Any value ``> 0`` is treated as a
        point.
    max_distance : int, default 35
        Maximum allowed Euclidean distance (in voxels) from a point to its nearest predicted
        junction voxel. Points farther than this threshold are dropped (not written to output).
    three : bool, default False
        If True, compute a single 3D distance transform; if False, compute per-slice 2D
        distance transforms.
    save : bool, default True
        If True and ``save_path`` is provided, saves the moved point volume to disk as ``.npy``.
    save_path : str | None, default None
        Destination path for saving when ``save=True``.

    Returns
    -------
    tuple[np.ndarray, int, int]
        moved_points : np.ndarray
            A uint8 volume (same shape as inputs) containing relocated points with value 255.
        total_points : int
            Number of original point voxels detected.
        total_moved_points : int
            Number of original points that were within ``max_distance`` (before de-duplication).

    Notes
    -----
    - Inputs must be 3D arrays and are expected to have the same shape (no explicit shape check).
    - Multiple points can map to the same junction voxel; the output stores a single 255 at that
      location.
    - ``distance_transform_edt(..., return_indices=True)`` can be extremely memory-intensive,
      especially in ``three=True`` mode.
    """
    #Load points and predictions (140GB RAM for 700x10000x10000 volume)
    points = np.load(points).astype(np.uint8) if isinstance(points, str) else points #Should already be uint8
    preds = np.load(preds).astype(np.uint8) if isinstance(preds, str) else preds
    
    #Check that both inputs are 3 dimensional
    if points.ndim != 3 or preds.ndim != 3:
        raise ValueError(f"Both points and preds must be 3D arrays. Got points.ndim={points.ndim}, preds.ndim={preds.ndim}")
    
    #Convert to boolean masks (140GB RAM)
    points_bool = (points > 0)
    preds_bool = (preds > 0)
    
    if three:
        #Compute distance transform from predicted gap junctions (280GB + 210GB RAM)
        distance, nearest_indices = distance_transform_edt(~preds_bool, return_indices=True)

        #Transform points from 3D array to a list of points
        points_list = np.argwhere(points_bool) #shape: (N_points, 3)

        #Refine points_list to only include points with distance < 70 voxels to nearest gap junction entity
        points_list_filtered = points_list[distance[points_list[:,0], points_list[:,1], points_list[:,2]] < max_distance]
        total_points = len(points_list)
        total_moved_points = len(points_list_filtered)

        #For each point find its nearest predicted gap junction
        #nearest_indices has shape (3, D, H, W) with [z, y, x] indices at each voxel
        nearest_gap_junctions_list = nearest_indices[:, points_list_filtered[:,0], points_list_filtered[:,1], points_list_filtered[:,2]].T
        #nearest_gap_junctions now has shape (N_points, 3)

        #Create an array of moved points (70GB RAM)
        moved_points = np.zeros_like(points, dtype=np.uint8)
        moved_points[nearest_gap_junctions_list[:,0], nearest_gap_junctions_list[:,1], nearest_gap_junctions_list[:,2]] = 255

        #Report statistics
        distances_moved = distance[points_list_filtered[:,0], points_list_filtered[:,1], points_list_filtered[:,2]]
        #distances_moved has shape (N, 1)
        print(f"Moved {total_moved_points} points to nearest blobs")
        print(f"Mean distance moved: {distances_moved.mean():.2f} voxels")
        print(f"Max distance moved: {distances_moved.max():.2f} voxels")
        
    else:
        #2D CALCULATIONS
        moved_points = np.zeros_like(points, dtype=np.uint8)
        total_points = 0
        total_moved_points = 0
        distance_list = []
        for i in range(preds.shape[0]):
            #Take image slice from volume
            img_pred = preds_bool[i,:,:]
            img_points = points_bool[i,:,:]
            
            #Compute distance transform from predicted gap junctions
            distance, nearest_indices = distance_transform_edt(~img_pred, return_indices=True)
            
            #Transform points from 2D array to a list of points
            points_list = np.argwhere(img_points) #shape: (N_points, 2)
            
            if points_list.size == 0:
                continue
            
            #Refine points_list to only include points with distance < 70 voxels to nearest gap junction entity
            points_list_filtered = points_list[distance[points_list[:,0], points_list[:,1]] < max_distance]
            num_points = len(points_list)
            num_moved_points = len(points_list_filtered)
            
            if num_moved_points == 0:
                total_points += num_points
                continue
            
            #For each point find its nearest predicted gap junction
            #nearest_indices has shape (2, D, H, W) with [z, y, x] indices at each voxel
            nearest_gap_junctions_list = nearest_indices[:, points_list_filtered[:,0], points_list_filtered[:,1]].T
            
            #Update the moved points volume and metrics
            moved_points[i, nearest_gap_junctions_list[:,0], nearest_gap_junctions_list[:,1]] = 255
            total_points += num_points
            total_moved_points += num_moved_points
            
            #Report statistics
            distances_moved = distance[points_list_filtered[:,0], points_list_filtered[:,1]]
            #distances_moved has shape (N, 1)
            distance_list.append(distances_moved)
        
        #Print statistics
        if distance_list:
            dist_all = np.concatenate(distance_list, axis=0)
            print(f"Moved {total_moved_points} points to nearest blobs")
            print(f"Mean distance moved: {dist_all.mean():.2f} voxels")
            print(f"Max distance moved: {dist_all.max():.2f} voxels")
        else:
            print(f"Moved {total_moved_points} points to nearest blobs")
            print("No points were within max_distance; no distance statistics available.")

    #Save moved points
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, moved_points)
        print(f"Moved points saved as {save_path}.")    
        
    return moved_points, total_points, total_moved_points

def stack_slices(slice_dir:str, multi_label:bool=True, save:bool=False, save_path:str=None, file_extension:str=".png") -> np.ndarray:
    """
    Load 2D image slices from a directory and stack them into a 3D volume.

    Parameters
    ----------
    slice_dir : str
        Directory containing per-slice image files (e.g. z-slices).
    multi_label : bool, default True
        If True, load images with cv2.IMREAD_UNCHANGED (preserve multi-channel or multi-class labels).
        If False, load as single-channel grayscale.
    save : bool, default False
        If True, save the stacked volume to save_path as .npy.
    save_path : str | None
        Output .npy file path. Required when save=True.
    file_extension : str, default ".png"
        File extension filter for slice selection.

    Returns
    -------
    np.ndarray
        3D volume with shape (Z, H, W) or (Z, H, W, C) depending on source images.

    Notes
    -----
    - Slices are sorted lexicographically; use zero-padded filenames to ensure correct z-order.
    - All slices must share identical spatial dimensions (and channels if multi-label).
    - Missing imports: ensure `import os` and `import cv2` are present at top of file.
    - `output_path` is not used; pass `save_path` when save=True.
    """
    #Load slices
    img_dir = Path(slice_dir)
    img_paths = sorted([p for p in os.listdir(img_dir) if p.endswith(file_extension)])
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(str(img_dir / img_path), cv2.IMREAD_UNCHANGED if multi_label else cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

    #Stack slices along Z
    volume = np.stack(imgs, axis=0)
    
    #Save stacked volume
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, volume)
        print(f"Volume successfully saved as {save_path}.")
        
    print(f"Volume successfully stacked from slices in {slice_dir}.")
    return volume

def transform_points_to_nearby_entities(preds:str|np.ndarray, points:str|np.ndarray, radius:int=10, dust_size:int=6, save:bool=True, save_path:str=None) -> tuple[np.ndarray, int]:
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
    preds : str | np.ndarray 
        Path to .npy file or actual .npy of the predicted binary volume.
    points : str | np.ndarray
        Path to .npy file or actual .npy of the point-annotation volume.
    radius : int, default=15
        Neighborhood radius in voxels in the downsampled grid
    (after the 2× downsampling in X and Y). Adjust accordingly if you need a radius
    in the original resolution.
    dust_size : int, default=6
        Minimum size of connected components to keep in preds.

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
    points = np.load(points).astype(np.uint8) if isinstance(points, str) else points
    preds = np.load(preds).astype(np.uint8) if isinstance(preds, str) else preds

    #Convert preds to entity array
    preds_filtered = cc3d.dust(preds, threshold=dust_size, connectivity=26, in_place=False)
    entities, num_entities = cc3d.connected_components(preds_filtered, connectivity=26, return_N=True, out_dtype=np.uint32)
    print("Entity array computed.")

    #Convert points to boolean array
    points_bool = (points > 0)

    #Distance transform to distance array using inverse (bitwise NOT) of points mask
    dist_to_points = distance_transform_edt(~points_bool).astype(np.float32)
    print("Distance transform to points computed.")

    #Create boolean mask of voxels within the specified radius of any point
    near_points = dist_to_points <= radius

    #Keep only entities that overlap with near_points
    #Step 1: Get intersection of near_points and entity voxels (using boolean AND) and keep only those labels
    keep_labels = np.unique(entities[near_points & (entities > 0)])
    
    #Step 2: Filter entity array to keep only labels found in Step 1
    filtered_entity_array = np.where(np.isin(entities, keep_labels), entities, 0)
    print("Filtered entity array computed.")
    
    #Save
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, filtered_entity_array)
        print(f"Filtered entity array saved as {save_path}.")
        
    return filtered_entity_array, num_entities

def upsample(array:str|np.ndarray, scale_factors:tuple[int,...], save:bool=True, save_path:str=None) -> np.ndarray:
    """
    Upsample a 3D NumPy array by integer factors using nearest-neighbor replication.

    Parameters
    ----------
    array : str | np.ndarray
        Path to a .npy file or actual .npy containing the source volume to upsample.
    scale_factors : tuple[int, ...]
        Integer scale per axis (z, y, x). Length must equal the array ndim (typically 3).
        Example: (2, 4, 4) doubles slices and quadruples in-plane resolution.
    save : bool, default True
        If True, writes the upsampled array to save_path as .npy.
    save_path : str | None
        Output path for the .npy file. Required when save is True.

    Returns
    -------
    np.ndarray
        Upsampled array with shape (D*s0, H*s1, W*s2) and the same dtype as input.

    Notes
    -----
    - Implemented via numpy.repeat along each axis (nearest-neighbor). Best for masks/labels.
    - For continuous images, consider interpolation-based methods (e.g., scipy.ndimage.zoom,
        skimage.transform.resize) to avoid blocky artifacts.
    - Assumes integer scale factors >= 1; no validation is performed here.
    """
    #Load array
    array = np.load(array) if isinstance(array, str) else array
    
    #Upsample
    upsampled_volume = np.repeat(np.repeat(np.repeat(array, scale_factors[0], axis=0), scale_factors[1], axis=1), scale_factors[2], axis=2)
    
    #Save downsampled volume
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, upsampled_volume)
        print(f"Object successfully upsampled and saved as {save_path}.")
        
    return upsampled_volume

def volume_to_slices(volume:str|np.ndarray, output_dir:str) -> None:
    """
    Save each z-slice of a 3D NumPy volume (.npy) as a PNG image.

    Parameters
    ----------
    volume : str | np.ndarray
        Path to a .npy file or actual .npy containing a 3D array with shape (Z, Y, X).
    output_dir : str
        Directory where PNG slices will be written. The directory is created
        and cleared (existing contents removed) before saving.

    Behavior
    --------
    - The volume is binarized: values > 0 become 255; others become 0, and
      the result is saved as uint8 PNGs.
    - Slices are named with zero-padded indices: slice_000.png, slice_001.png, ...
    - I/O errors (e.g., unreadable file, unwritable directory) propagate.

    Returns
    -------
    None
    """
    #Load volume
    volume = np.load(volume) if isinstance(volume, str) else volume
    
    #Convert to boolean mask
    volume[volume > 0] = 255
    volume = volume.astype(np.uint8)
    
    #Ensure output directory is empty
    check_output_directory(Path(output_dir), clear=True)
    
    #Save every slice in volume
    for i in range(volume.shape[0]):
        slice = volume[i,:,:]
        cv2.imwrite(f"{output_dir}/slice_{i:03d}.png", slice)
        
    print(f"Finished slicing volume into {i+1} slices.")
    print(f"Slices saved to {output_dir}")
    
if __name__ == "__main__":
    start = time()
    
    # #Task 3: Filter neuron segmentation mask by neuron-only labels in SEM_adult
    # #Read neuron labels
    # df = pd.read_csv("/home/tommy111/projects/def-mzhen/tommy111/neuron_ids_no_muscles.csv")
    # sem_adult_neuron_ids = df[df['adult']>0]['adult'].tolist()
    
    # #Clear output directory if it exists
    # check_output_directory("/home/tommy111/scratch/Neurons/SEM_adult_filtered/", clear=True)
    
    # #Filter neuron mask by labels in sem adult
    # data = Path("/home/tommy111/scratch/Neurons/SEM_adult")
    # for img in data.glob("*.png"):
    #     img_read = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    #     filter_labels(img_read, sem_adult_neuron_ids, save=True, save_path=f"/home/tommy111/scratch/Neurons/SEM_adult_filtered/{str(img.name)}")
    # print("Task 3 finished.")
    
    # #Task 4: Generate a more accurate neuron mask by using neuron IDs
    # #Stack into volume
    # data = Path("/home/tommy111/scratch/Neurons/SEM_adult_filtered/")
    # vol = np.stack([cv2.imread(str(img), cv2.IMREAD_UNCHANGED) for img in data.glob("*.png")], axis=0)
    # vol[vol > 0] = 255
    # #Downsample
    # downsample(vol, block_size=(1, 4, 4), save_path="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_block_downsampled4x.npy")
    # print("Task 4 finished.")
    
    # #Task 5: Generate the final neuron mask by binary_closing and hole filling
    # neuron_volume = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_block_downsampled4x.npy")
    # nr_volume = generate_mask(neuron_volume, dilation_radius=15, save_path="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_NRmask2_block_downsampled4x.npy")
    # downsample(nr_volume, block_size=(1,2,2), save_path="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_NRmask2_block_downsampled8x.npy")
    # print("Task 5 finished.")
    
    #Task 6: Constrain predictions to within the neuron mask and calculate entity metrics
    # nr_volume = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_NRmask2_block_downsampled4x.npy")
    # preds = np.load("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled4x.npy").astype(bool)
    # nr_preds = preds & nr_volume
    # np.save("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_constrainedNR2_block_downsampled4x.npy", nr_preds.astype(np.uint8))
    
    #Need to move points again but only in x and y, so will recalculate points volume.
    print("Calculating entity metrics from points moved only in x & y.")
    move_points_to_junctions(points="/home/tommy111/scratch/outputs/sem_adult_GJ_points_downsampled4x.npy",
                             preds="/home/tommy111/scratch/sem_adult_GJs_entities_downsampled4x.npy",
                             save=True,
                             save_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled4x.npy")
    calculate_entity_metrics(preds="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled4x.npy",
                             points="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled4x.npy",
                             nerve_ring_mask="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_NRmask2_block_downsampled4x.npy")
    print("Task 6 finished.")
    
    # #Task 7: Generate full-sized images for NR mask and constrained predictions and 
    # #1. Get volume
    # vol = np.load("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_constrainedNR2_block_downsampled4x.npy")
    # #2. Upsample by 4x in each dimension except z
    # vol_upsampled = upsample(vol, scale_factors=(1,4,4), save=False)
    # #3. Volume to slices
    # volume_to_slices(volume=vol_upsampled, output_dir="/home/tommy111/scratch/split_volumes/sem_adult_NR_predictions/")
    # #4. Transfer to DL computer and upload to VAST
    # #5. Export as vsseg file
    
    # #Task 2: Generate dilated GJ points as sections for SEM_adult
    # point_volume = json_to_volume(json_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_GJs.json",
    #                volume_shape=(700, 11008, 19968),
    #                voxel_size=(30, 4, 4),
    #                point_value=255,
    #                save=False)
    
    # downsample(point_volume, block_size=(1,4,4), save_path="/home/tommy111/scratch/outputs/sem_adult_GJ_points_downsampled4x.npy")
    
    # moved_points, num_points, num_moved_points = move_points_to_junctions(preds="/home/tommy111/scratch/sem_adult_GJs_entities_downsampled4x.npy",
    #                                                                       points="/home/tommy111/scratch/outputs/sem_adult_GJ_points_downsampled4x.npy",
    #                                                                       save=False)
    # print(f"Total original points: {num_points}, Moved points: {num_moved_points}")
    # moved_points_upsampled = upsample(moved_points, scale_factors=(1,4,4), save=False)
    
    # enlarged_point_volume = enlarge(moved_points_upsampled, iterations=5, save=False)
    
    # volume_to_slices(volume=enlarged_point_volume, output_dir="/home/tommy111/scratch/split_volumes/sem_adult_gj_points")
    
    # #Task 0: Calculate Entity metrics for GJs constrained in nerve ring
    # #SEM_adult
    # neuron_volume = stack_slices(slice_dir="/home/tommy111/scratch/Neurons/SEM_adult")
    # neuron_volume_downsampled = downsample(neuron_volume, block_size=(1,4,4), save=False)
    # neuron_volume_downsampled[neuron_volume_downsampled > 0] = 255
    # neuron_mask = neuron_volume_downsampled.astype(np.uint8)
    # neuron_mask_enlarged = enlarge(neuron_mask, iterations=15, save=False)
    # neuron_mask_enlarged_downsampled = downsample(neuron_mask_enlarged, block_size=(1,2,2), save=False)
    # calculate_entity_metrics(preds="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled8x.npy",
    #                         points="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled8x.npy",
    #                         nerve_ring_mask=neuron_mask_enlarged_downsampled)
    
    # #Task 1: Get Chemical Synapses for Erin
    # #Convert json to volume
    # cs_volume = json_to_volume(json_path="/home/tommy111/projects/def-mzhen/tommy111/cs_point_annotations/sem_adult_CSs.json",
    #                volume_shape=(700, 11008, 19968),
    #                voxel_size=(30, 4, 4),
    #                point_value=255,
    #                save=False)
    # #Downsample by 4x for SEM Adult
    # downsample(cs_volume, block_size=(1,8,8), save_path="/home/tommy111/projects/def-mzhen/tommy111/cs_point_annotations/sem_adult_CSs_block_downsampled8x.npy")
    # del cs_volume
    
    # #Get unfiltered neuron masks as volumes
    # #Adult
    # neuron_adult = stack_slices(slice_dir="/home/tommy111/scratch/Neurons/SEM_adult", save=False)
    # neuron_adult[neuron_adult > 0] = 255
    # downsample(neuron_adult, block_size=(1,8,8), save_path="/home/tommy111/scratch/Neurons/SEM_adult_neurons_unfiltered_block_downsampled8x.npy")
    # del neuron_adult
    
    # #Dauer 1
    # neuron_dauer1 = stack_slices(slice_dir="/home/tommy111/scratch/Neurons/SEM_dauer_1", save=False)
    # neuron_dauer1[neuron_dauer1 > 0] = 255
    # downsample(neuron_dauer1, block_size=(1,4,4), save_path="/home/tommy111/scratch/Neurons/SEM_dauer_1_neurons_unfiltered_block_downsampled4x.npy")
    # del neuron_dauer1
    
    # #Dauer 2
    # neuron_dauer2 = stack_slices(slice_dir="/home/tommy111/scratch/Neurons/SEM_dauer_2", save=False)
    # neuron_dauer2[neuron_dauer2 > 0] = 255
    # downsample(neuron_dauer2, block_size=(1,4,4), save_path="/home/tommy111/scratch/Neurons/SEM_dauer_2_neurons_unfiltered_block_downsampled4x.npy")
    
    end = time()
    
    print(f"Job finished in {(end-start)/60:.2f} minutes")