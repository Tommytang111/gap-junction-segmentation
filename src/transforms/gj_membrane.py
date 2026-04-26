#Libraries
from turtle import pd

import numpy as np
import cv2
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pickle
import sys
sys.path.append("/home/tommy111/projects/def-mzhen/tommy111/code/")
from src.utils import check_output_directory
from src.transforms.transform_objects import volume_to_slices, upsample
from scipy.ndimage import sobel

#FUNCTIONS 
def extract_membranes(neurons:np.ndarray|str, radius:int=1) -> np.ndarray:
    """
    Extract membrane voxels by dilation and XOR.

    Generates a thin membrane ring around neuron regions by dilating each
    neuron mask and taking the logical XOR between the original mask and
    the dilated mask. Supports both 2D images and 3D volumes. For 3D
    inputs, processing is done independently on each Z-slice (2D slice-wise).

    Parameters
    ----------
    neurons : np.ndarray or str
        2D or 3D neuron mask array with shape (Y, X) or (Z, Y, X).
        If a str filepath is provided, the array is loaded with np.load().
        Non-zero values are treated as neuron foreground.
    radius : int, optional
        Radius of the disk-shaped structuring element used for dilation.
        Larger values produce thicker membrane rings. Default is 1.

    Returns
    -------
    np.ndarray
        uint8 membrane mask with the same shape as `neurons`, where 255
        indicates membrane voxels and 0 indicates non-membrane.

    Raises
    ------
    ValueError
        If `neurons` is not a 2D or 3D array.

    Examples
    --------
    >>> membrane = extract_membranes(neurons=neuron_mask_2d)

    >>> membrane = extract_membranes(
    ...     neurons="/path/to/neuron_volume.npy",
    ...     radius=3
    ... )
    """
    from skimage.morphology import disk
    from scipy.ndimage import binary_dilation
    
    #Load neuron mask
    neurons = np.load(neurons) if isinstance(neurons, str) else neurons
    neurons_bin = neurons.astype(bool)

    se = disk(radius)

    if neurons_bin.ndim == 2:
        dilated = binary_dilation(neurons_bin, structure=se)
        membrane = np.logical_xor(neurons_bin, dilated)
        return membrane.astype(np.uint8) * 255

    if neurons_bin.ndim == 3:
        membrane = np.zeros_like(neurons_bin, dtype=np.uint8)
        for z in range(neurons_bin.shape[0]):
            dilated = binary_dilation(neurons_bin[z], structure=se)
            membrane[z] = np.logical_xor(neurons_bin[z], dilated).astype(np.uint8) * 255
        return membrane

    raise ValueError(f"Expected 2D or 3D array, got shape {neurons.shape}")

def extract_membranes_with_gradient(neurons: np.ndarray | str, kernel_size: int = 3, threshold: float = 0, save:bool=False, save_path:str=None) -> np.ndarray:
    """
    Extract neuron membrane voxels using Sobel gradient edge detection.

    Applies Sobel filters along X and Y axes to detect boundaries between
    neuron regions. Pixels where the gradient magnitude exceeds `threshold`
    are classified as membrane. Custom kernel sizes can be defined. For 3D 
    volumes, the Sobel filter is applied independently to each Z-slice 
    (slice-wise 2D processing).

    Parameters
    ----------
    neurons : np.ndarray or str
        2D or 3D integer-labeled neuron volume (Y, X) or (Z, Y, X).
        If a str filepath is provided, the array is loaded with np.load().
    kernel_size : int
        Size of the kernel to use.
        Must be an odd integer.
        Default is 3.
    threshold : float, optional
        Minimum gradient magnitude to be classified as membrane.
        Default is 0, meaning any non-zero gradient is classified as membrane.
    save : bool, optional
        If True, saves the resulting membrane mask to disk at `save_path`.
        Default is False.
    save_path : str, optional
        File path where the membrane mask will be saved as a .npy file.
        Only used if `save` is True. Parent directory is created if it
        does not exist. Default is None.

    Returns
    -------
    np.ndarray
        uint8 membrane mask of the same shape as `neurons`, where 255
        indicates a membrane voxel and 0 indicates non-membrane.

    Examples
    --------
    >>> membrane = extract_membranes_with_gradient(neurons=neuron_volume)

    >>> membrane = extract_membranes_with_gradient(
    ...     neurons=neuron_volume,
    ...     threshold=10.0,
    ...     kernel_size=5,
    ...     save=True,
    ...     save_path="outputs/membranes/membrane.npy"
    ... )
    """
    
    neurons = np.load(neurons) if isinstance(neurons, str) else neurons
    arr = neurons.astype(float)

    def _membrane_2d(img2d: np.ndarray, k_size:int) -> np.ndarray:
        if k_size == 3:
            sx = sobel(img2d, axis=0)
            sy = sobel(img2d, axis=1)
            
        elif kernel_size % 2 == 1:
            img = neurons.astype(np.float64)
            sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        else:
            raise ValueError("Kernel size must be odd.")
            
        grad = np.hypot(sx, sy)
        return (grad > threshold).astype(np.uint8) * 255

    if arr.ndim == 2:
        final_membrane = _membrane_2d(arr, k_size=kernel_size)

    if arr.ndim == 3:
        membrane = np.zeros_like(arr, dtype=np.uint8)
        for z in range(arr.shape[0]):
            membrane[z] = _membrane_2d(arr[z], k_size=kernel_size)
        final_membrane = membrane
        
    #Optionally save membrane 
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, final_membrane)
        print(f"Membrane saved as {save_path}.")
        
    return final_membrane

def expand_neurons_to_membrane(neuron_labels: np.ndarray | str, membrane_mask: np.ndarray | str, dilation_factor: int = 1, max_iterations: int = 100, save:bool=False, save_path:str=None) -> np.ndarray:
    """
    Expand labeled neuron regions until they reach and overlap 1 pixel into the membrane boundary.
    Each neuron expands locally - different edges can expand at different rates until they hit membrane.
    
    Parameters:
    -----------
    neuron_labels : np.ndarray or str
        Labeled mask where each neuron has a unique integer label (shape: Z, Y, X)
    membrane_mask : np.ndarray or str
        Binary membrane mask (shape: Z, Y, X), where membrane pixels are non-zero
    max_iterations : int
        Maximum number of dilation iterations to prevent infinite loops
    
    Returns:
    --------
    np.ndarray
        Expanded neuron labels mask with same shape as input
    """
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk
    
    # Load data if paths are given
    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    membrane_mask = np.load(membrane_mask) if isinstance(membrane_mask, str) else membrane_mask
    
    # Create output array
    expanded_labels = neuron_labels.copy()
    membrane_binary = (membrane_mask > 0) #Should already be binarized anyways
    
    # Structuring element for dilation
    se = disk(dilation_factor)
    
    # Process each z-slice independently
    slices = expanded_labels.shape[0]
    for z in tqdm(range(slices), total=slices, desc="Expanding neurons to membrane"):
        neuron_slice = expanded_labels[z]
        membrane_slice = membrane_binary[z]
        
        # Get all unique neuron labels in this slice (excluding background)
        unique_labels = np.unique(neuron_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Track which pixels have reached membrane for each neuron
        for label in unique_labels:
            # Get mask for this specific neuron
            neuron_mask = (neuron_slice == label).astype(bool)
            
            # Track pixels that have reached the membrane (stop expanding from these)
            reached_membrane = np.zeros_like(neuron_mask, dtype=bool)
            
            # Iteratively expand
            for iteration in range(max_iterations):
                # Dilate only from pixels that haven't reached membrane yet
                active_mask = neuron_mask & ~reached_membrane
                
                if not np.any(active_mask):
                    break  # All edges have reached membrane
                
                # Dilate the active region
                dilated = binary_dilation(active_mask, structure=se)
                
                # Find new pixels added by dilation
                new_pixels = dilated & ~neuron_mask
                
                # Only expand into background (not other neurons)
                can_expand = new_pixels & (neuron_slice == 0)
                
                if not np.any(can_expand):
                    break  # No more room to expand
                
                # Check which new pixels hit the membrane
                new_membrane_pixels = can_expand & membrane_slice
                
                # Add expandable pixels to the neuron
                neuron_slice[can_expand] = label
                neuron_mask = (neuron_slice == label)
                
                # Mark pixels that touched membrane (and their 1-pixel overlap) as "reached"
                if np.any(new_membrane_pixels):
                    # Dilate the membrane-touching pixels by 1 to get the overlap region
                    membrane_region = binary_dilation(new_membrane_pixels, structure=disk(1))
                    reached_membrane |= membrane_region & neuron_mask
        
        expanded_labels[z] = neuron_slice
        
    #Optionally save expanded labels 
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, expanded_labels)
        print(f"Expanded labels saved as {save_path}.")
    
    return expanded_labels

def expand_neurons_to_membrane2(neuron_labels: np.ndarray | str, membrane_mask: np.ndarray | str, max_iterations: int = 100, penetration: int = 1, connectivity: int = 8, save: bool = False, save_path: str = None) -> np.ndarray:
    """
    Expand labeled neurons simultaneously into background, with optional membrane penetration.

    This version avoids per-label order bias by growing all neuron labels at the same time.
    A background pixel is assigned from neighboring labels each iteration.
    If multiple labels compete for the same pixel, the smallest label id is chosen
    deterministically.

    Parameters
    ----------
    neuron_labels : np.ndarray or str
        Integer labeled mask (Z, Y, X), 0 is background.
    membrane_mask : np.ndarray or str
        Binary membrane mask (Z, Y, X), non-zero means membrane.
    max_iterations : int, optional
        Maximum number of simultaneous growth steps per slice.
    penetration : int, optional
        Allowed depth (in voxels) into membrane regions. 0 means no membrane entry.
    connectivity : int, optional
        Neighborhood for growth: 4 or 8.
    save : bool, optional
        If True, save output to `save_path`.
    save_path : str, optional
        Output .npy path.

    Returns
    -------
    np.ndarray
        Expanded neuron labels with same shape as input.
        
    Disclaimers
    -------
    This function was written by GPT-5.3
    """
    from scipy.ndimage import distance_transform_edt

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if penetration < 0:
        raise ValueError("penetration must be >= 0")

    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    membrane_mask = np.load(membrane_mask) if isinstance(membrane_mask, str) else membrane_mask

    expanded_labels = neuron_labels.copy().astype(np.int32, copy=False)
    membrane_binary = membrane_mask > 0

    def _grow_once(labels_slice: np.ndarray, allowed: np.ndarray, conn: int) -> int:
        """
        One simultaneous growth step.
        Returns number of newly assigned pixels.
        """
        unlabeled = (labels_slice == 0) & allowed
        if not np.any(unlabeled):
            return 0

        proposed = np.zeros_like(labels_slice, dtype=np.int32)

        def propose_from_neighbor(src_rows, src_cols, dst_rows, dst_cols):
            src_vals = labels_slice[src_rows, src_cols]
            dst_mask = unlabeled[dst_rows, dst_cols] & (src_vals > 0)
            if not np.any(dst_mask):
                return
            dst_r = dst_rows[dst_mask]
            dst_c = dst_cols[dst_mask]
            src_v = src_vals[dst_mask]
            cur = proposed[dst_r, dst_c]
            proposed[dst_r, dst_c] = np.where(cur == 0, src_v, np.minimum(cur, src_v))
            
        H, W = labels_slice.shape

        # 4-neighborhood
        propose_from_neighbor(np.s_[1:H, :],   np.s_[:], np.s_[0:H-1, :], np.s_[:])   # from down to up
        propose_from_neighbor(np.s_[0:H-1, :], np.s_[:], np.s_[1:H, :],   np.s_[:])   # from up to down
        propose_from_neighbor(np.s_[:, 1:W],   np.s_[:], np.s_[:, 0:W-1], np.s_[:])   # from right to left
        propose_from_neighbor(np.s_[:, 0:W-1], np.s_[:], np.s_[:, 1:W],   np.s_[:])   # from left to right

        if conn == 8:
            # diagonals
            propose_from_neighbor(np.s_[1:H, 1:W],   np.s_[:], np.s_[0:H-1, 0:W-1], np.s_[:])
            propose_from_neighbor(np.s_[0:H-1, 0:W-1], np.s_[:], np.s_[1:H, 1:W],   np.s_[:])
            propose_from_neighbor(np.s_[1:H, 0:W-1], np.s_[:], np.s_[0:H-1, 1:W],   np.s_[:])
            propose_from_neighbor(np.s_[0:H-1, 1:W], np.s_[:], np.s_[1:H, 0:W-1],   np.s_[:])

        new_pixels = unlabeled & (proposed > 0)
        n_new = int(np.count_nonzero(new_pixels))
        if n_new > 0:
            labels_slice[new_pixels] = proposed[new_pixels]
        return n_new

    for z in tqdm(range(expanded_labels.shape[0]), total=expanded_labels.shape[0], desc="Expanding neurons to membrane"):
        labels_slice = expanded_labels[z]
        membrane_slice = membrane_binary[z]

        # Allow expansion everywhere non-membrane, and optionally into membrane up to `penetration` depth.
        if penetration == 0:
            allowed = ~membrane_slice
        else:
            # Depth inside membrane (distance to nearest non-membrane pixel)
            mem_depth = distance_transform_edt(membrane_slice)
            allowed_membrane = membrane_slice & (mem_depth <= float(penetration))
            allowed = (~membrane_slice) | allowed_membrane
            
        # Never overwrite already-labeled pixels
        allowed = allowed & (labels_slice == 0)

        for _ in range(max_iterations):
            n_new = _grow_once(labels_slice, allowed, connectivity)
            if n_new == 0:
                break

        expanded_labels[z] = labels_slice

    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        np.save(save_path, expanded_labels)
        print(f"Expanded labels saved as {save_path}.")

    return expanded_labels


def analyze_gj_per_neuron(neuron_membrane_mask: np.ndarray, neuron_labels: np.ndarray, gj_segmentation: np.ndarray, save:bool=True, save_path:str=None) -> dict:
    """
    Analyze gap junction proteins on each neuron's membrane slice by slice.
    
    Parameters:
    -----------
    neuron_membrane_mask : np.ndarray
        Binary mask of neuron membranes (shape: Z, Y, X)
    neuron_labels : np.ndarray
        Mask with different labels for each neuron (shape: Z, Y, X)
    gj_segmentation : np.ndarray
        Segmentation prediction mask for gap junction proteins (shape: Z, Y, X)
    
    Returns:
    --------
    dict
        Dictionary where keys are neuron labels and values are dicts containing:
        - 'total_voxels': total number of membrane voxels for that neuron
        - 'gj_voxels': number of gap junction voxels on that neuron's membrane
        - 'gj_fraction': fraction of gap junction to total voxels
    """
    #Load data if paths are given
    neuron_membrane_mask = np.load(neuron_membrane_mask) if isinstance(neuron_membrane_mask, str) else neuron_membrane_mask
    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    gj_segmentation = np.load(gj_segmentation) if isinstance(gj_segmentation, str) else gj_segmentation
    
    # Initialize results dictionary
    results = {}
    
    # Process slice by slice along z-axis (axis 0)
    for z in tqdm(range(neuron_membrane_mask.shape[0]), total=neuron_membrane_mask.shape[0], desc="Analyzing neuron gjs per slice"):
        # Get current slice for all masks
        membrane_slice = neuron_membrane_mask[z]
        labels_slice = neuron_labels[z]
        gj_slice = gj_segmentation[z]
        
        # Get intersection: membrane pixels with neuron labels
        # This gives us the neuron identity at each membrane pixel
        neuron_membrane_intersection = labels_slice * (membrane_slice > 0)
        
        # Get gap junction on membranes
        gj_on_membrane = gj_slice * (membrane_slice > 0)
        
        # Find unique neuron labels in this slice (excluding 0/background)
        unique_labels = np.unique(neuron_membrane_intersection)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Count voxels for each neuron
        for label in unique_labels:
            # Get mask for this neuron's membrane in this slice
            neuron_mask = (neuron_membrane_intersection == label)
            
            # Count total membrane voxels for this neuron
            total_voxels = np.sum(neuron_mask)
            
            # Count gap junction voxels on this neuron's membrane
            gj_voxels = np.sum((gj_on_membrane > 0) & neuron_mask)
            
            # Update results
            if label not in results:
                results[label] = {
                    'total_voxels': 0,
                    'gj_voxels': 0,
                    'gj_fraction': 0.0
                }
            
            results[label]['total_voxels'] += total_voxels
            results[label]['gj_voxels'] += gj_voxels
    
    #Calculate fractions for each neuron
    for label in results:
        total = results[label]['total_voxels']
        gj = results[label]['gj_voxels']
        results[label]['gj_fraction'] = gj / total if total > 0 else 0.0
    
    #Optionally save analysis results 
    if save and save_path is not None:
        check_output_directory(Path(save_path).parent, clear=False)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Neuronal analysis results saved as {save_path}.")

    return results

def get_electrical_connectivity(neuron_membrane_mask: np.ndarray | str, neuron_labels: np.ndarray | str, gj_segmentation: np.ndarray | str, *, contact_connectivity: int = 8, save:bool=True, **save_paths:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate:
      1) Contactome connectivity between neuron pairs:
         total number of membrane-adjacency "contact voxels" between touching neuron pairs.
         (Computed from label adjacencies restricted to membrane pixels.)

      2) Gap junction voxel connectivity between neuron pairs:
         number of border adjacency pairs where at least one pixel is a GJ pixel.
         Uses the same adjacency logic as the contactome, so GJ ≤ contactome always.

      3) Normalized GJ connectivity:
         GJ connectivity / contactome (fraction), elementwise.

    Parameters
    ----------
    neuron_membrane_mask : np.ndarray or str
        Binary/uint8 membrane mask (Z, Y, X). Non-zero means membrane.
    neuron_labels : np.ndarray or str
        Integer neuron instance labels (Z, Y, X). 0 is background.
    gj_segmentation : np.ndarray or str
        Binary/uint8 GJ prediction (Z, Y, X). Non-zero means GJ.
    contact_connectivity : int
        4 or 8. If 8, diagonal adjacencies also contribute to contactome.

    Returns
    -------
    (contactome_df, gj_df, normalized_df) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    import pandas as pd

    if contact_connectivity not in (4, 8):
        raise ValueError(f"contact_connectivity must be 4 or 8, got {contact_connectivity}")

    # Load data if paths are given
    neuron_membrane_mask = np.load(neuron_membrane_mask) if isinstance(neuron_membrane_mask, str) else neuron_membrane_mask
    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    gj_segmentation = np.load(gj_segmentation) if isinstance(gj_segmentation, str) else gj_segmentation

    # All neuron ids
    all_neuron_labels = np.unique(neuron_labels)
    all_neuron_labels = all_neuron_labels[all_neuron_labels > 0].astype(int)

    #Initialize connectivity matrix
    contactome_matrix = pd.DataFrame(0, index=all_neuron_labels, columns=all_neuron_labels, dtype=np.int32)
    gj_connectivity_matrix = pd.DataFrame(0, index=all_neuron_labels, columns=all_neuron_labels, dtype=np.int32)

    def _accumulate_undirected_pairs(mat: pd.DataFrame, a: np.ndarray, b: np.ndarray):
        """
        Given two same-length arrays a,b of neuron ids for adjacent pixels,
        accumulate counts into mat symmetrically for undirected pairs.
        """
        if a.size == 0:
            return

        lo = np.minimum(a, b).astype(int, copy=False)
        hi = np.maximum(a, b).astype(int, copy=False)

        pairs = np.stack([lo, hi], axis=1)
        uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        for (i, j), c in zip(uniq_pairs, counts):
            # i != j always by construction of adjacency mask, but keep safe
            if i == j:
                continue
            mat.at[i, j] += int(c)
            mat.at[j, i] += int(c)

    z_slices = neuron_membrane_mask.shape[0]
    for z in tqdm(range(z_slices), total=z_slices, desc="Calculating electrical connectivity"):
        #Get current slice for all masks
        membrane_slice = neuron_membrane_mask[z] > 0
        labels_slice = neuron_labels[z].astype(np.int32, copy=False)
        gj_slice = gj_segmentation[z] > 0

        # Neuron id at membrane locations, else 0
        L = labels_slice * membrane_slice

        # Contactome (adjacency-based)
        # Count label disagreements across neighbor pairs, restricted to membrane pixels (both sides must be membrane-labeled)
        # Horizontal neighbors
        left = L[:, :-1]  # All pixels except the rightmost column
        right = L[:, 1:]  # All pixels except the leftmost column
        
        #Get all pixels where two different neurons are touching
        m = (left > 0) & (right > 0) & (left != right)
        _accumulate_undirected_pairs(contactome_matrix, left[m], right[m])

        # Vertical neighbors
        up = L[:-1, :]
        down = L[1:, :]
        m = (up > 0) & (down > 0) & (up != down)
        _accumulate_undirected_pairs(contactome_matrix, up[m], down[m])

        if contact_connectivity == 8:
            # Diagonal down-right
            ul = L[:-1, :-1]
            dr = L[1:, 1:]
            m = (ul > 0) & (dr > 0) & (ul != dr)
            _accumulate_undirected_pairs(contactome_matrix, ul[m], dr[m])

            # Diagonal down-left
            ur = L[:-1, 1:]
            dl = L[1:, :-1]
            m = (ur > 0) & (dl > 0) & (ur != dl)
            _accumulate_undirected_pairs(contactome_matrix, ur[m], dl[m])

        # Gap junction connectivity (adjacency-based, same logic as contactome)
        # A GJ contact pair: two adjacent membrane pixels belonging to different neurons,
        # where at least one of the two pixels is a GJ pixel.
        # This ensures GJ counts are a strict subset of contactome counts.
        if np.any(gj_slice):
            G = gj_slice  # boolean GJ mask for this slice

            # Horizontal
            m = (left > 0) & (right > 0) & (left != right) & (G[:, :-1] | G[:, 1:])
            _accumulate_undirected_pairs(gj_connectivity_matrix, left[m], right[m])

            # Vertical
            m = (up > 0) & (down > 0) & (up != down) & (G[:-1, :] | G[1:, :])
            _accumulate_undirected_pairs(gj_connectivity_matrix, up[m], down[m])

            if contact_connectivity == 8:
                # Diagonal down-right
                m = (ul > 0) & (dr > 0) & (ul != dr) & (G[:-1, :-1] | G[1:, 1:])
                _accumulate_undirected_pairs(gj_connectivity_matrix, ul[m], dr[m])

                # Diagonal down-left
                m = (ur > 0) & (dl > 0) & (ur != dl) & (G[:-1, 1:] | G[1:, :-1])
                _accumulate_undirected_pairs(gj_connectivity_matrix, ur[m], dl[m])

    # Normalized GJ = GJ / contactome (elementwise), 0 where no contact
    normalized_gj_matrix = gj_connectivity_matrix.div(contactome_matrix.where(contactome_matrix > 0), fill_value=0.0).astype(float)
    
    #Optionally save analysis results
    if save and save_paths:
        if "contactome" in save_paths:
            check_output_directory(Path(save_paths["contactome"]).parent, clear=False)
            with open(save_paths["contactome"], "wb") as f:
                pickle.dump(contactome_matrix, f)
            print(f"Contactome saved as {save_paths['contactome']}.")
        if "gj_connectivity" in save_paths:
            check_output_directory(Path(save_paths["gj_connectivity"]).parent, clear=False)
            with open(save_paths["gj_connectivity"], "wb") as f:
                pickle.dump(gj_connectivity_matrix, f)
            print(f"GJ connectivity saved as {save_paths['gj_connectivity']}.")
        if "normalized_gj_connectivity" in save_paths:
            check_output_directory(Path(save_paths["normalized_gj_connectivity"]).parent, clear=False)
            with open(save_paths["normalized_gj_connectivity"], "wb") as f:
                pickle.dump(normalized_gj_matrix, f)
            print(f"Normalized GJ connectivity saved as {save_paths['normalized_gj_connectivity']}.")
        
    return contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix

def calculate_gj_relative_intensity(
    em_volume: np.ndarray, 
    neuron_membranes: np.ndarray, 
    neuron_labels: np.ndarray, 
    unique_entities: np.ndarray, 
    radius: int = 100
) -> pd.DataFrame:
    """
    Compute per-entity gap junction (GJ) relative intensity against local non-GJ membrane.

    For each non-zero `connector_ID` in `unique_entities`, the function:
      1) computes the mean intensity of the darkest 50% of voxels belonging to that entity,
      2) finds the two neurons most overlapping the entity (via `neuron_label volume`),
      3) collects membrane voxels within a 3D spherical neighborhood of radius `radius`
         centered at the entity centroid, restricted to:
           - membrane voxels (`membrane_volume > 0`)
           - non-GJ voxels (`entity_id_volume == 0`)
           - belonging to either of the two neurons
      4) computes relative intensity = (darkest50_mean / surrounding_membrane_mean),
      5) returns a table of results.

    Parameters
    ----------
    em_volume : np.ndarray
        3D raw image intensity volume with shape (Z, Y, X). Has to be uint8 in [0, 255]
    neuron_membranes : np.ndarray
        3D membrane mask with shape (Z, Y, X). Non-zero values indicate membrane voxels.
    neuron_labels : np.ndarray
        3D integer neuron label volume with shape (Z, Y, X) where labels have been expanded
        onto membranes. Used to infer which two neurons the GJ entity belongs to based on
        spatial overlap.
    unique_entities : np.ndarray
        3D volume with shape (Z, Y, X) containing per-voxel GJ entity IDs / connector IDs.
        Background must be 0; each GJ entity should have a positive integer ID.
    radius : int, optional
        Radius (in voxels) of the spherical neighborhood used to sample surrounding membrane.
        Default is 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per `connector_ID` and columns:
          - 'connector_ID' : int
          - 'average_intensity' : float
                Relative intensity (darkest50_mean / surrounding_membrane_mean).
                NaN if no valid surrounding membrane is found or if the denominator is 0.
          - 'neuron_1' : int
          - 'neuron_2' : int
        If an entity overlaps only one neuron label, 'neuron_2' is set to 0.

    Notes
    -----
    * The spherical neighborhood is implemented via a bounding box crop plus a distance mask,
      to avoid computing distances over the full volume.
    * If multiple neuron labels overlap the GJ entity, the two labels with the largest overlap
      counts are selected as (neuron_1, neuron_2).
    * This function assumes all inputs share the same (Z, Y, X) shape.

    Examples
    --------
    >>> df = calculate_gj_relative_intensity(
    ...     em_volume=raw,
    ...     neuron_membranes=membrane,
    ...     neuron_labels=expanded_neurons,
    ...     unique_entities=connector_ids,
    ...     radius=100,
    ... )
    >>> df.head()
    """
    if em_volume.dtype != np.uint8:
        raise ValueError(f"Expected em_volume to be uint8, got {em_volume.dtype}")
    
    results = []
    
    #Get all unique gap junction entities (ignore background 0)
    unique_ids = np.unique(unique_entities)
    unique_ids = unique_ids[unique_ids > 0]
    
    Z, Y, X = em_volume.shape
    
    for connector_id in unique_ids:
        #1. Get GJ voxels and average intensity of top 50% darkest voxels
        gj_coords = np.argwhere(unique_entities == connector_id)
        if len(gj_coords) == 0:
            continue
            
        gj_intensities = em_volume[unique_entities == connector_id]
        sorted_intensities = np.sort(gj_intensities)
        half_thresh = max(1, len(sorted_intensities) // 2)
        darkest_50_avg = np.mean(sorted_intensities[:half_thresh])
        
        #Determine the two unique neurons this entity belongs to
        overlapping_neurons = neuron_labels[unique_entities == connector_id]
        unique_neurons, counts = np.unique(overlapping_neurons[overlapping_neurons > 0], return_counts=True)
        
        if len(unique_neurons) >= 2:
            top_2_idx = np.argsort(counts)[-2:]
            neuron1 = unique_neurons[top_2_idx[1]]
            neuron2 = unique_neurons[top_2_idx[0]]
        elif len(unique_neurons) == 1:
            neuron1 = unique_neurons[0]
            neuron2 = 0
        else:
            neuron1, neuron2 = 0, 0
            
        #2. Generate a sphere around the entity's centroid
        z_c, y_c, x_c = np.mean(gj_coords, axis=0).astype(int)
        
        #Calculate bounding box bounds to avoid full-volume operations
        z_min, z_max = max(0, z_c - radius), min(Z, z_c + radius + 1)
        y_min, y_max = max(0, y_c - radius), min(Y, y_c + radius + 1)
        x_min, x_max = max(0, x_c - radius), min(X, x_c + radius + 1)
        
        local_membrane = neuron_membranes[z_min:z_max, y_min:y_max, x_min:x_max]
        local_neurons = neuron_labels[z_min:z_max, y_min:y_max, x_min:x_max]
        local_entities = unique_entities[z_min:z_max, y_min:y_max, x_min:x_max]
        local_img = em_volume[z_min:z_max, y_min:y_max, x_min:x_max]
        
        #Distance calculation for the sphere mask
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist_sq = (zz - z_c)**2 + (yy - y_c)**2 + (xx - x_c)**2
        local_sphere_mask = dist_sq <= radius**2
        
        #3. Filter for surrounding membrane voxels based on criteria
        valid_membrane_mask = (
            local_sphere_mask & 
            (local_membrane > 0) & 
            (local_entities == 0) & 
            ((local_neurons == neuron1) | (local_neurons == neuron2))
        )
        
        surrounding_membrane_intensities = local_img[valid_membrane_mask]
        
        if len(surrounding_membrane_intensities) > 0:
            membrane_avg_intensity = np.mean(surrounding_membrane_intensities)
        else:
            membrane_avg_intensity = np.nan
            
        #4. Calculate GJ entity relative intensity
        if not np.isnan(membrane_avg_intensity) and membrane_avg_intensity != 0:
            relative_intensity = darkest_50_avg / membrane_avg_intensity
        else:
            relative_intensity = np.nan
            
        #5. Append values for the dictionary output
        results.append({
            'connector_ID': connector_id,
            'average_intensity': relative_intensity,
            'neuron_1': neuron1,
            'neuron_2': neuron2
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__": 
    start = time.time()
    
    print("Running gap junction analysis for sem dauer 2... \n")
    
    # #Load data
    # neurons = np.load("/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_block_downsampled4x.npy")
    # membrane = np.load("/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy")
    
    # #Task 1: Extract membrane
    # # membrane = extract_membranes(neurons, radius=5)
    # # np.save("/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy", membrane)
    # membrane_upsampled = upsample(membrane, (1,4,4), save=False)
    # volume_to_slices(membrane_upsampled, "/home/tommy111/scratch/split_volumes/sem_dauer_2_neuron_membrane")
    
    # #Task 2: Expand neurons to membrane
    # expanded_neurons = expand_neurons_to_membrane(neuron_labels=neurons, membrane_mask=membrane, dilation_factor=2)
    # np.save("/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_not_uniform_expanded_downsampled4x_dilation2.npy", expanded_neurons)
    # expanded_neurons_upsampled = upsample(expanded_neurons, (1,4,4), save=False)
    # volume_to_slices(expanded_neurons_upsampled, "/home/tommy111/scratch/split_volumes/sem_dauer_2_neurons_only_with_labels_non_uniform_expanded_dilation2")
    
    gjs = np.load("/home/tommy111/projects/def-mzhen/tommy111/em_objects/gj_point_annotations/sem_dauer_2/sem_dauer_2_high_confidence_NR_entities_downsampled4x.npy")
    gjs[gjs>0] = 255
    gjs = gjs.astype(np.uint8)
    
    #Task 3: Calculate gap junctions per neuron and write output
    neuronal_gj_dict = analyze_gj_per_neuron(neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy", 
                                             neuron_labels="/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_not_uniform_expanded_downsampled4x.npy", 
                                             gj_segmentation=gjs)
            
    import pickle
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_neuronal_hc_gj_analysis_h1qrqboc.pkl", "wb") as f:
        pickle.dump(neuronal_gj_dict, f)
    
    #Task 4: Calculate electrical connectivity matrix 
    
    #MODEL h1qrqboc
    contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix = get_electrical_connectivity(
        neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy", 
        neuron_labels="/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_not_uniform_expanded_downsampled4x.npy", 
        gj_segmentation=gjs
    )
    
    #Write out to pickle
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_contactome_h1qrqboc.pkl", "wb") as f:
        pickle.dump(contactome_matrix, f)
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_neuronal_hc_gj_connectivity_h1qrqboc.pkl", "wb") as f:
        pickle.dump(gj_connectivity_matrix, f)
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_normalized_hc_gj_connectivity_h1qrqboc.pkl", "wb") as f:
        pickle.dump(normalized_gj_matrix, f)
        
        
    gjs = np.load("/home/tommy111/projects/def-mzhen/tommy111/em_objects/gj_point_annotations/sem_dauer_2/sem_dauer_2_low_confidence_NR_entities_downsampled4x.npy")
    gjs[gjs>0] = 255
    gjs = gjs.astype(np.uint8)
    
    #Task 3: Calculate gap junctions per neuron and write output
    neuronal_gj_dict = analyze_gj_per_neuron(neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy", 
                                             neuron_labels="/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_not_uniform_expanded_downsampled4x.npy", 
                                             gj_segmentation=gjs)
            
    import pickle
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_neuronal_lc_gj_analysis_h1qrqboc.pkl", "wb") as f:
        pickle.dump(neuronal_gj_dict, f)
    
    #Task 4: Calculate electrical connectivity matrix 
    
    #MODEL h1qrqboc
    contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix = get_electrical_connectivity(
        neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_dauer_2/SEM_dauer_2_neuron_membrane_downsampled4x.npy", 
        neuron_labels="/home/tommy111/scratch/Neurons/SEM_dauer_2/SEM_dauer_2_neurons_only_with_labels_not_uniform_expanded_downsampled4x.npy", 
        gj_segmentation=gjs
    )
    
    #Write out to pickle
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_contactome_h1qrqboc.pkl", "wb") as f:
        pickle.dump(contactome_matrix, f)
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_neuronal_lc_gj_connectivity_h1qrqboc.pkl", "wb") as f:
        pickle.dump(gj_connectivity_matrix, f)
    with open("/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_dauer_2/SEM_dauer_2_normalized_lc_gj_connectivity_h1qrqboc.pkl", "wb") as f:
        pickle.dump(normalized_gj_matrix, f)
    
    end = time.time()
    print("Job completed.")
    print(f"Total runtime: {(end - start)/60:.2f} minutes")