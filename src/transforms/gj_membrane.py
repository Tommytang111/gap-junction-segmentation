#Libraries
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#FUNCTIONS 
def extract_membranes(neurons: np.ndarray | str, radius: int = 1) -> np.ndarray:
    """
    Dilation-based membrane: return the 1-pixel-thick (or thin) ring created by dilation.
    Works for 2D images or 3D stacks (dilates each z-slice in 2D).
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

def extract_membranes_with_gradient(neurons: np.ndarray | str, threshold: float = 0,) -> np.ndarray:
    """Extract membrane using Sobel gradient (thin edges). Works for 2D or 3D (slice-wise)."""
    from scipy.ndimage import sobel
    import numpy as np
    
    neurons = np.load(neurons) if isinstance(neurons, str) else neurons
    arr = neurons.astype(float)

    def _membrane_2d(img2d: np.ndarray) -> np.ndarray:
        sx = sobel(img2d, axis=0)
        sy = sobel(img2d, axis=1)
        grad = np.hypot(sx, sy)
        return (grad > threshold).astype(np.uint8) * 255

    if arr.ndim == 2:
        return _membrane_2d(arr)

    if arr.ndim == 3:
        membrane = np.zeros_like(arr, dtype=np.uint8)
        for z in range(arr.shape[0]):
            membrane[z] = _membrane_2d(arr[z])
        return membrane

    raise ValueError(f"Expected 2D or 3D array, got shape {neurons.shape}")

def expand_neurons_to_membrane(neuron_labels: np.ndarray | str, membrane_mask: np.ndarray | str, max_iterations: int = 100) -> np.ndarray:
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
    membrane_binary = (membrane_mask > 0)
    
    # Structuring element for dilation
    se = disk(1)
    
    # Process each z-slice independently
    slices = expanded_labels.shape[0]
    for z in tqdm(range(slices), total=slices, desc="Expanding neurons to membrane"):
        labels_slice = expanded_labels[z]
        membrane_slice = membrane_binary[z]
        
        # Get all unique neuron labels in this slice (excluding background)
        unique_labels = np.unique(labels_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Track which pixels have reached membrane for each neuron
        for label in unique_labels:
            # Get mask for this specific neuron
            neuron_mask = (labels_slice == label).astype(bool)
            
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
                can_expand = new_pixels & (labels_slice == 0)
                
                if not np.any(can_expand):
                    break  # No more room to expand
                
                # Check which new pixels hit the membrane
                new_membrane_pixels = can_expand & membrane_slice
                
                # Add expandable pixels to the neuron
                labels_slice[can_expand] = label
                neuron_mask = (labels_slice == label)
                
                # Mark pixels that touched membrane (and their 1-pixel overlap) as "reached"
                if np.any(new_membrane_pixels):
                    # Dilate the membrane-touching pixels by 1 to get the overlap region
                    membrane_region = binary_dilation(new_membrane_pixels, structure=se)
                    reached_membrane |= membrane_region & neuron_mask
        
        expanded_labels[z] = labels_slice
    
    return expanded_labels

def analyze_gj_per_neuron(neuron_membrane_mask: np.ndarray, neuron_labels: np.ndarray, gj_segmentation: np.ndarray) -> dict:
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
    
    # Calculate fractions for each neuron
    for label in results:
        total = results[label]['total_voxels']
        gj = results[label]['gj_voxels']
        results[label]['gj_fraction'] = gj / total if total > 0 else 0.0
    
    return results

def get_electrical_connectivity(neuron_membrane_mask: np.ndarray | str, neuron_labels: np.ndarray | str, gj_segmentation: np.ndarray | str, *, contact_connectivity: int = 8,):
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
    
    return contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix

if __name__ == "__main__": 
    start = time.time()
    
    print("Calculating neuronal GJ connectivity\n")
    
    #Load data
    #neurons = np.load("/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_block_downsampled4x.npy")
    #neuron_labels = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy")
    #print(np.unique(neurons, return_counts=True))
    #print(np.unique(neuron_labels, return_counts=True))
    #membrane = np.load("/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy")
    
    # #Task 1: Extract membrane
    # membrane = extract_membranes_with_gradient(neurons)
    # np.save("/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", membrane)
    
    # #Task 2: Expand neurons to membrane
    # expanded_neurons = expand_neurons_to_membrane(neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy",
    #                                               membrane_mask=membrane)
    # np.save("/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", expanded_neurons)
    
    # #Task 3: Calculate gap junctions per neuron and write output
    # neuronal_gj_dict = analyze_gj_per_neuron(neuron_membrane_mask=membrane, 
    #                                          neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", 
    #                                          gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy")
    
    # with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_analysis_u4lqcs5g.txt", "wb") as f:
    #     for neuron_label, stats in neuronal_gj_dict.items():
    #         f.write(f"Neuron {neuron_label}: Total Membrane Voxels = {stats['total_voxels']}, Gap Junction Voxels = {stats['gj_voxels']}, Gap Junction Fraction = {stats['gj_fraction']:.6f}\n".encode())
            
    # import pickle
    # with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_analysis_u4lqcs5g.pkl", "wb") as f:
    #     pickle.dump(neuronal_gj_dict, f)
    
    #Task 4: Calculate electrical connectivity matrix 
    #MODEL p03lmvzp 
    contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix = get_electrical_connectivity(
        neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", 
        neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", 
        gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy"
    )
    
    #Write out to pickle
    import pickle
    with open("/home/tommy111/scratch/Membranes/SEM_adult_contactome_p03lmvzp.pkl", "wb") as f:
        pickle.dump(contactome_matrix, f)
    with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_p03lmvzp.pkl", "wb") as f:
        pickle.dump(gj_connectivity_matrix, f)
    with open("/home/tommy111/scratch/Membranes/SEM_adult_normalized_gj_connectivity_p03lmvzp.pkl", "wb") as f:
        pickle.dump(normalized_gj_matrix, f)
    
    # Create heatmaps for all three matrices
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    
    step = 20
    
    sns.heatmap(contactome_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[0])
    axes[0].set_title('Contactome (Shared Membrane Voxels)')
    sns.heatmap(gj_connectivity_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[1])
    axes[1].set_title('Gap Junction Connectivity')
    sns.heatmap(normalized_gj_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[2])
    axes[2].set_title('Normalized GJ Connectivity (GJ/Contactome)')
    
    plt.tight_layout()
    plt.savefig("/home/tommy111/scratch/Membranes/SEM_adult_connectivity_matrices_p03lmvzp.png", dpi=300)
    
    #MODEL u4lqcs5g
    contactome_matrix, gj_connectivity_matrix, normalized_gj_matrix = get_electrical_connectivity(
        neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", 
        neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", 
        gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy"
    )
    
    #Write out to pickle
    import pickle
    with open("/home/tommy111/scratch/Membranes/SEM_adult_contactome_u4lqcs5g.pkl", "wb") as f:
        pickle.dump(contactome_matrix, f)
    with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_u4lqcs5g.pkl", "wb") as f:
        pickle.dump(gj_connectivity_matrix, f)
    with open("/home/tommy111/scratch/Membranes/SEM_adult_normalized_gj_connectivity_u4lqcs5g.pkl", "wb") as f:
        pickle.dump(normalized_gj_matrix, f)
    
    # Create heatmaps for all three matrices
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    
    sns.heatmap(contactome_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[0])
    axes[0].set_title('Contactome (Shared Membrane Voxels)')
    sns.heatmap(gj_connectivity_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[1])
    axes[1].set_title('Gap Junction Connectivity')
    sns.heatmap(normalized_gj_matrix, annot=False, cmap='viridis', vmin=0, xticklabels=step, yticklabels=step, square=True, ax=axes[2])
    axes[2].set_title('Normalized GJ Connectivity (GJ/Contactome)')
    
    plt.tight_layout()
    plt.savefig("/home/tommy111/scratch/Membranes/SEM_adult_connectivity_matrices_u4lqcs5g.png", dpi=300)
    
    end = time.time()
    print("Job completed.")
    print(f"Total runtime: {(end - start)/60:.2f} minutes")