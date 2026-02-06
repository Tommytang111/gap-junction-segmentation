#Libraries
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#Load data
neurons = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_block_downsampled4x.npy")
#neuron_labels = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy")
#print(np.unique(neurons, return_counts=True))
#print(np.unique(neuron_labels, return_counts=True))

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

def expand_neurons_to_membrane(neuron_labels: np.ndarray | str, 
                                membrane_mask: np.ndarray | str,
                                max_iterations: int = 100) -> np.ndarray:
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
    for z in tqdm(range(expanded_labels.shape[0]), total=expanded_labels.shape[0], desc="Expanding neurons to membrane"):
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

def analyze_gj_per_neuron(neuron_membrane_mask: np.ndarray, 
                          neuron_labels: np.ndarray, 
                          gj_segmentation: np.ndarray) -> dict:
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
    for z in range(neuron_membrane_mask.shape[0]):
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

def get_contactome_and_gj_connectivity(neuron_membrane_mask: np.ndarray,
                                       neuron_labels: np.ndarray,
                                       gj_segmentation: np.ndarray):
    """
    Calculate contactome (membrane contact voxels) and gap junction connectivity 
    between neuron pairs, then compute the ratio of GJ voxels to contact voxels.
    
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
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Three DataFrames:
        1. Contactome matrix: number of membrane contact voxels between neuron pairs
        2. GJ connectivity matrix: number of gap junction voxels between neuron pairs
        3. GJ/Contactome ratio matrix: fraction of contact voxels that are gap junctions
    """
    import pandas as pd
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk
    
    # Load data if paths are given
    neuron_membrane_mask = np.load(neuron_membrane_mask) if isinstance(neuron_membrane_mask, str) else neuron_membrane_mask
    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    gj_segmentation = np.load(gj_segmentation) if isinstance(gj_segmentation, str) else gj_segmentation
    
    # Find all unique neuron labels (excluding background)
    all_neuron_labels = np.unique(neuron_labels)
    all_neuron_labels = all_neuron_labels[all_neuron_labels > 0].astype(int)
    
    # Initialize matrices
    contactome_matrix = pd.DataFrame(0, index=all_neuron_labels, columns=all_neuron_labels, dtype=np.int32)
    gj_connectivity_matrix = pd.DataFrame(0, index=all_neuron_labels, columns=all_neuron_labels, dtype=np.int32)
    
    # Structuring element for dilation (to detect contact)
    se = disk(1)
    
    # Process slice by slice along z-axis
    for z in tqdm(range(neuron_membrane_mask.shape[0]), total=neuron_membrane_mask.shape[0], desc="Calculating contactome and GJ connectivity"):
        # Get current slice for all masks
        membrane_slice = neuron_membrane_mask[z] > 0
        labels_slice = neuron_labels[z]
        gj_slice = gj_segmentation[z] > 0
        
        # Get neuron labels at membrane locations
        neuron_membrane_intersection = labels_slice * membrane_slice
        
        # Get unique neurons in this slice
        unique_neurons = np.unique(neuron_membrane_intersection)
        unique_neurons = unique_neurons[unique_neurons > 0]
        
        # For each neuron, find contacts with other neurons
        for neuron_id in unique_neurons:
            # Get this neuron's membrane pixels
            neuron_membrane = (neuron_membrane_intersection == neuron_id)
            
            # Dilate by 1 pixel to find potential contacts
            dilated = binary_dilation(neuron_membrane, structure=se)
            
            # Find which other neurons are in the dilated region
            potential_contacts = labels_slice * dilated * membrane_slice
            
            # Get unique neighbor neurons (excluding self and background)
            neighbor_neurons = np.unique(potential_contacts)
            neighbor_neurons = neighbor_neurons[(neighbor_neurons > 0) & (neighbor_neurons != neuron_id)]
            
            # For each neighbor, count contact voxels and GJ voxels
            for neighbor_id in neighbor_neurons:
                # Get neighbor's membrane pixels
                neighbor_membrane = (neuron_membrane_intersection == neighbor_id)
                
                # Find contact region: where dilated neuron overlaps with neighbor's membrane
                contact_region = dilated & neighbor_membrane
                
                # Count contact voxels
                contact_voxels = np.sum(contact_region)
                
                # Count gap junction voxels in contact region
                gj_voxels = np.sum(contact_region & gj_slice)
                
                # Update matrices (symmetric)
                neuron_id_int = int(neuron_id)
                neighbor_id_int = int(neighbor_id)
                
                contactome_matrix.loc[neuron_id_int, neighbor_id_int] += contact_voxels
                gj_connectivity_matrix.loc[neuron_id_int, neighbor_id_int] += gj_voxels
    
    # Calculate ratio matrix
    gj_ratio_matrix = gj_connectivity_matrix.copy().astype(float)
    for i in gj_ratio_matrix.index:
        for j in gj_ratio_matrix.columns:
            contact = contactome_matrix.loc[i, j]
            if contact > 0:
                gj_ratio_matrix.loc[i, j] = gj_connectivity_matrix.loc[i, j] / contact
            else:
                gj_ratio_matrix.loc[i, j] = 0.0
    
    return contactome_matrix, gj_connectivity_matrix, gj_ratio_matrix

def get_electrical_connectivity(neuron_membrane_mask: np.ndarray,
                                neuron_labels: np.ndarray,
                                gj_segmentation: np.ndarray):
    """
    Calculate gap junction voxel connectivity between pairs of neurons using 
    connected components analysis to identify gap junction entities.
    
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
    pd.DataFrame
        Connectivity matrix as a pandas DataFrame with neuron labels as both 
        row and column indices. DataFrame.loc[i, j] contains the number of gap 
        junction voxels connecting neurons i and j. The matrix is symmetric.
    """
    import pandas as pd
    from scipy.ndimage import label as connected_components
    
    # Load data if paths are given
    neuron_membrane_mask = np.load(neuron_membrane_mask) if isinstance(neuron_membrane_mask, str) else neuron_membrane_mask
    neuron_labels = np.load(neuron_labels) if isinstance(neuron_labels, str) else neuron_labels
    gj_segmentation = np.load(gj_segmentation) if isinstance(gj_segmentation, str) else gj_segmentation
    
    # Find all unique neuron labels (excluding background)
    all_neuron_labels = np.unique(neuron_labels)
    all_neuron_labels = all_neuron_labels[all_neuron_labels > 0].astype(int)
    
    # Initialize connectivity matrix using pandas DataFrame
    connectivity_matrix = pd.DataFrame(
        0, 
        index=all_neuron_labels, 
        columns=all_neuron_labels, 
        dtype=np.int32
    )
    
    # Process slice by slice along z-axis (axis 0)
    for z in range(neuron_membrane_mask.shape[0]):
        # Get current slice for all masks
        membrane_slice = neuron_membrane_mask[z]
        labels_slice = neuron_labels[z]
        gj_slice = gj_segmentation[z]
        
        # Get gap junction on membranes (binary mask)
        gj_on_membrane = (gj_slice > 0) & (membrane_slice > 0)
        
        # Skip if no gap junctions in this slice
        if not np.any(gj_on_membrane):
            continue
        
        # Perform 2D connected components analysis on gap junction mask
        labeled_gj, num_entities = connected_components(gj_on_membrane)
        
        # Get intersection: membrane pixels with neuron labels
        neuron_membrane_intersection = labels_slice * (membrane_slice > 0)
        
        # Process each gap junction entity
        for entity_id in range(1, num_entities + 1):
            # Get mask for this entity
            entity_mask = (labeled_gj == entity_id)
            
            # Get neuron labels that this entity touches
            neuron_labels_at_entity = neuron_membrane_intersection[entity_mask]
            unique_neurons = np.unique(neuron_labels_at_entity)
            unique_neurons = unique_neurons[unique_neurons > 0]  # Exclude background
            
            # Check if entity connects exactly two different neurons
            if len(unique_neurons) == 2:
                neuron1, neuron2 = int(unique_neurons[0]), int(unique_neurons[1])
                
                # Count voxels of this entity on the membrane of both neurons
                entity_voxel_count = np.sum(entity_mask)
                
                # Add to connectivity matrix (symmetric)
                connectivity_matrix.loc[neuron1, neuron2] += entity_voxel_count
                connectivity_matrix.loc[neuron2, neuron1] += entity_voxel_count
    
    return connectivity_matrix

if __name__ == "__main__": 
    start = time.time()
    
    print("Calculating gap junctions per neuron\n")
    membrane = np.load("/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy")
    
    # #Task 1: Extract membrane
    # membrane = extract_membranes_with_gradient(neurons)
    # np.save("/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", membrane)
    
    #Task 2: Expand neurons to membrane
    expanded_neurons = expand_neurons_to_membrane(neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy",
                                                  membrane_mask=membrane)
    np.save("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", expanded_neurons)
    
    # #Task 3: Calculate gap junctions per neuron and write output
    # neuronal_gj_dict = analyze_gj_per_neuron(neuron_membrane_mask=membrane, 
    #                                          neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_not_uniform_expanded_block_downsampled4x.npy", 
    #                                          gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy")
    
    # with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_analysis_p03lmvzp.txt", "wb") as f:
    #     for neuron_label, stats in neuronal_gj_dict.items():
    #         f.write(f"Neuron {neuron_label}: Total Membrane Voxels = {stats['total_voxels']}, Gap Junction Voxels = {stats['gj_voxels']}, Gap Junction Fraction = {stats['gj_fraction']:.6f}\n".encode())
    
    # #Task 4: Calculate electrical connectivity matrix
    # connectivity_matrix = get_electrical_connectivity(neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", 
    #                                          neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy", 
    #                                          gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy")
    
    # #Write out to pickle and text
    # import pickle
    # with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_p03lmvzp.pkl", "wb") as f:
    #     pickle.dump(connectivity_matrix, f)
    
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(connectivity_matrix, annot=False, cmap='viridis', square=True)
    # plt.title('Neuronal Gap Junction Connectivity')
    # plt.xlabel('Neuron ID')
    # plt.ylabel('Neuron ID')
    # plt.show()
    # plt.savefig("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_heatmap_p03lmvzp.png", dpi=300)
            
    end = time.time()
    print("Job completed.")
    print(f"Total runtime: {(end - start)/60:.2f} minutes")