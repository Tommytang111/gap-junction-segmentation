#Libraries
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

#Load data
neurons = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_block_downsampled4x.npy")
#neuron_labels = np.load("/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy")
#print(np.unique(neurons, return_counts=True))
#print(np.unique(neuron_labels, return_counts=True))

#Get membrane 
def extract_membranes(neurons:np.ndarray | str) -> np.ndarray:
    from scipy.ndimage import grey_erosion
    
    neurons = np.load(neurons) if isinstance(neurons, str) else neurons
    
    membrane = np.zeros_like(neurons, dtype=np.uint8)
    for z in range(neurons.shape[0]):
        eroded = grey_erosion(neurons[z], size=3)
        membrane[z] = (neurons[z] != eroded).astype(np.uint8) * 255
        
    return membrane

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
    
    # #Task 1: Extract membrane
    # membrane = extract_membranes(neurons)
    # np.save("/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", membrane)
    
    # #Task 2: Calculate gap junctions per neuron and write output
    # neuronal_gj_dict = analyze_gj_per_neuron(neuron_membrane_mask=membrane, 
    #                                          neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy", 
    #                                          gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy")
    
    #Task 3: Calculate electrical connectivity matrix
    connectivity_matrix = get_electrical_connectivity(neuron_membrane_mask="/home/tommy111/scratch/Membranes/SEM_adult_neuron_membrane_downsampled4x.npy", 
                                             neuron_labels="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy", 
                                             gj_segmentation="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy")
    
    #Write out to pickle and text
    import pickle
    with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_p03lmvzp.pkl", "wb") as f:
        pickle.dump(connectivity_matrix, f)
    # with open("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_analysis_p03lmvzp.txt", "wb") as f:
    #     for neuron_label, stats in neuronal_gj_dict.items():
    #         f.write(f"Neuron {neuron_label}: Total Membrane Voxels = {stats['total_voxels']}, Gap Junction Voxels = {stats['gj_voxels']}, Gap Junction Fraction = {stats['gj_fraction']:.6f}\n".encode())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(connectivity_matrix, annot=False, cmap='viridis', square=True)
    plt.title('Neuronal Gap Junction Connectivity')
    plt.xlabel('Neuron ID')
    plt.ylabel('Neuron ID')
    plt.show()
    plt.savefig("/home/tommy111/scratch/Membranes/SEM_adult_neuronal_gj_connectivity_heatmap_p03lmvzp.png", dpi=300)
            
    end = time.time()
    print(f"Total runtime: {(end - start)/60:.2f} minutes")