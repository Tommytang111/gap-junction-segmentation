#Script to extract objects corresponding to the nerve ring from a segmentation volume
#Tommy Tang
#November 12, 2025

#Adult needs to be further downsampled by 2x in x and y to match dauer datasets
#Do F1 just for dauers for now

import numpy as np
from pathlib import Path
from skimage.measure import block_reduce
from scipy.ndimage import binary_dilation
import cv2
import os
import time

def make_neuron_volume(neuron_path, output_path):
    #Load segmentation volume
    img_dir = Path(neuron_path)
    img_paths = sorted(os.listdir(img_dir))
    imgs = []

    for img_path in img_paths:
        img = cv2.imread(str(img_dir / img_path), cv2.IMREAD_UNCHANGED)
        imgs.append(img)

    volume = np.stack(imgs, axis=0)
    volume_downsampled = block_reduce(volume, block_size=(1,4,4), func=np.max)

    np.save(output_path, volume_downsampled)
    
def create_nerve_ring_mask(neuron_path, output_path):
    #Load neuron segmentation volume
    volume = np.load(neuron_path)
    
    #Binarize
    volume[volume > 0] = 255
    
    #Downsample again
    volume_downsampled = block_reduce(volume, block_size=(1,2,2), func=np.max)
    
    #Dilate volume (transforms to bool then back to uint8)
    volume_dilated = binary_dilation(volume_downsampled, iterations=2).astype(np.uint8)
    
    #Save nerve ring mask
    nerve_ring_mask_dilated = volume_dilated * 255

    np.save(output_path, nerve_ring_mask_dilated)
    
if __name__ == "__main__":
    start_time = time.time()
    
    #make_neuron_volume("/home/tommy111/scratch/Neurons/SEM_adult", "/home/tommy111/scratch/Neurons/SEM_adult_neurons_downsampled4x.npy")
    # make_neuron_volume("/home/tommy111/scratch/Neurons/SEM_dauer_1", "/home/tommy111/scratch/Neurons/SEM_dauer_1_neurons_downsampled2x.npy")
    # make_neuron_volume("/home/tommy111/scratch/Neurons/SEM_dauer_2", "/home/tommy111/scratch/Neurons/SEM_dauer_2_neurons_downsampled2x.npy")
    
    create_nerve_ring_mask("/home/tommy111/scratch/Neurons/SEM_adult_neurons_downsampled4x.npy", "/home/tommy111/scratch/Neurons/SEM_adult_NR_mask_downsampled8x.npy")
    create_nerve_ring_mask("/home/tommy111/scratch/Neurons/SEM_dauer_1_neurons_downsampled2x.npy", "/home/tommy111/scratch/Neurons/SEM_dauer_1_NR_mask_downsampled4x.npy")
    create_nerve_ring_mask("/home/tommy111/scratch/Neurons/SEM_dauer_2_neurons_downsampled2x.npy", "/home/tommy111/scratch/Neurons/SEM_dauer_2_NR_mask_downsampled4x.npy")
    
    end_time = time.time()
    print("Nerve ring volume complete.")
    print(f"Execution time: {(end_time - start_time):.2f} seconds")