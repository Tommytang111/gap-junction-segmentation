#Extracting specific mask labels from neuron masks for Lucinda
#Tommy Tang
#Nov 15, 2025

import numpy as np
import cv2
import os
from src.utils import check_output_directory
from time import time

def extract_labels_from_masks(input_dir:str, output_dir:str, labels_to_extract:list[int]):
    """Extract and save label-filtered segmentation masks.
    For each mask image in input_dir, loads it unchanged, keeps only the pixels
    whose values are in labels_to_extract, sets all other pixels to 0, and writes
    the result to output_dir using the original filename. The output directory is
    created if it does not exist.
    This function expects single-channel, integer-labeled masks. The original dtype
    and shape are preserved.
    Args:
        input_dir (str): Path to the directory containing input mask images.
        output_dir (str): Path to the directory where filtered masks will be saved.
        labels_to_extract (list[int]): Label values to retain in the output masks.
    Returns:
        None
    Raises:
        FileNotFoundError: If input_dir does not exist.
        cv2.error: If OpenCV fails to read or write an image.
        OSError: If a file operation fails.
    Notes:
        - Files are processed in sorted order and saved with the same basenames.
        - Pixels not listed in labels_to_extract are replaced with 0.
        - Non-image files in input_dir may cause read/write errors.
    """
    check_output_directory(output_dir)
    neuron_masks = sorted(os.listdir(input_dir))
    for mask in neuron_masks:
        #Read mask
        mask_path = os.path.join(input_dir, mask)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        #Filter mask
        mask_filtered = np.where(np.isin(mask, labels_to_extract), mask, 0)
        #Save mask
        output_path = os.path.join(output_dir, os.path.basename(mask_path))
        cv2.imwrite(output_path, mask_filtered)
    
if __name__ == "__main__":
    start = time.time()
    #Tip: If more labels need to be extracted, change the code to read from a dataframe instead.
    extract_labels_from_masks(input_dir="/home/tommy111/scratch/Neurons/SEM_adult",
                              output_dir="/home/tommy111/scratch/Neurons/SEM_adult_filtered",
                              labels_to_extract=[53, 172, 182, 6, 70, 112, 27, 193, 78, 
                                                 140, 60, 113, 109, 104, 151, 42, 128, 71,
                                                 135, 134, 87, 15, 84, 51, 39, 136, 20, 5,
                                                 16, 115, 118, 94, 200, 201, 41, 34, 202, 
                                                 121, 98, 154, 80, 184, 57, 132])
    extract_labels_from_masks(input_dir="/home/tommy111/scratch/Neurons/SEM_dauer_1",
                              output_dir="/home/tommy111/scratch/Neurons/SEM_dauer_1_filtered",
                              labels_to_extract=[181,346,260,217,121,95,251,200,38,73,170,
                                                 253,240,291,233,34,287,109,63,30,265,148,
                                                 186,278,7,139,35,42,165,102,320,193,13,329,
                                                 31,204,207,50,98,68,116,194,97,44])
    extract_labels_from_masks(input_dir="/home/tommy111/scratch/Neurons/SEM_dauer_2",
                              output_dir="/home/tommy111/scratch/Neurons/SEM_dauer_2_filtered",
                              labels_to_extract=[2489,1736,705,1640,943,1918,1606,622,120,
                                                 2404,2419,845,1933,1240,361,295,1485,302,
                                                 1815,623,65,1403,2559,1975,1704,891,1051,
                                                 2032,739,201,1812,2501,1575,94,2087,786,
                                                 2650,560,608,1782,1060,11803,1641,565])
    end = time.time()
    print(f"Total time: {(end - start)/60:.2f} minutes")