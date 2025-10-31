#Convert 3D volumes to individual 2D slices and save as PNG images
#October 20, 2025
#Tommy Tang

#Libraries
import numpy as np
import cv2
import sys
#sys.path.append('/home/tommy111/projects/def-mzhen/tommy111/code')
from src.utils import _ensure_empty_dir

def main():
    #Load volume
    volume1 = np.load('/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs_entities_upsampled.npy')
    volume2 = np.load('/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs_entities_upsampled.npy')

    #Convert volume1 from boolean mask to uint8 for visualization
    #volume1 = (volume1.astype(np.uint8)) * 255
    volume1[volume1 > 0] = 255
    volume2[volume2 > 0] = 255
    volume1 = volume1.astype(np.uint8)
    volume2 = volume2.astype(np.uint8)

    #Ensure output directories are empty
    _ensure_empty_dir('/home/tommy111/scratch/split_volumes/sem_dauer_1_GJs_entities')
    _ensure_empty_dir('/home/tommy111/scratch/split_volumes/sem_dauer_2_GJs_entities')

    for i in range(volume1.shape[0]):
        volume1_slice = volume1[i, :, :]
        volume2_slice = volume2[i, :, :]
        cv2.imwrite(f'/home/tommy111/scratch/split_volumes/sem_dauer_1_GJs_entities/slice_{i:03d}.png', volume1_slice)
        cv2.imwrite(f'/home/tommy111/scratch/split_volumes/sem_dauer_2_GJs_entities/slice_{i:03d}.png', volume2_slice)

    print("Done slicing volumes.")
    print("Number of slices created (Dauer 1):", volume1.shape[0])
    print("Number of slices created (Dauer 2):", volume2.shape[0])
    print('Slices saved to /home/tommy111/scratch/split_volumes/')
    
if __name__ == "__main__":
    main()
