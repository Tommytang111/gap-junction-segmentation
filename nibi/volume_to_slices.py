#Import Libraries
import numpy as np
import cv2

#Load volume
volume1 = np.load('/home/tommy111/scratch/sem_dauer_2_GJs_enlarged.npy')
#volume2 = np.load('/home/tommy111/scratch/sem_dauer_2_GJs_entities.npy')

#Convert volume1 from boolean mask to uint8 for visualization
volume1 = (volume1.astype(np.uint8)) * 255
#volume2[volume2 > 0] = 255
#volume2 = volume2.astype(np.uint8)

for i in range(volume1.shape[0]):
    volume1_slice = volume1[i, :, :]
    #volume2_slice = volume2[i, :, :]
    cv2.imwrite(f'/home/tommy111/scratch/split_volumes/sem_dauer_2_GJs_enlarged/slice_{i:03d}.png', volume1_slice)
    #cv2.imwrite(f'/home/tommy111/scratch/split_volumes/sem_dauer_2_GJs_entities/slice_{i:03d}.png', volume2_slice)

print("Done slicing volumes.")
print("Number of slices created:", volume1.shape[0])
print('Slices saved to /home/tommy111/scratch/split_volumes/')
