from scipy.ndimage import binary_dilation, generate_binary_structure
import numpy as np
import time

start=time.time()

#Import GJ volume
#points1 = np.load("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs.npy").astype(np.uint8)
points2 = np.load("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs.npy").astype(np.uint8)

#2D square with thickness 1 along Z
structure_xy = generate_binary_structure(2, 2)[None, :, :]

#Enlarge GJ points 
#points1_enlarged = binary_dilation(points1, structure=structure_xy, iterations=14)
points2_enlarged = binary_dilation(points2, structure=structure_xy, iterations=10)

#Save enlarged GJ points
#np.save("/home/tommy111/scratch/sem_dauer_1_GJs_enlarged.npy", points1_enlarged)
np.save("/home/tommy111/scratch/sem_dauer_2_GJs_enlarged.npy", points2_enlarged)

end = time.time()

print(f'Script completed in {(end - start):.2f} seconds.')
