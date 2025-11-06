#Downsample multiple volumes by the same factor using block_reduce from skimage
#October 13, 2025
#Tommy Tang

from skimage.measure import block_reduce
import numpy as np

#Define volumes of interest
pred_volume_path = "/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume.npy"
point_volume_path = "/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_GJs.npy"

#Load volumes
pred_volume = np.load(pred_volume_path)
point_volume = np.load(point_volume_path)

downsampled_pred_volume = block_reduce(pred_volume, block_size=(1, 2, 2), func=np.max)
downsampled_point_volume = block_reduce(point_volume, block_size=(1, 2, 2), func=np.max)

#Save the downsampled volumes
np.save("/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled2x.npy", downsampled_pred_volume)
np.save("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_GJs_downsampled2x.npy", downsampled_point_volume)

#Completion messages
print("Dataset: Sem Adult")
print(f"Original volume # of GJs: {np.sum(pred_volume > 0)}")
print(f"Original volume # of GJ points: {np.sum(point_volume > 0)}")
print(f"Downsampled volume # of GJs: {np.sum(downsampled_pred_volume > 0)}")
print(f"Downsampled volume # of GJ points: {np.sum(downsampled_point_volume > 0)}")
print("Volumes successfully downsampled and saved.")