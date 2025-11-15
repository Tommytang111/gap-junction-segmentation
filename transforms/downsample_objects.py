#Downsample multiple volumes by the same factor using block_reduce from skimage
#October 13, 2025
#Tommy Tang

from skimage.measure import block_reduce
import numpy as np

def downsample(volume_path, block_size, save=True, save_path=None):
    #Load volume
    volume = np.load(volume_path)
    
    #Downsample
    downsampled_volume = block_reduce(volume, block_size=block_size, func=np.max)
    
    #Save downsampled volume
    if save:
        np.save(save_path, downsampled_volume)
        
    print(f"Volume successfully downsampled and saved in {save_path}.")

if __name__ == "__main__":  
    downsample(volume_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled4x.npy",
               block_size=(1,2,2),
               save=True,
               save_path="/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_moved_GJs_downsampled8x.npy")
    
    
    
    
    # downsample(volume_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_2_s000-972/volume.npy",
    #            block_size=(1,4,4),
    #            save=True,
    #            save_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_2_s000-972/volume_block_downsampled4x.npy")
    # downsample(volume_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_1_s000-850/volume.npy",
    #            block_size=(1,4,4),
    #            save=True,
    #            save_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_1_s000-850/volume_block_downsampled4x.npy")
    # downsample(volume_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled4x.npy",
    #            block_size=(1,2,2),
    #            save=True,
    #            save_path="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_block_downsampled8x.npy")

#Completion messages
#print("Dataset: Sem Adult")
# print(f"Original volume # of GJs: {np.sum(pred_volume > 0)}")
# print(f"Original volume # of GJ points: {np.sum(point_volume > 0)}")
# print(f"Downsampled volume # of GJs: {np.sum(downsampled_pred_volume > 0)}")
# print(f"Downsampled volume # of GJ points: {np.sum(downsampled_point_volume > 0)}")
#print("Volumes successfully downsampled and saved.")