#Upsample volumes using np.kron
#October 28, 2025
#Tommy Tang

import numpy as np

def main():
    #Define volumes of interest
    volume_path = "/home/tommy111/scratch/sem_dauer_1_GJs_entities.npy"
    volume_path2 = "/home/tommy111/scratch/sem_dauer_2_GJs_entities.npy"

    #Load volumes
    point_volume = np.load(volume_path)
    point_volume2 = np.load(volume_path2)
    
    #Convert to uint8 from uint16 to save memory
    point_volume[point_volume > 0] = 255
    point_volume2[point_volume2 > 0] = 255
    point_volume = point_volume.astype(np.uint8)
    point_volume2 = point_volume2.astype(np.uint8)

    #Upsample volumes by a factor of 2 in x and y
    upsampled_volume = np.kron(point_volume, np.ones((1, 2, 2)))
    upsampled_volume2 = np.kron(point_volume2, np.ones((1, 2, 2)))

    #Save the upsampled volumes
    np.save("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_1_GJs_entities_upsampled.npy", upsampled_volume)
    np.save("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_dauer_2_GJs_entities_upsampled.npy", upsampled_volume2)

    #Completion messages
    print("Dataset: Sem Dauer 1")
    print(f"Original volume # of GJ points: {np.sum(point_volume > 0)}")
    print(f"Upsampled volume # of GJ points: {np.sum(upsampled_volume > 0)}")
    print()
    print("Dataset: Sem Dauer 2")
    print(f"Original volume # of GJ points: {np.sum(point_volume2 > 0)}")
    print(f"Upsampled volume # of GJ points: {np.sum(upsampled_volume2 > 0)}")
    print("Volumes successfully upsampled and saved in /home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/")
    
if __name__ == "__main__":
    main()