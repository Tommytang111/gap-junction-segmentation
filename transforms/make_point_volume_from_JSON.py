import pandas as pd
import numpy as np
import cv2 

#Read data as json
gj_points = pd.read_json("/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_GJs.json", orient='index')
gj_points.rename(columns={0: "points"}, inplace=True)

#Create point volume
point_vol = np.zeros((700, 11008, 19968), dtype=np.uint8)  # Z, Y, X

#Assign a value for each point in the volume
for gj in gj_points['points']:
    if (gj[2]//30) < 700:
        point_vol[int(gj[2])//30, int(gj[1])//4, int(gj[0])//4] = 255 
    
#Save the point volume
volume_path = "/home/tommy111/projects/def-mzhen/tommy111/gj_point_annotations/sem_adult_GJs.npy"
print(f"Saving volume to {volume_path}")
np.save(volume_path, point_vol)