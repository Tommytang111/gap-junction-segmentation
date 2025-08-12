"""
Split, predict and stitch volumetric pipeline for gap junction segmentation.
Tommy Tang
June 2, 2025
"""

#Requirements:
#Full sections exported from VAST from dataset of interest

#Libraries
from utils import create_dataset_2d, assemble_imgs
from inference import inference
from models import TestDataset
import albumentations as A
import os
import cv2
import numpy as np

class GapJunctionSegmentationPipeline:
    def __init__(self, name, model_path, dataset_class, sections_dir, output_dir, pred_dir, assembled_dir, volume_dir, template, augmentations, img_size=512):
        self.name = name
        self.model = model_path
        self.dataset_class = dataset_class
        self.sections = sections_dir
        self.tiles = output_dir
        self.pred = pred_dir
        self.assembled = assembled_dir
        self.volume = volume_dir
        self.template = template
        self.img_size = img_size
        self.augmentations = augmentations
        
    def create_dataset(self):
        self.max_y, self.max_x, self.max_sections = create_dataset_2d(imgs_dir=self.sections, output_dir=self.tiles, img_size=self.img_size, create_overlap=True, test=True)

    def run_inference(self):
        inference(model_path=self.model,
                  dataset=self.dataset_class,
                  input_dir=self.tiles,
                  output_dir=self.pred,
                  augmentation=self.augmentations,
                  filter=True
                )

    def stitch_predictions(self):
        #Need to crop tiles since I created overlapping tiles
        assemble_imgs(img_dir=os.path.join(self.tiles, "imgs"),
                    gt_dir=None,
                    pred_dir=self.pred,
                    save_dir=self.assembled,
                    s_range=range(0, self.max_sections),
                    x_range=range(0, self.max_x),
                    y_range=range(0, self.max_y),
                    img_templ=self.template,
                    seg_templ=self.template)
        
    def stack_slices(self):
        #Get prediction files
        predictions = sorted(os.listdir(self.assembled))

        #Read all predictions and append to list
        pred_list = []
        for pred in predictions:
            pred_read = cv2.imread(os.path.join(self.assembled, pred), cv2.IMREAD_GRAYSCALE)
            pred_list.append(pred_read)

        #Stack all predictions into a 3D numpy array
        pred_3d = np.stack(pred_list, axis=0)

        #Save the volume
        np.save(os.path.join(self.volume, "volume.npy"), pred_3d)
        
        return pred_3d
    
    def visualize_volume(self):
        "something filler"

def main():
    #Augmentation
    valid_augmentation = A.Compose([
        A.Normalize(mean=0, std=1), #Specific to the dataset, very important to set these values to the same as training
        A.ToTensorV2()
    ])
    
    #Step 0: Create pipeline
    pipeline = GapJunctionSegmentationPipeline(
        #Name of job / Name of final volume output (I recommend model + data + "segmentation volume")
        name="unet_8jkuifab_sem_adult_s000-699_segmentation_volume_test",
        #Path to model
        model_path="/home/tommytang111/gap-junction-segmentation/models/best_models/unet_base_516imgs_sem_adult_8jkuifab.pt",
        #Dataset class (How to process data for the model)
        dataset_class=TestDataset,
        #Path to sections
        sections_dir="/home/tommytang111/gap-junction-segmentation/data/sem_adult_test/SEM_full/s000-001",
        #Path to where to save tiles
        output_dir="/home/tommytang111/gap-junction-segmentation/data/sem_adult_test/SEM_split/s000-001",
        #Path to where to save predictions
        pred_dir="/home/tommytang111/gap-junction-segmentation/outputs/inference_results/unet_8jkuifab/sem_adult_s000-001",
        #Path to where to save assembled results
        assembled_dir="/home/tommytang111/gap-junction-segmentation/outputs/assembled_results/unet_8jkuifab/sem_adult_s000-001",
        #Path to where to save volume results
        volume_dir="/home/tommytang111/gap-junction-segmentation/outputs/volumetric_results/unet_8jkuifab/sem_adult_s000-001",
        #Template name for images and masks, edit as needed
        template="SEM_adult_image_export_",
        #Augmentations to use for inference, edit above as needed
        augmentations=valid_augmentation,
        #Image size of tiles, default is 512
        img_size=512
    )
    print("Pipeline initialized with name:", pipeline.name)

    #Step 1: Split Slices -> Tiles
    pipeline.create_dataset()
    print("Dataset created with tiles in:", pipeline.tiles)
    
    #Step 2: Run inference on tiles -> Get masks
    pipeline.run_inference()
    print("Inference completed with predictions in:", pipeline.pred)

    #Step 3: Assemble Masks -> Slices
    pipeline.stitch_predictions()
    print("Predictions assembled into slices in:", pipeline.assembled)
    
    #Step 4: Stack Slices -> 3D Volume
    volume = pipeline.stack_slices()
    print("3D volume created with shape:", volume.shape)
    print("Volume saved in:", pipeline.volume)
    
    #Step 5: Plot segmentation volume


if __name__ == "__main__":
    main()
    