"""
Split, predict and stitch volumetric pipeline for gap junction segmentation.
Tommy Tang
Last Updated: Sept 1, 2025
"""

#Requirements:
#Full sections exported from VAST from dataset of interest

#Libraries
from utils import create_dataset_2d, assemble_imgs, check_output_directory
from inference import inference
from models import TestDataset
import albumentations as A
import os
import cv2
import numpy as np
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm

class GapJunctionSegmentationPipeline:
    def __init__(self, name, model_path, dataset_class, sections_dir, output_dir, pred_dir, assembled_dir, volume_dir, template, augmentations, overlap=True, img_size=512, batch_size=8, num_workers=None):
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
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Print number of workers being used
        print(f"Using {self.num_workers} worker processes")
        
    def create_dataset(self):
        self.max_y, self.max_x, self.max_sections, self.section_size = create_dataset_2d(imgs_dir=self.sections, output_dir=self.tiles, img_size=self.img_size, create_overlap=self.overlap, test=True)

    def run_inference(self):
        inference(model_path=self.model,
                  dataset=self.dataset_class,
                  input_dir=self.tiles,
                  output_dir=self.pred,
                  augmentation=self.augmentations,
                  batch_size=self.batch_size,
                  clear=True,
                  filter=True
                )

    def stitch_predictions(self):
        assemble_imgs(img_dir=None,
                    gt_dir=None,
                    pred_dir=self.pred,
                    save_dir=self.assembled,
                    s_range=range(0, self.max_sections),
                    x_range=range(0, self.max_x),
                    y_range=range(0, self.max_y),
                    img_templ=self.template,
                    seg_templ=self.template,
                    overlap=self.overlap,
                    s_size=self.section_size
                    )
        
    def _read_image_worker(self, section):
        """Worker function to read a single section file"""
        img_path = os.path.join(self.assembled, section)
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def stack_slices(self):
        #Create volume directory if it doesn't exist or clear it otherwise
        check_output_directory(self.volume)
        
        #Get prediction files
        predictions = sorted(os.listdir(self.assembled))
        
        #Use multiprocessing pool to read images in parallel
        print(f"Reading {len(predictions)} image slices using {self.num_workers} processes...")
        
        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=self.num_workers) as pool:
        #Read all predictions and append to list in parallel
            pred_list = list(tqdm(
                pool.imap(self._read_image_worker, predictions),
                total=len(predictions),
                desc="Loading images"
            ))
        
        #Stack all predictions into a 3D numpy array
        print("Stacking slices into volume...")
        pred_3d = np.stack(pred_list, axis=0)

        #Save the volume
        volume_path = os.path.join(self.volume, "volume.npy")
        print(f"Saving volume to {volume_path}")
        np.save(volume_path, pred_3d)
        
        return pred_3d
    
    #def visualize_volume(self, volume):

def main():
    #Get number of CPUs from Slurm environment or default to all available CPUs
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    
    #Set number of OpenMP threads (important for numpy operations)
    os.environ["OMP_NUM_THREADS"] = str(num_workers)
    os.environ["MKL_NUM_THREADS"] = str(num_workers)

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
        model_path="/home/tommy111/projects/def-mzhen/tommy111/models/best_models/unet_base_516imgs_sem_adult_8jkuifab.pt",
        #Dataset class (How to process data for the model)
        dataset_class=TestDataset,
        #Path to sections
        sections_dir="/home/tommy111/projects/def-mzhen/tommy111/data/sem_adult/SEM_full/s000-699",
        #Path to where to save tiles
        output_dir="/home/tommy111/scratch/outputs/sem_adult_split/s000-699",
        #Path to where to save predictions
        pred_dir="/home/tommy111/scratch/outputs/inference_results/unet_8jkuifab/sem_adult_s000-699",
        #Path to where to save assembled results
        assembled_dir="/home/tommy111/scratch/outputs/assembled_results/unet_8jkuifab/sem_adult_s000-699/preds",
        #Path to where to save volume results
        volume_dir="/home/tommy111/scratch/outputs/volumetric_results/unet_8jkuifab/sem_adult_s000-699",
        #Template name for images and masks, edit as needed
        template="SEM_adult_image_export_",
        #Augmentations to use for inference, edit above as needed
        augmentations=valid_augmentation,
        #Image size of tiles, default is 512
        img_size=512,
        #Batch size for inference, default is 8
        batch_size=32,
        #Number of workers for data loading and processing, default is all available CPUs
        num_workers=num_workers
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

if __name__ == "__main__":
    main()
    