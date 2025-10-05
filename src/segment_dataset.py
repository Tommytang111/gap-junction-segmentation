"""
Split, predict and stitch volumetric pipeline for gap junction segmentation.
Tommy Tang
Last Updated: Oct 2, 2025
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
from pathlib import Path
from tqdm import tqdm
import time
import scipy.ndimage as ndi

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
        self.max_y, self.max_x, self.max_sections, self.section_shape = create_dataset_2d(imgs_dir=self.sections, output_dir=self.tiles, img_size=self.img_size, create_overlap=self.overlap, test=True)
        
    def run_inference(self):
        inference(model_path=self.model,
                  dataset=self.dataset_class,
                  input_dir=self.tiles,
                  output_dir=self.pred,
                  augmentation=self.augmentations,
                  batch_size=self.batch_size,
                  clear=True,
                  filter=False
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
                    s_size=self.section_shape
                    )
        
        #Check that files were assembled before moving onto stitching
        assembled_pred_dir = os.path.join(self.assembled, "preds")
        self.assembled_pred_dir = assembled_pred_dir  # Store for debugging if needed
        
        if os.path.exists(assembled_pred_dir):
            files = os.listdir(assembled_pred_dir)
            #Filter for image files only
            image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
            image_files = [f for f in files if f.lower().endswith(image_extensions)]
            while len(image_files) < self.max_sections:
                print(f"Waiting for all sections to be assembled... {len(image_files)}/{self.max_sections} completed.")
                time.sleep(10)  #Wait for 10 seconds before checking again
                files = os.listdir(assembled_pred_dir)
                image_files = [f for f in files if f.lower().endswith(image_extensions)]
        
    def _read_image_worker(self, section):
        """Worker function to read a single section file"""
        img_path = os.path.join(self.assembled_pred_dir, section)
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def stack_slices(self):
        #Create volume directory if it doesn't exist or clear it otherwise
        check_output_directory(self.volume)
        
        #Get prediction files
        predictions = sorted(os.listdir(self.assembled_pred_dir))

        #Use multiprocessing pool to read images in parallel
        print(f"Reading {len(predictions)} image slices using {self.num_workers} processes...")
        
        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=self.num_workers) as pool:
        #Read all predictions  and append to list in parallel
            pred_list = list(tqdm(
                pool.imap(self._read_image_worker, predictions), #Can also use pool.map, but not a huge difference
                total=len(predictions),
                desc="Loading images"
            ))
        
        #Stack all predictions into a 3D numpy array
        print("Stacking slices into volume...")
        pred_3d = np.stack(pred_list, axis=0)
        self.volume_file = pred_3d

        #Save the volume
        volume_path = os.path.join(self.volume, "volume.npy")
        print(f"Saving volume to {volume_path}")
        np.save(volume_path, pred_3d)
        
    def downsample_volume(self):
        print("Downsampling volume...")
        #Downsample the volume to max OpenGL size limit
        downsampled_volume_file = ndi.zoom(self.volume_file, (1, 1024/self.volume_file.shape[1], 1024/self.volume_file.shape[2]), order=0).astype(np.uint8)
        self.downsampled_volume = downsampled_volume_file
        
        #Save the downsampled volume
        downsampled_volume_path = os.path.join(self.volume, "volume_downsampled.npy")
        np.save(downsampled_volume_path, downsampled_volume_file)
        
def main():
    #Get number of CPUs from Slurm environment or default to all available CPUs
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    
    #Set number of OpenMP threads (important for numpy operations)
    os.environ["OMP_NUM_THREADS"] = str(num_workers)
    os.environ["MKL_NUM_THREADS"] = str(num_workers)
    
    #Access SLURM_TMPDIR
    tmpdir = os.getenv("SLURM_TMPDIR")
    assert tmpdir is not None, "$SLURM_TMPDIR not found. This job requires access to a temporary directory on the compute node."

    #Augmentation
    valid_augmentation = A.Compose([
        A.Normalize(mean=0, std=1), #Specific to the dataset, very important to set these values to the same as training
        A.ToTensorV2()
    ])
    
    #Step 0: Create pipeline
    pipeline = GapJunctionSegmentationPipeline(
        #Name of job / Name of final volume output (I recommend model + data + "segmentation volume")
        name="unet_h1qrqboc_sem_dauer_2_s000-972_segmentation_volume",
        #Path to model
        model_path="/home/tommy111/projects/def-mzhen/tommy111/models/best_models/unet_base_516imgs_sem_dauer_2_h1qrqboc.pt",
        #Dataset class (How to process data for the model)
        dataset_class=TestDataset,
        #Path to sections
        sections_dir="/home/tommy111/projects/def-mzhen/tommy111/data/sem_dauer_2/SEM_full/s000-972",
        #Path to where to save tiles
        output_dir= Path(tmpdir) / "outputs/sem_dauer_2_split/s000-972",
        #Path to where to save predictions
        pred_dir= Path(tmpdir) / "outputs/inference_results/unet_h1qrqboc/sem_dauer_2_s000-972",
        #Path to where to save assembled results
        assembled_dir="/home/tommy111/projects/def-mzhen/tommy111/outputs/assembled_results/unet_h1qrqboc/sem_dauer_2_s000-972",
        #Path to where to save volume and downsampled volume results
        volume_dir="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_2_s000-972",
        #Template name for images and masks, edit as needed
        template="SEM_dauer_2_export_",
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
    print()

    #Step 1: Split Slices -> Tiles
    pipeline.create_dataset()
    print(f"Max Sections: {pipeline.max_sections}\n, X Tiles: {pipeline.max_x}\n, Y Tiles: {pipeline.max_y}\n, Section Dimensions: {pipeline.section_shape}")
    print("Dataset created with tiles in:", pipeline.tiles)
    print()
    
    #Step 2: Run inference on tiles -> Get masks
    pipeline.run_inference()
    print("Inference completed with predictions in:", pipeline.pred)
    print()

    #Step 3: Assemble Masks -> Slices
    pipeline.stitch_predictions()
    print("Predictions assembled into slices in:", pipeline.assembled_pred_dir)
    print()

    #Step 4: Stack Slices -> 3D Volume
    pipeline.stack_slices()
    print("3D volume created with shape:", pipeline.volume_file.shape)
    print("Volume saved in:", pipeline.volume)
    print()
    
    #Step 5: Downsample 3D Volume -> Visualizable Volume
    pipeline.downsample_volume()
    print(f"Downsampling complete with shape: {pipeline.downsampled_volume.shape}")

if __name__ == "__main__":
    main()
    