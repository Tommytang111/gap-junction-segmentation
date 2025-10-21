"""
Split, predict and stitch volumetric pipeline for gap junction segmentation.
Tommy Tang
Last Updated: Oct 2, 2025
"""
##segment_dataset.py

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
from entity_detection import GapJunctionEntityDetector

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

    def calculate_entity_metrics(self, gt_volume_path=None):
        """
        Calculate entity-level metrics on the assembled 3D volume.
        
        Args:
            gt_volume_path: Optional path to ground truth volume (.npy file).
                          If None, only counts predicted entities without F1 score.
                          
        Returns:
            results_dict: Dictionary with entity metrics and statistics
        """
        print("\n" + "="*60)
        print("Step 6: Calculating Entity-Level Metrics")
        print("="*60)
        
        # Initialize entity detector
        # Using threshold=127 because your volume is uint8 (0-255)
        entity_detector = GapJunctionEntityDetector(
            threshold=127,  # For uint8 images
            min_size=10,    # Minimum voxels for a valid gap junction
            connectivity=26  # 26-connectivity
        )
        
        # Get predicted volume
        if not hasattr(self, 'volume_file'):
            print("Loading volume from disk...")
            volume_path = os.path.join(self.volume, "volume.npy")
            pred_volume = np.load(volume_path)
        else:
            pred_volume = self.volume_file
        
        print(f"Volume shape: {pred_volume.shape}")
        print(f"Volume dtype: {pred_volume.dtype}")
        print(f"Volume range: [{pred_volume.min()}, {pred_volume.max()}]")
        
        # Extract entities from prediction
        print("\nExtracting gap junction entities from predictions...")
        labeled_pred, num_pred = entity_detector.extract_entities(pred_volume)
        
        print(f"✓ Found {num_pred} gap junction entities in predicted volume")
        
        # Save labeled volume
        labeled_path = os.path.join(self.volume, "volume_labeled.npy")
        np.save(labeled_path, labeled_pred)
        print(f"✓ Saved labeled volume to {labeled_path}")
        
        # Calculate entity size statistics
        stats = entity_detector.get_entity_statistics(labeled_pred)
        
        print(f"\nEntity Size Statistics:")
        print(f"  Number of entities: {stats['num_entities']}")
        print(f"  Mean size: {stats['mean_size']:.1f} voxels")
        print(f"  Median size: {stats['median_size']:.1f} voxels")
        print(f"  Min size: {stats['min_size']} voxels")
        print(f"  Max size: {stats['max_size']} voxels")
        print(f"  Std dev: {stats['std_size']:.1f} voxels")
        
        results = {
            'num_entities': num_pred,
            'labeled_volume': labeled_pred,
            'statistics': stats
        }
        
        # If ground truth is provided, calculate F1 score
        if gt_volume_path is not None and os.path.exists(gt_volume_path):
            print(f"\nLoading ground truth from {gt_volume_path}...")
            gt_volume = np.load(gt_volume_path)
            
            if gt_volume.shape != pred_volume.shape:
                print(f"WARNING: Shape mismatch! GT: {gt_volume.shape}, Pred: {pred_volume.shape}")
                print("Cannot calculate F1 score.")
            else:
                print("Calculating entity-level F1 score...")
                f1, precision, recall, metrics = entity_detector.calculate_entity_f1(
                    pred_volume, 
                    gt_volume,
                    iou_threshold=0.5
                )
                
                print(f"\n{'='*60}")
                print("Entity Detection Performance")
                print(f"{'='*60}")
                print(f"  F1 Score:           {f1:.4f}")
                print(f"  Precision:          {precision:.4f}")
                print(f"  Recall:             {recall:.4f}")
                print(f"  True Positives:     {metrics['tp']}")
                print(f"  False Positives:    {metrics['fp']}")
                print(f"  False Negatives:    {metrics['fn']}")
                print(f"  Predicted Entities: {metrics['num_pred']}")
                print(f"  GT Entities:        {metrics['num_gt']}")
                print(f"{'='*60}")
                
                # Save metrics to file
                metrics_path = os.path.join(self.volume, "entity_metrics.txt")
                with open(metrics_path, 'w') as f:
                    f.write(f"Entity Detection Metrics\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"F1 Score: {f1:.4f}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"True Positives: {metrics['tp']}\n")
                    f.write(f"False Positives: {metrics['fp']}\n")
                    f.write(f"False Negatives: {metrics['fn']}\n")
                    f.write(f"Predicted Entities: {metrics['num_pred']}\n")
                    f.write(f"Ground Truth Entities: {metrics['num_gt']}\n")
                print(f"✓ Saved metrics to {metrics_path}")
                
                results.update({
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'metrics': metrics
                })
        
        return results
        
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

    #Step 1: Split Slices -> Tiles
    pipeline.create_dataset()
    print(f"Max Sections: {pipeline.max_sections}\n, X Tiles: {pipeline.max_x}\n, Y Tiles: {pipeline.max_y}\n, Section Dimensions: {pipeline.section_shape}")
    print("Dataset created with tiles in:", pipeline.tiles)
    
    #Step 2: Run inference on tiles -> Get masks
    pipeline.run_inference()
    print("Inference completed with predictions in:", pipeline.pred)

    #Step 3: Assemble Masks -> Slices
    pipeline.stitch_predictions()
    print("Predictions assembled into slices in:", pipeline.assembled_pred_dir)

    #Step 4: Stack Slices -> 3D Volume
    pipeline.stack_slices()
    print("3D volume created with shape:", pipeline.volume_file.shape)
    print("Volume saved in:", pipeline.volume)
    
    #Step 5: Downsample 3D Volume -> Visualizable Volume
    pipeline.downsample_volume()
    print(f"Downsampling complete with shape: {pipeline.downsampled_volume.shape}")

    #Step 6: Calculate Entity-Level Metrics
    # we need to specify the path of our gt volume in order for this section to work
    # Otherwise set to None to just count entities
    # entity_results = pipeline.calculate_entity_metrics(gt_volume_path=None)
    # print(f"\n✓ Entity detection complete! Found {entity_results['num_entities']} gap junction entities")
    
    # print(f"✓ Labeled volume saved to: {pipeline.volume}/volume_labeled.npy")
    # print(f"✓ Statistics: Mean size = {entity_results['statistics']['mean_size']:.1f} voxels")

if __name__ == "__main__":
    
    main()
    