"""
Inference for Gap Junction Segmentation API.
Tommy Tang
June 2, 2025
"""

#LIBRARIES
from pathlib import Path
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import subprocess
from scipy.ndimage import label
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex
)

#Custom libraries
from models import UNet, TestDataset
from utils import filter_pixels, resize_image, assemble_imgs, split_img, check_output_directory, create_dataset_2d

#FUNCTIONS
#NEED FUNCTION FOR GETTING TILES FROM A LARGE SECTION
#NEED FUNCTION FOR STITCHING IMAGES BACK TOGETHER
#NEED FUNCTION FOR RESIZING IMAGES

def inference(model_path:str, input_dir:str, output_dir:str, threshold:float=0.5):
    """
    Runs inference using a trained UNet model on a dataset of images to generate segmentation masks.

    This function loads a trained UNet model, processes images from the specified input directory,
    generates predicted segmentation masks, and saves the results to the output directory. The input
    directory must contain 'imgs' and 'gts' subdirectories. The output masks are thresholded using
    a sigmoid activation and saved as binary images.

    Parameters:
        model_path (str): Path to the trained model weights (.pt file).
        input_dir (str): Path to the input directory containing 'imgs' and 'gts' subfolders.
        output_dir (str): Path to the directory where predicted masks will be saved.
        threshold (float, optional): Threshold for binarizing the predicted mask after sigmoid activation. Default is 0.5.

    Returns:
        None
    """
    #Check if input directory has the required subdirectories
    data = os.listdir(input_dir)
    if not ("imgs" in data and "gts" in data):
        raise ValueError("Input directory must contain 'imgs' and 'gts' subdirectories.")

    #Data and Labels (sorted because naming convention is typically dataset, section, coordinates. Example: SEM_Dauer_2_image_export_s000 -> 001)
    imgs = [i for i in sorted(os.listdir(Path(input_dir) / "imgs"))] 
    labels = [i for i in sorted(os.listdir(Path(input_dir) / "gts"))]

    #Create TestDataset class (Note:There are other dataset types in datasets.py). This defines how images/data is read from disk.
    dataset = TestDataset(input_dir, three_dim=False)
    #Load dataset class in Dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    #Load model and set to evaluation mode
    #model = joblib.load(model_dir)
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda") #Send to gpu
    model.eval() 

    #Check if output directory exists, if not create it
    check_output_directory(output_dir)

    #Keeps track of the image number in the batch
    img_num = 0 
    
    #Generates gap junction prediction masks per batch
    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            image = batch[0].to("cuda")
            batch_pred = model(image)
            for i in range(batch_pred.shape[0]): #For each image in the batch
                #Convert tensor to binary mask using Sigmoid activation function
                gj_pred = (nn.Sigmoid()(batch_pred[i]) >= threshold)
                gj_pred = gj_pred.squeeze(0).detach().cpu().numpy().astype("uint8") #Convert tensory to numpy array
                save_name = Path(output_dir) / re.sub(r'.png$', r'_pred.png', imgs[img_num])
                cv2.imwrite(save_name, gj_pred * 255) #All values either black:0 or white:255
                img_num += 1

def visualize(data_dir:str, pred_dir:str, base_name:str=None, style:int=1, random:bool=True, figsize:tuple=(15,5)) -> plt.Figure:
    """
    Visualizes the segmentation model predictions through a variety of custom plots.
    """
    #Check if input directory has the required subdirectories
    data = os.listdir(data_dir)
    if not ("imgs" in data and "gts" in data):
        raise ValueError("Input directory must contain 'imgs' and 'gts' subdirectories.")
    
    #Plotting functions
    def plot1(img, pred, gts, double_overlay, figsize=figsize):
        fig, ax = plt.subplots(1,4, figsize=figsize)
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title('Image')
        #ax[1].imshow(random_resized_img, cmap="gray")
        #ax[1].set_title('Cropped/Paded')
        ax[1].imshow(pred, cmap="gray")
        ax[1].set_title('Prediction')
        ax[2].imshow(gts, cmap="gray")
        ax[2].set_title('Truth')
        ax[3].imshow(cv2.cvtColor(double_overlay, cv2.COLOR_BGR2RGB))
        ax[3].set_title('Overlay')
        
    def plot2(img, pred_overlay, gts_overlay, figsize=figsize):
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title('Image')
        ax[1].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Prediction')
        ax[2].imshow(cv2.cvtColor(gts_overlay, cv2.COLOR_BGR2RGB))
        ax[2].set_title('Truth')
        
    if random:
        #Data Source
        imgs = [i for i in sorted(os.listdir(Path(data_dir) / "imgs"))] 

        #Test a random image, prediction, and label from the dataset
        random_path = rd.choice(imgs)
        
    else:
        assert base_name is not None, "base_name must be provided if random is False."
    
    #Image of interest 
    name = random_path if random else base_name
    
    #Image, ground truth, and prediction
    img = cv2.imread(Path(data_dir) / "imgs" / name)
    gts = cv2.imread(Path(data_dir) / "gts" / re.sub(r'.png$', r'_label.png', str(name)), cv2.IMREAD_GRAYSCALE)
    gts[gts > 0] = 255 #Binarize to 0 and 255
    pred = cv2.imread(str(Path(pred_dir) / re.sub(r'.png$', r'_pred.png', str(name))), cv2.IMREAD_GRAYSCALE)

    #Resize image to (X, Y) if needed
    resized_img = img.copy() if img.shape[:2] == (512, 512) else np.array(resize_image(Path(data_dir) / "imgs" / name, 512, 512, (0,0,0), channels=True))

    #Make overlays
    pred2 = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    pred2[pred == 255] = [255, 0, 0] #Blue
    pred_overlay = cv2.addWeighted(resized_img, 1, pred2, 1, 0)
    gts2 = cv2.cvtColor(gts, cv2.COLOR_GRAY2BGR)
    gts2[gts == 255] = [0, 60, 255] #Orange
    gts_overlay = cv2.addWeighted(resized_img, 1, gts2, 1, 0)
    #Double overlay
    double_overlay = cv2.addWeighted(pred_overlay, 1, gts2, 1, 0)
    
    #Generate plot
    plot1(resized_img, pred, gts, double_overlay) if style == 1 else plot2(resized_img, pred_overlay, gts_overlay)
    print(f"Showing: {name}")

    return plt.gcf()
    
def evaluate():
    """
    Evaluates model performance on an example dataset.
    """
    #Create results dictionary
    results = {'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
            'dice': []
    }
    
    #We have a list of all the input image file names in imgs
    for img in tqdm(imgs):
        #Load Predictions
        gj_pred = Path(output_dir) / re.sub(r'.png$', r'_pred.png', img)
        gj_pred = cv2.imread(gj_pred, cv2.IMREAD_GRAYSCALE)

        #Load labels
        gj_label = Path(dataset_dir) / 'gts' / re.sub(r'.png$', r'_label.png', img)
        gj_label = cv2.imread(gj_label, cv2.IMREAD_GRAYSCALE)
        gj_label[gj_label != 0] = 255  # Convert 1s to 255 if they aren't already 255
        gj_label = filter_pixels(gj_label)  # Filter out potential errors
        
        #Ensure same dimensions
        if gj_pred.shape != gj_label.shape:
            gj_pred = cv2.resize(gj_pred, (gj_pred.shape[1], gj_pred.shape[0]))
            
        #Binarize masks (0 or 1)
        gj_pred_binary = (gj_pred > 127).astype(np.uint8)
        gj_label_binary = (gj_label > 128).astype(np.uint8)
        
        #Flatten masks for metric calculations
        gj_pred_flat = gj_pred_binary.flatten()
        gj_label_flat = gj_label_binary.flatten()
        
        #Calculate metrics
        cm = confusion_matrix(gj_label_flat, gj_pred_flat, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() #Extract from confusion matrix
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * iou) / (1 + iou)
        
        #Append to results
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['iou'].append(iou)
        results['dice'].append(dice)

    #Calculate averages
    for key in results:
        results[key] = np.mean(results[key])

    print(results)
    
    #Plot bar chart of evaluation results
    plt.figure(figsize=(10,6))
    plt.title('modelv7 Post-Inference Evaluation on sem_adult sections 200-209')
    plt.bar(results.keys(), results.values())
    plt.ylim(0,1)
    plt.xlabel('Segmentation Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.show()
    
def main():
    #Process images if necessary
    
    #Run inference
    # inference(model_path="/home/tommytang111/gap-junction-segmentation/models/unet_base_pooled_516imgs_sem_dauer_2_516imgs_sem_adult_s1mdf621.pt",
    #           input_dir="/home/tommytang111/gap-junction-segmentation/data/sem_dauer_2/SEM_split/s000-050_filtered",
    #           output_dir="/home/tommytang111/gap-junction-segmentation/outputs/inference_results/sem_dauer_2/s000-050_filtered"
    #           )
    
    #Visualize results
    fig = visualize(data_dir="/home/tommytang111/gap-junction-segmentation/data/sem_dauer_2/SEM_split/s000-050_filtered",
                    pred_dir="/home/tommytang111/gap-junction-segmentation/outputs/inference_results/sem_dauer_2/s000-050_filtered",
                    style=1, random=False, base_name="SEM_dauer_2_image_export_s032_Y9_X15.png")
    plt.show()

    #Evaluate model performance
    #evaluate()

if __name__ == "__main__":
    main()
