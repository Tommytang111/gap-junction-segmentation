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

#Custom libraries
from models import UNet, TestDataset
from utils import filter_pixels, resize_image, assemble_imgs, split_img, check_output_directory, create_dataset_2d

#FUNCTIONS

#NEED FUNCTION FOR GETTING TILES FROM A LARGE SECTION
#NEED FUNCTION FOR STITCHING IMAGES BACK TOGETHER

def predict_multiple_models(model1_path, model2_path, model3_path, data_dir):
    """
    Compare predictions from three different UNet models on a randomly selected image.
    
    This function loads a random image from the specified dataset directory, runs inference
    using three different trained UNet models, and creates comparison visualizations showing
    the original image, individual predictions, ground truth, and overlay comparisons.
    
    Args:
        model1_path (str): Path to the first trained model checkpoint (.pt file).
                          Expected to be the 516imgs_sem_adult model.
        model2_path (str): Path to the second trained model checkpoint (.pt file).
                          Expected to be the 516imgs_sem_dauer_2 model.
        model3_path (str): Path to the third trained model checkpoint (.pt file).
                          Expected to be the 1032imgs_pooled model.
        data_dir (str or Path): Path to the dataset directory containing 'imgs' and 'gts' 
                               subdirectories with corresponding image and label files.
    
    Returns:
        tuple: A tuple containing two matplotlib figure objects:
            - fig1 (matplotlib.figure.Figure): Figure showing the original grayscale image.
            - fig2 (matplotlib.figure.Figure): 2x4 subplot comparison figure containing:
                - Top row: Individual predictions from each model and ground truth
                - Bottom row: Overlay visualizations (predictions/truth over original image)
    
    Note:
        - Requires CUDA-capable GPU for model inference.
        - Label files are expected to follow the naming convention: 
          original_name.png -> original_name_label.png
        - All models should be UNet architectures with compatible input/output dimensions.
        - Uses single_image_inference() function for individual model predictions.
    
    Example:
        >>> fig1, fig2 = predict_multiple_models(
        ...     model1_path='models/adult_model.pt',
        ...     model2_path='models/dauer_model.pt', 
        ...     model3_path='models/pooled_model.pt',
        ...     data_dir='data/test_dataset'
        ... )
        >>> fig1.savefig('original_image.png')
        >>> fig2.savefig('model_comparison.png')
    """
    imgs = os.listdir(Path(data_dir) / "imgs")
    random_img = random.choice(imgs)
    random_img_path = Path(data_dir) / "imgs" / random_img

    img1 = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
    gts1 = cv2.imread(Path(data_dir) / "gts" / re.sub(r'.png$', r'_label.png', str(random_img)), cv2.IMREAD_GRAYSCALE)

    model1_pred = single_image_inference(image=img1,
                    model_path=model1_path,
                    model=UNet())
    model2_pred = single_image_inference(image=img1,
                    model_path=model2_path,
                    model=UNet())
    model3_pred = single_image_inference(image=img1,
                    model_path=model3_path,
                    model=UNet())

    fig1 = plt.figure(1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    #Plot
    fig2 = plt.figure(2, figsize=(16, 12), dpi=300)
    plt.subplot(241)
    plt.imshow(model1_pred, cmap='gray')
    plt.title('Model 516imgs_sem_adult', fontsize=10)
    plt.subplot(242)
    plt.imshow(model2_pred, cmap='gray')
    plt.title('Model 516imgs_sem_dauer_2', fontsize=10)
    plt.subplot(243)
    plt.imshow(model3_pred, cmap='gray')
    plt.title('Model 1032imgs_pooled', fontsize=10)
    plt.subplot(244)
    plt.imshow(gts1, cmap='gray')
    plt.title('Truth', fontsize=10)
    plt.subplot(245)
    plt.imshow(img1, cmap='gray')
    plt.imshow(model1_pred, cmap='gray', alpha=0.5)
    plt.subplot(246)
    plt.imshow(img1, cmap='gray')
    plt.imshow(model2_pred, cmap='gray', alpha=0.5)
    plt.subplot(247)
    plt.imshow(img1, cmap='gray')
    plt.imshow(model3_pred, cmap='gray', alpha=0.5)
    plt.subplot(248)
    plt.imshow(img1, cmap='gray')
    plt.imshow(gts1, cmap='gray', alpha=0.5)
    #plt.tight_layout
    plt.subplots_adjust(wspace=0.2, hspace=-0.5)

    return fig1, fig2

def single_image_inference(image:np.ndarray, model_path:str, model):
    """
    Makes a mask by predicting on a single image
    """
    #Setup model
    model = model
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda") #Send to gpu
    model.eval() 
    
    #Prepare image
    image = torch.from_numpy(image).float() / 255.0       # Convert to float tensor, normalize if needed
    image = image.unsqueeze(0).unsqueeze(0)               # Add shape: (1, 1, H, W) for batch and channel
    
    #Inference
    with torch.no_grad():
        pred = model(image.to('cuda')) 
        pred = nn.Sigmoid()(pred) >= 0.5 #Binarize with sigmoid activation function
        pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype("uint8") #Convert from tensor back to image
    
    return pred

def inference(model_path:str, input_dir:str, output_dir:str, threshold:float=0.5, filter:bool=False):
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
        filter (bool): Applies a pixel filter if true. Default is false.

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
        for batch in tqdm(dataloader, desc="Predicting"):
            image = batch[0].to("cuda")
            batch_pred = model(image)
            for i in range(batch_pred.shape[0]): #For each image in the batch
                #Convert tensor to binary mask using Sigmoid activation function
                gj_pred = (nn.Sigmoid()(batch_pred[i]) >= threshold)
                gj_pred = gj_pred.squeeze(0).detach().cpu().numpy().astype("uint8") #Convert tensor to numpy array
                if filter:
                    gj_pred = filter_pixels(gj_pred, size_threshold=10) #Apply filter_pixels 
                save_name = Path(output_dir) / re.sub(r'.png$', r'_pred.png', imgs[img_num]) 
                cv2.imwrite(save_name, gj_pred * 255) #All values either black:0 or white:255
                img_num += 1

def visualize(data_dir:str, pred_dir:str, base_name:str=None, style:int=1, random:bool=True, figsize:tuple=(15,5)) -> plt.Figure:
    """
    Visualizes segmentation model predictions through custom plots comparing original images, predictions, and ground truth.

    This function creates visual comparisons between original images, model predictions, and ground truth segmentation masks.
    It supports two plotting styles: a 4-panel view with individual components and overlay, or a 3-panel view with 
    colored overlays. Images are automatically resized to 512x512 if needed, and overlays use distinct colors 
    (blue for predictions, orange for ground truth) for easy comparison.

    Parameters:
        data_dir (str): Path to the directory containing 'imgs' and 'gts' subdirectories with original images and ground truth masks.
        pred_dir (str): Path to the directory containing predicted segmentation masks (with '_pred.png' suffix).
        base_name (str, optional): Specific image filename to visualize. Required if random=False. Default is None.
        style (int, optional): Plotting style. 1 for 4-panel view (grayscale + overlay), 2 for 3-panel view (colored overlays). Default is 1.
        random (bool, optional): If True, randomly selects an image from the dataset. If False, uses base_name. Default is True.
        figsize (tuple, optional): Figure size as (width, height) in inches. Default is (15, 5).

    Returns:
        plt.Figure: The matplotlib figure object containing the visualization.

    Raises:
        ValueError: If data_dir does not contain required 'imgs' and 'gts' subdirectories.
        AssertionError: If random=False but base_name is None.

    Examples:
        # Visualize a random image with 4-panel layout
        fig = visualize("/path/to/data", "/path/to/predictions")
        
        # Visualize a specific image with 3-panel colored overlay layout
        fig = visualize("/path/to/data", "/path/to/predictions", 
                       base_name="image_001.png", style=2, random=False)
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
    pred2[pred == 255] = [0, 60, 255] #Blue
    pred_overlay = cv2.addWeighted(resized_img, 1, pred2, 1, 0)
    gts2 = cv2.cvtColor(gts, cv2.COLOR_GRAY2BGR)
    gts2[gts == 255] = [255, 0, 0] #Orange
    gts_overlay = cv2.addWeighted(resized_img, 1, gts2, 1, 0)
    #Double overlay
    double_overlay = cv2.addWeighted(pred_overlay, 1, gts2, 1, 0)
    
    #Generate plot
    plot1(resized_img, pred, gts, double_overlay) if style == 1 else plot2(resized_img, pred_overlay, gts_overlay)
    print(f"Showing: {name}")

    return plt.gcf()

def evaluate(data_dir:str, pred_dir:str, figsize=(10, 6), title:str="Model X Post-Inference Evaluation on Data Y") -> plt.Figure:
    """
    Evaluates segmentation model performance by comparing predictions against ground truth masks.

    This function computes standard segmentation metrics (accuracy, precision, recall, F1-score, IoU) 
    for all images in a dataset by comparing predicted masks with ground truth labels. It processes 
    each image individually to calculate per-image metrics, then averages them across the entire dataset.
    The results are displayed in a bar chart for easy visualization of model performance.

    Parameters:
        data_dir (str): Path to the directory containing 'imgs' and 'gts' subdirectories with original 
                       images and ground truth masks.
        pred_dir (str): Path to the directory containing predicted segmentation masks (with '_pred.png' suffix).
        figsize (tuple, optional): Figure size as (width, height) in inches for the results bar chart. 
                                 Default is (10, 6).
        title (str, optional): Title for the evaluation results bar chart. 
                              Default is "Model X Post-Inference Evaluation on Data Y".

    Returns:
        plt.Figure: The matplotlib figure object containing the evaluation results bar chart.

    Raises:
        ValueError: If data_dir does not contain required 'imgs' and 'gts' subdirectories.

    Notes:
        - Images are automatically resized to match dimensions if needed.
        - Ground truth labels are filtered to remove small pixel islands (<8 pixels).
        - Predictions are binarized using threshold of 127.
        - Metrics are calculated per-image and then averaged across all images.
        - Results are printed to console and displayed as a bar chart.

    Examples:
        # Evaluate model performance with default settings
        fig = evaluate("/path/to/data", "/path/to/predictions")
        
        # Evaluate with custom title and figure size
        fig = evaluate("/path/to/data", "/path/to/predictions", 
                      figsize=(12, 8), 
                      title="UNet Performance on Test Dataset")
    """
    #Check if input directory has the required subdirectories
    data = os.listdir(data_dir)
    if not ("imgs" in data and "gts" in data):
        raise ValueError("Input directory must contain 'imgs' and 'gts' subdirectories.")
    
    #Data and Labels (sorted because naming convention is typically dataset, section, coordinates. Example: SEM_Dauer_2_image_export_s000 -> 001)
    imgs = [i for i in sorted(os.listdir(Path(data_dir) / "imgs"))] 
    
    #Create results dictionary
    results = {
        'f1': [],
        'recall': [],
        'precision': [],
        'iou': [],
        'accuracy': []
    }
    
    #We have a list of all the input image file names in imgs
    for img in tqdm(imgs, desc="Evaluating"):
        #Load Predictions
        gj_pred = Path(pred_dir) / re.sub(r'.png$', r'_pred.png', img)
        gj_pred = cv2.imread(gj_pred, cv2.IMREAD_GRAYSCALE)

        #Load labels
        gj_label = Path(data_dir) / 'gts' / re.sub(r'.png$', r'_label.png', img)
        gj_label = cv2.imread(gj_label, cv2.IMREAD_GRAYSCALE)
        gj_label[gj_label != 0] = 255  # Convert 1s to 255 if they aren't already 255
        
        #Ensure same dimensions
        if gj_pred.shape != gj_label.shape:
            gj_pred = cv2.resize(gj_pred, (gj_pred.shape[1], gj_pred.shape[0]))
            
        #Binarize masks (0 or 1)
        gj_pred_binary = (gj_pred > 127).astype(np.uint8)
        gj_label_binary = (gj_label > 127).astype(np.uint8)
        
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
        #dice = (2 * iou) / (1 + iou)

        #Append to results
        results['f1'].append(f1)
        results['recall'].append(recall)
        results['precision'].append(precision)
        results['iou'].append(iou)
        results['accuracy'].append(accuracy)

    #Calculate averages
    for key in results:
        results[key] = np.mean(results[key])

    print(results)
    
    #Plot bar chart of evaluation results
    plt.figure(figsize=figsize)
    plt.title(f"{title}")
    plt.bar(results.keys(), results.values())
    plt.ylim(0,1)
    plt.xlabel('Segmentation Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    
    return plt.gcf()
    
def main():
    #Data and Model
    model_path = "/home/tommytang111/gap-junction-segmentation/models/unet_base_516imgs_sem_adult_ak5v2m7m.pt"
    data_dir = "/home/tommytang111/gap-junction-segmentation/data/sem_dauer_2/SEM_split/s000-050_filtered"
    pred_dir = "/home/tommytang111/gap-junction-segmentation/outputs/inference_results/unet_ak5v2m7m/s000-050_filtered"
    
    #Process images if necessary
    
    #Run inference
    # inference(model_path=model_path,
    #           input_dir=data_dir,
    #           output_dir=pred_dir,
    #           filter=True
    #           )
    
    #Visualize results
    for i in range(1):
        fig = visualize(data_dir=data_dir,
                        pred_dir=pred_dir,
                        style=1, random=True, base_name="SEM_dauer_2_image_export_s032_Y9_X15.png")
        plt.show()

    #Evaluate model performance
    performance_plot = evaluate(data_dir=data_dir,
                                pred_dir=pred_dir,
                                title="Model Unet_ak5v2m7m Performance on sem_dauer_2_s000-050_filtered",
                                )
    plt.show()

if __name__ == "__main__":
    main()
