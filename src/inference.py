"""
Inference and other tools for Gap Junction Segmentation API.
Tommy Tang
June 2, 2025
"""

#LIBRARIES
from src.utils import *
from pathlib import Path
import re
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import subprocess
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex
)
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import label
#Custom libraries
from unet import *

#FUNCTIONS
def filter_pixels(img) -> np.ndarray:
    """
    Changes all non-zero pixel islands in an image to zero if they are less than 8 pixels in size. Designed for greyscale images.
    """
    # Create a copy to avoid modifying the original during iteration
    filtered = img.copy()
    # Label connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(img > 0, structure=structure)
    # For each pixel, check if its component has at least 8 pixels
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] != 0:
                component_label = labeled[y, x]
                if component_label == 0:
                    filtered[y, x] = 0
                    continue
                # Count pixels in this component
                count = np.sum(labeled == component_label)
                if count < 8:
                    filtered[y, x] = 0
    return filtered
        
#NEED FUNCTION FOR GETTING TILES FROM A LARGE SECTION
#NEED FUNCTION FOR STITCHING IMAGES BACK TOGETHER
#NEED FUNCTION FOR RESIZING IMAGES

def inference():
    """
    Generates gap junction prediction masks for each batch, converting each tensor into a numpy array with cpu as uint8. 
    Keeps track of the image number while going through batches, assuming the data is sorted by alphabetically ascending 
    order when read from disk. 
    """
    #Important Paths
    section_of_interest = "s200-209" #Section of interest, usually 10 slices
    model_dir = "/home/tommytang111/models/modelv6.pk1" #Model 2d_gd_mem_run1
    output_dir = Path("/home/tommytang111/inference_results/sem_adult") / section_of_interest #Output for inference
    dataset_dir = Path("/home/tommytang111/data/sem_adult/SEM_split") / section_of_interest #Data

    #Data and Labels (sorted because naming convention is typically dataset, section, coordinates. Example: SEM_Dauer_2_image_export_s000 -> 001)
    imgs = [i for i in sorted(os.listdir(Path(dataset_dir) / "imgs"))] 
    labels = [i for i in sorted(os.listdir(Path(dataset_dir) / "gts"))]

    #Create TestDataset class (Note:There are other dataset types in datasets.py). This defines how images/data is read from disk.
    dataset = TestDataset(dataset_dir, td=False, membrane=False)
    #Load dataset class in Dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

    #Load model
    model = joblib.load(model_dir)
    model = model.to("cuda") #Send to gpu
    model.eval() #Set evaluation mode

    #NOTE: I SHOULD NORMALIZE THE IMAGES BEFORE PASSING THEM TO THE MODEL

    check_directory(output_dir)

    img_num = 0 #References which image is being refered to in imgs:List

    with torch.no_grad(): 
        for batch in tqdm(dataloader):
            image = batch[0].to("cuda")
            batch_pred = model(image)
            for i in range(batch_pred.shape[0]): #For each image in the batch
                #Convert tensor to binary mask using Sigmoid activation first
                gj_pred = (nn.Sigmoid()(batch_pred[i]) >= 0.5)
                gj_pred = gj_pred.squeeze(0).detach().cpu().numpy().astype("uint8")
                save_name = Path(output_dir) / re.sub(r'.png$', r'_pred.png', imgs[img_num])
                cv2.imwrite(save_name, gj_pred * 255) #All values either black:0 or white:255
                img_num += 1
                
def visualize_results():
    """
    Visualizes the results of the segmentation model by overlaying the predicted masks on the original images.
    """

    #Test a random image from the dataset
    random_path = random.choice(imgs)
    random_img = Path(dataset_dir) / "imgs" / random_path
    random_resized_img = np.array(resize_image(str(random_img), 512, 512, (0,0,0)))
    random_pred = cv2.imread(str(Path(output_dir) / re.sub(r'.png$', r'_pred.png', str(random_path))), cv2.IMREAD_GRAYSCALE)

    #Make overlay
    random_pred2 = cv2.cvtColor(random_pred, cv2.COLOR_GRAY2RGB)
    random_pred2[random_pred == 255] = [0, 0, 255] #Blue
    overlay = cv2.addWeighted(random_resized_img, 1, random_pred2, 1, 0)

    #Plot
    fig, ax = plt.subplots(1,4, figsize=(15,5))
    ax[0].imshow(cv2.imread(random_img), cmap="gray")
    ax[0].set_title('Original')
    ax[1].imshow(random_resized_img, cmap="gray")
    ax[1].set_title('Cropped/Paded')
    ax[2].imshow(random_pred, cmap="gray")
    ax[2].set_title('Gap Junction')
    ax[3].imshow(overlay)
    ax[3].set_title('Overlay')
    
def evaluate_model():
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
    """
    Main function to run inference, visualize results, and evaluate the model.
    """
    #Run inference
    inference()
    
    #Visualize results
    visualize_results()
    
    #Evaluate model performance
    evaluate_model()

if __name__ == "__main__":
    main()
