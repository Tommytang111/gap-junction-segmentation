"""
Complete set of utility functions for Gap Junction Segmentation API.
Tommy Tang
June 2, 2025
"""

#LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import cv2
import re
from PIL import Image
from typing import Union
from scipy.ndimage import label

#FUNCTIONS
def resize_image(image:Union[str,np.ndarray], new_width:int, new_length:int, pad_clr:tuple, channels=True) -> Image.Image:
    """
    Resize an image to fit within a specified width and length, maintaining the aspect ratio.
    If the image does not fit within the specified dimensions, it will be padded with a specified color.
    This is not the same as just padding an image, because the old image will always try to maximize its area 
    in the new image.
    
    image: Image to be resized
    new_width: New width of the image
    new_length: New length of the image
    pad_clr: Color to pad the image with if it does not fit within the specified dimensions
    channels: If False, a grayscale image will be returned as grayscale (1 channel dim) after resizing.
    
    Returns: Resized image as a PIL image.
    """
    #Open Image as either string or numpy array
    if isinstance(image, str):
        img = Image.open(image) 
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError("Unsupported image type")
    
    orig_width, orig_height = img.size

    # Compute scaling factor to fit within target box
    scale = min(new_width / orig_width, new_length / orig_height)
    resized_width = int(orig_width * scale)
    resized_height = int(orig_height * scale)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    # Determine mode and pad color
    mode = img.mode
    if mode == 'L' and not channels:
        pad_color = pad_clr[0] if isinstance(pad_clr, (tuple, list)) else pad_clr
    elif mode == 'L' and channels:
        # If channels=True, convert to RGB
        img = img.convert('RGB')
        mode = 'RGB'
        pad_color = pad_clr
    else:
        pad_color = pad_clr
        
    # Create new image and paste resized image onto center
    new_img = Image.new(mode, (new_width, new_length), pad_color)
    paste_x = (new_width - resized_width) // 2
    paste_y = (new_length - resized_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img
    
def split_img(img:np.ndarray, offset=256, tile_size=512, names=False):
    """ 
    Splits an image into tiles of a specified size, with an optional offset to remove borders.
    
    Parameters:
        img (np.ndarray): Input image as a NumPy array.
        offset (int): Number of pixels to crop from each border before splitting.
        tile_size (int): Size of each square tile.
        names (bool): If True, also returns a list of tile names.

    Returns:
        list: List of image tiles (np.ndarray).
        list (optional): List of tile names if names=True.

    Example:
        tiles, names = split_img(image, offset=0, tile_size=512, names=True)
    """
    if offset:
        img = img[offset:-offset, offset:-offset]
    imgs = []
    names_list = []
    for i in range(0, img.shape[0], tile_size):
        for j in range(0, img.shape[1], tile_size):
            imgs.append(img[i:i+tile_size, j:j+tile_size])
            names_list.append("Y{}_X{}".format(i//tile_size, j//tile_size))
    return (imgs, names_list) if names else imgs

def write_imgs(source_path:str, target_path:str, suffix:str="", index:int=1):
    """
    Splits all images in a source directory into smaller tiles and saves them to a target directory.

    For each image in the source directory, this function:
      - Reads the image in grayscale.
      - Splits the image into 512x512 tiles (with no offset).
      - Saves each tile to the target directory, naming each tile using the original filename's stem,
        the provided suffix, the tile index, and the original file extension.

    If the target directory exists, it will be cleared before saving new tiles.

    Parameters:
        source_path (str): Path to the directory containing source images.
        target_path (str): Path to the directory where split tiles will be saved.
        suffix (str): Suffix to add to each tile's filename before the index and extension (default: "").
        index (int): Starting index for naming the tiles (default: 1).

    Returns:
        None

    Example:
        write_imgs(
            source_path="/path/to/source",
            target_path="/path/to/target",
            suffix="_part",
            index=1
        )
    """
    if os.path.exists(target_path):
        subprocess.run(f"rm -f {target_path}/*", shell=True)
    else:
        os.makedirs(target_path)

    imgs = os.listdir(source_path)
    
    for img in imgs:
        read_img = cv2.imread(f"{source_path}/{img}", cv2.IMREAD_GRAYSCALE)
        split_imgs = split_img(read_img, offset=0, tile_size=512)
        for i, split in enumerate(split_imgs):
            cv2.imwrite(f"{target_path}/{Path(img).stem}{suffix}{i+index}{Path(img).suffix}", split)
            
def assemble_imgs(img_dir:str, gt_dir:str, pred_dir:str, save_dir:str, img_templ:str, seg_templ:str, s_range:range, x_range:range, y_range:range, missing_dir:str=None):
    """
    Assembles (stitches together) image tiles back into full sections.
    
    Parameters:
        img_dir (str): Directory containing image tiles
        gt_dir (str): Directory containing ground truth tiles (can be None)
        pred_dir (str): Directory containing prediction tiles
        save_dir (str): Directory to save assembled results
        missing_dir (str): Directory to copy missing tiles from (optional)
        img_templ (str): Template for image filenames
        seg_templ (str): Template for segmentation filenames
        s_range (range): Range of section indices
        x_range (range): Range of X tile indices
        y_range (range): Range of Y tile indices
    
    Returns:
        None
    """
    from tqdm import tqdm
    import shutil
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for s in s_range:
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        
        for y in tqdm(y_range, desc=f"Processing section {s}"):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            
            for x in x_range:
                # Create filename suffix
                suffix = f"s{str(s).zfill(3)}_Y{y}_X{x}"
                
                # Handle missing images
                img_path = os.path.join(img_dir, img_templ + suffix + ".png")
                if not os.path.isfile(img_path):
                    if missing_dir is not None:
                        shutil.copy(os.path.join(missing_dir, img_templ + suffix + ".png"), img_path)
                    else:
                        raise FileNotFoundError(f"Missing image {img_templ + suffix + '.png'}")
                
                # Load image
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    raise ValueError(f"Could not load image: {img_path}")
                
                # Load ground truth if directory provided
                gt = None
                if gt_dir:
                    gt_path = os.path.join(gt_dir, seg_templ + suffix + "_label.png")
                    if os.path.isfile(gt_path):
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        gt = np.zeros_like(im)
                
                # Load prediction
                pred_path = os.path.join(pred_dir, img_templ + suffix + "_pred.png")
                if os.path.isfile(pred_path):
                    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                else:
                    pred = np.zeros_like(im)
                
                # Ensure all arrays have the same shape
                if gt is not None and gt.shape != im.shape:
                    gt = np.zeros_like(im)
                if pred.shape != im.shape:
                    pred = np.zeros_like(im)
                
                # Append to row accumulators
                y_acc_img.append(im)
                y_acc_pred.append(pred)
                if gt_dir:
                    y_acc_gt.append(gt)
            
            # Concatenate row tiles horizontally
            if y_acc_img:  # Check if we have any tiles
                s_acc_img.append(np.concatenate(y_acc_img, axis=1))
                s_acc_pred.append(np.concatenate(y_acc_pred, axis=1))
                if gt_dir:
                    s_acc_gt.append(np.concatenate(y_acc_gt, axis=1))
        
        # Concatenate all rows vertically to form complete section
        if s_acc_img:  # Check if we have any rows
            assembled_img = np.concatenate(s_acc_img, axis=0)
            assembled_pred = np.concatenate(s_acc_pred, axis=0)
            if gt_dir:
                assembled_gt = np.concatenate(s_acc_gt, axis=0)
            
            # Create output filename suffix
            out_suffix = f"s{str(s).zfill(3)}"
            
            # Save assembled results
            cv2.imwrite(os.path.join(save_dir, img_templ + out_suffix + "_img.png"), assembled_img)
            cv2.imwrite(os.path.join(save_dir, img_templ + out_suffix + "_pred.png"), assembled_pred)
            if gt_dir:
                cv2.imwrite(os.path.join(save_dir, seg_templ + out_suffix + "_label.png"), assembled_gt)
            
            print(f"Saved assembled section {s} with shape {assembled_img.shape}")

def overlay_img(img:str, pred:str):
    """
    Overlay a prediction mask on top of a grayscale image for visualization.

    This function loads a grayscale image and a prediction mask, then overlays the prediction
    on the image with transparency for easy visual inspection.

    Parameters:
        img (str): Path to the grayscale image file.
        pred (str): Path to the prediction mask file (should be same size as img).

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object with the overlay plot.

    Example:
        plot = overlay_img(
            img="/path/to/assembled_img.png",
            pred="/path/to/assembled_pred.png"
        )
    """
    #Read image and prediction as grayscale
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
    
    #Plot overlaid images
    plot = plt.figure(dpi=300)
    plt.imshow(img, cmap="gray")
    plt.imshow(pred, cmap="gray", alpha=0.4)
    return plot

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

def is_blurry(image_path, threshold=250):
    """
    Check if an image is blurry using the Laplacian variance method. Can tune threshold to adjust sensitivity. 
    Current threshold of 400 is a good default for SEM images.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold