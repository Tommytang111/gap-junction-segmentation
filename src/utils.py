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
    
def split_img(img, offset=256, tile_size=512, names=False):
    """
    Split a full EM image into tiles of a specified size, with an optional offset.
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

def split_large_img(source_path, target_path):
    """
    Split a large EM image into smaller images of size 512x512.
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
            cv2.imwrite(f"{target_path}/{re.sub(r'.png$', r'', img)}_part{i+5}.png", split)
            
"""!python3 main_processing.py \
  --postprocessing \
  --imgs_dir "/mnt/e/Mishaal/sem_dauer_2/image_export" \
  --preds_dir "/home/tommytang111/results" \
  --output_dir "/home/tommytang111/assembled_results" \
  --img_template SEM_dauer_2_image_export_ \
  --seg_template SEM_dauer_2_image_export_ \
  --Smin 0 --Smax 51 \
  --Ymin 0 --Ymax 17 \
  --Xmin 0 --Xmax 19 \
  --offset 0"""

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