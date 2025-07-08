"""
Complete set of utility functions for Gap Junction Segmentation API.
Tommy Tang
June 2, 2025
"""

#LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.ndimage import label
from typing import Union
import os
import subprocess
from pathlib import Path
import cv2
from PIL import Image

#FUNCTIONS
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

def check_filtered(folder:str, filter_func=sobel_filter):
    """
    Checks all images in a folder using a specified filter function and reports those that should be excluded.

    Parameters:
        folder (str): Path to the folder containing images to check.
        filter_func (callable): A function that takes an image path and returns 'exclude' if the image should be excluded,
                                or any other value if it should be kept. Default is sobel_filter.

    Returns:
        None. Prints the names of excluded images and a summary count.
    """
    count = 0
    for img in os.listdir(folder):
        if filter_func(os.path.join(folder, img)) == 'exclude':
            print(f"Image {Path(img).name} is excluded.")
            count += 1

    print(f"Total excluded images: {count}/{len(os.listdir(folder))}")

def check_img_size(folder:str, target_size=(512, 512)):
    """
    Checks if all images in a folder are of a specified size.

    Parameters:
        folder (str): Path to the folder containing images to check.
        target_size (tuple): Expected size of the images as (height, width).

    Returns:
        None
    """
    count = 0
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        read_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if read_img is None:
            print(f"Could not read image: {img_path}")
            continue
        if read_img.shape != target_size:
            print(f"Image {img} does not match target size {target_size}. Found size: {read_img.shape}")
            count += 1
            
    print(f'Total images not matching size {target_size}: {count}/{len(os.listdir(folder))}')

def filter_pixels(img: np.ndarray, size_threshold: int = 8) -> np.ndarray:
    """
    Removes small non-zero pixel islands from a grayscale image.

    For each connected component (island) of non-zero pixels in the input image,
    this function sets all pixels in that component to zero if the component contains
    fewer than the number of pixels defined by size_threshold. Designed for use with grayscale images to filter out small
    annotation errors or noise.

    Parameters:
        img (np.ndarray): Input grayscale image as a NumPy array.
        size_threshold (int): Minimum size of pixel islands to keep (default: 8).

    Returns:
        np.ndarray: The filtered image with small pixel islands removed.
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
                if count < size_threshold:
                    filtered[y, x] = 0
    return filtered

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

def plot_3D(file:str, title:str='Title', xlab:str='X', ylab:str='Y', zlab:str='Z', elev:int=20, azim:int=90, figx:int=10, figy:int=7, dpi:int=None):
    """
    Plots a grayscale image as a 3D surface, where the Z axis represents pixel intensity.

    Parameters:
        file (str): Path to the image file to plot.
        title (str): Title of the plot (default: 'Title').
        xlab (str): Label for the X axis (default: 'X').
        ylab (str): Label for the Y axis (default: 'Y').
        zlab (str): Label for the Z axis (default: 'Z').
        elev (int): Elevation angle in the z plane for the 3D plot view (default: 20).
        azim (int): Azimuth angle in the x,y plane for the 3D plot view (default: 90).
        figx (int): Width of the figure in inches (default: 10).
        figy (int): Height of the figure in inches (default: 7).
        dpi (int, optional): Dots per inch for the figure. If None, uses default.

    Returns:
        None. Displays the 3D surface plot of the image.
    """
    #Load image as grayscale
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    #Get image dimensions
    x_size = img.shape[1]
    y_size = img.shape[0]
    
    #Create X, Y coordinate arrays
    X, Y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    
    fig = plt.figure(figsize=(figx, figy), dpi=dpi)
    ax = plt.axes(projection='3d')
    
    #Plot surface
    surface = ax.plot_surface(X, Y, img, cmap='magma', alpha=0.5)
    ax.set_title(title, y=0.90)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_zlim(0, 255)
    ax.invert_xaxis()
    ax.grid(True)
    ax.set_box_aspect(None, zoom=0.85)
    ax.view_init(elev=elev, azim=azim)
    #ax.contour3D(X, Y, img, 50, cmap='magma', alpha=1)
    #colorbar modifiers
    #surface.set_clim(0, 255)  # Set color limits for the surface
    #cbar_ax = fig.add_axes([0.8, 0.35, 0.02, 0.3])  
    #fig.colorbar(surface, cax=cbar_ax)
    plt.tight_layout()
    plt.show()

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
    
def sobel_filter(image_path, threshold_blur=35, threshold_artifact=25, verbose=False, apply_filter=False):
    """
    Assess image quality by evaluating sharpness and artifact level using the Sobel filter.

    This function computes the gradient magnitude of a grayscale image using the Sobel operator.
    It then uses the mean and standard deviation of the gradient magnitude to determine if the image
    should be excluded due to blurriness or excessive artifacts, or if it is suitable for further use.
    Thresholds can be adjusted for different datasets.

    Parameters:
        image_path (str): Path to the image file to be evaluated.
        threshold_blur (float): Mean gradient threshold below which the image is considered blurry.
        threshold_artifact (float): Standard deviation threshold below which the image is considered to have artifacts.
        verbose (bool): If True, prints the mean and standard deviation of the gradient magnitude.
        apply_filter (bool): If True, returns the mean and standard deviation instead of classification.

    Returns:
        str or tuple: If apply_filter is False, returns 'exclude' if the image is likely blurry or has artifacts,
                      otherwise returns 'ok'. If apply_filter is True, returns (mean_grad, std_grad).

    Raises:
        ValueError: If the image cannot be read from the provided path.

    Example:
        result = sobel_filter('/path/to/image.png')
        if result == 'ok':
            print("Image is suitable for use.")
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Compute Sobel gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    #Get image of pixel-wise sobel-filtered magnitudes
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    #Get mean and std of gradient magnitude
    mean_grad = grad_mag.mean()
    std_grad = grad_mag.std()

    if verbose:
        print(f"Image: {image_path}, \nMean Gradient: {mean_grad} \nStd Gradient: {std_grad}")

    if apply_filter:
        return grad_mag
    else:
        # These thresholds should be tuned per dataset depending on scale and contrast of images
        # Good EM images (for sem_adult and sem_dauer_2) have:
        # 1. mean_grad > 35 
        # 2. std_grad > 25
        # 3. std_grad < 1.1 * mean_grad (content to noise ratio)
        # 4. std_grad < 56 if mean_grad > 86
        
        if (mean_grad < threshold_blur) or (std_grad < threshold_artifact) or (std_grad > 1.1*mean_grad) or ((std_grad > 56) and (mean_grad < 86)):
            return 'exclude'
        else:
            return 'ok'
    
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

def view_label(file:str):
    """
    Displays a grayscale label, converting all non-zero pixels to 255 (white).

    Parameters:
        file (str): Path to the image file to display.

    Returns:
        None. Shows the processed image using matplotlib.
    """
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img[img != 0] = 255
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

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
