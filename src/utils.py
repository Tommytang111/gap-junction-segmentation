"""
Complete set of utility functions for Gap Junction Segmentation API.
Tommy Tang
Last Updated: Sept 1, 2025
"""

#LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from typing import Union
from torch import nn
import torch
import os
import time
import shutil
import subprocess
from pathlib import Path
import cv2
from tqdm import tqdm
from PIL import Image
import re
import random
from sklearn.model_selection import train_test_split
from scipy.ndimage import label

#DEPENDENCY FUNCTIONS
def assemble_img(tile_dir:str, template:str, suffix:str, s_range:range, x_range:range, y_range:range, crop:bool=False):
    """
    Assemble tiled prediction images into full sections.

    Reads tiles from tile_dir following the filename pattern:
        template + f"s{section:03d}_Y{y}_X{x}" + suffix

    For each section index in s_range the function:
    - loads every tile for y in y_range and x in x_range (grayscale),
    - optionally center-crops each tile from 512x512 to 256x256 (tile[128:384, 128:384]),
    - stitches tiles horizontally per row and concatenates rows vertically to form a full section,
    - collects the assembled section and its output filename (template + out_infix + suffix).

    Parameters:
        tile_dir (str): Directory containing tile images.
        template (str): Filename prefix template (e.g. "SEM_adult_image_export_").
        suffix (str): Filename suffix (e.g. "_pred.png").
        s_range (range): Range of section indices to assemble.
        x_range (range): Range of X tile indices.
        y_range (range): Range of Y tile indices.
        crop (bool): Whether or not to center crop the tile

    Returns:
        tuple:
            - assembled_sections (list[np.ndarray]): List of assembled section images (uint8 arrays).
            - assembled_sections_names (list[str]): Corresponding output filenames for each assembled section.

    Raises:
        FileNotFoundError: If any expected tile file is missing in tile_dir.
    """
    assembled_sections = []
    assembled_sections_names = []
    #Make sure at least one directory is provided
    for s in s_range:
        s_acc = []
        for y in y_range:
            y_acc = []
            for x in x_range:
                #Create filename infix
                infix = f"s{str(s).zfill(3)}_Y{y}_X{x}"
                #Assemble tile name
                tile_path = os.path.join(tile_dir, template + infix + suffix)
                if not os.path.isfile(tile_path):
                    raise FileNotFoundError(f"Missing image {template + infix + suffix}")
                
                #Step 1: Read tile
                tile = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
                #Step 2: Center crop tile 512x512 -> 256x256
                if crop:
                    tile = tile[128:384, 128:384]
                #Step 3: Append to row accumulator
                y_acc.append(tile)
            #Step 4: Concatenate tiles in row accumulator and append to section accumulator
            #Check if we have any tiles
            if y_acc:
                s_acc.append(np.concatenate(y_acc, axis=1))
        #Step 5: Concatenate tiles in section accumulator to create an entire section
        #Check if we have any rows
        if s_acc:
            assembled_section = np.concatenate(s_acc, axis = 0)
            assembled_sections.append(assembled_section)
            #Create filename infix
            out_infix = f"s{str(s).zfill(3)}"
            assembled_sections_names.append(template + out_infix + suffix)

    return assembled_sections, assembled_sections_names
            
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
        apply_filter (bool): If True, returns the sobel filtered image instead of performing classification.

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
        
def split_img(img:np.ndarray, offset=False, overlap=False, tile_size=512, names=False, xysizes=False):
    """ 
    Splits an image into tiles of a specified size, with an optional offset to remove borders.
    
    Parameters:
        img (np.ndarray): Input image as a NumPy array.
        offset (int): Number of pixels to crop from each border before splitting.
        overlap (bool): If True, splits into overlapping tiles.
        tile_size (int): Size of each square tile.
        names (bool): If True, also returns a list of tile names.
        xysizes (bool): If True, also returns a list of largest sizes in each dimension (ex. max(Y)=20, max(X)=20, max(S)=20)

    Returns:
        list: List of image tiles (np.ndarray).
        list (optional): List of tile names if names=True.
        list (optional): List of largest sizes in each dimension if xysizes=True.

    Example:
        tiles, names = split_img(image, offset=0, overlap=True, tile_size=512, names=True)
    """
    if overlap:
        stride = tile_size // 2
    
    if offset:
        img = img[offset:-offset, offset:-offset]
    imgs = []
    names_list = []
    for i in range(0, img.shape[0], stride if overlap else tile_size):
        for j in range(0, img.shape[1], stride if overlap else tile_size):
            imgs.append(img[i:i+tile_size, j:j+tile_size])
            names_list.append(f"Y{i//stride if overlap else i//tile_size}_X{j//stride if overlap else j//tile_size}")

    size_list = [i//stride + 1 if overlap else i//tile_size + 1, j//stride + 1 if overlap else j//tile_size + 1]

    if names and xysizes:
        return (imgs, names_list, size_list)
    elif names:
        return(imgs, names_list)
    elif xysizes:
        return(imgs, size_list)
    else:
        return imgs

#FUNCTIONS
def assemble_imgs(img_dir:str, gt_dir:str, pred_dir:str, save_dir:str, s_range:range, x_range:range, y_range:range, s_size:tuple, img_templ:str=None, seg_templ:str=None, overlap:bool=False):
    """
    Assemble per-tile images, ground-truth masks and/or prediction tiles into full-section images and save to disk.

    Reads tiled files from the provided directories using filename templates and the index ranges,
    stitches tiles into complete sections, and writes the assembled sections into subfolders
    under save_dir ("imgs", "gts", "preds").

    Overlap condition: overlap should be set depending on how tiles were created from sections during preprocessing.
    If overlapping tiles were created, set overlap=True to ensure predictions are stitched properly. 
    Do not provide img_dir or gt_dir in this case as only pred_dir should be stitched with overlap.
    Final assembled predictions will be cropped to original section size due to overlap stitching resulting in slightly 
    larger output dimensions.

    Parameters
    - img_dir (str | None): Directory containing image tiles to assemble (pass None to skip).
    - gt_dir  (str | None): Directory containing ground-truth tiles to assemble (pass None to skip).
    - pred_dir (str | None): Directory containing prediction tiles to assemble (pass None to skip).
    - save_dir (str): Directory where assembled outputs will be written. Subfolders "imgs", "gts", "preds" are created as needed.
    - s_range (range): Range of section indices to assemble (e.g. range(0, 51)).
    - x_range (range): Range of X tile indices per row (e.g. range(0, 20)).
    - y_range (range): Range of Y tile indices per column (e.g. range(0, 18)).
    - s_size (tuple): Tuple of (height, width) of the full section size in pixels (e.g. (4608, 5120)).
    - img_templ (str | None): Filename template / prefix used for tiles (e.g. "SEM_adult_image_export_"). Used to build filenames.
    - seg_templ (str | None): Filename template / prefix used for segmentation (gt) tiles.
    - overlap (bool): If True, prediction tiles are cropped prior to stitching (set when tiles were created with overlap).

    Returns
    - None. Assembled images are written to disk under save_dir.

    Notes
    - Relies on helper function assemble_image.
    - This function asserts that at least one of img_dir, gt_dir or pred_dir is provided.
    - assemble_img is used internally and may raise FileNotFoundError if expected tiles are missing.
    - cv2.imwrite return values are not returned; check the filesystem and permissions if files do not appear.
    """
    #Make sure at least one directory is provided
    assert img_dir or gt_dir or pred_dir, "At least one directory must be provided"

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Assemble images/masks/predictions and save
    if img_dir:
        imgs, img_names = assemble_img(tile_dir=img_dir, template=img_templ, suffix=".png", s_range=s_range, x_range=x_range, y_range=y_range)
        check_output_directory(os.path.join(save_dir, "imgs"), clear=True)
        for img, name in zip(imgs, img_names): 
            cv2.imwrite(os.path.join(save_dir, "imgs", name), img)
    if gt_dir:
        gts, gt_names = assemble_img(tile_dir=gt_dir, template=seg_templ, suffix="_label.png", s_range=s_range, x_range=x_range, y_range=y_range)
        check_output_directory(os.path.join(save_dir, "gts"), clear=True)
        for gt, name in zip(gts, gt_names):
            cv2.imwrite(os.path.join(save_dir, "gts", name), gt)
    if pred_dir:
        if overlap:
            preds, pred_names = assemble_img(tile_dir=pred_dir, template=img_templ, suffix="_pred.png", s_range=s_range, x_range=x_range, y_range=y_range, crop=True)
        else:
            preds, pred_names = assemble_img(tile_dir=pred_dir, template=img_templ, suffix="_pred.png", s_range=s_range, x_range=x_range, y_range=y_range)
        check_output_directory(os.path.join(save_dir, "preds"), clear=True)
        for pred, name in zip(preds, pred_names):
            pred = pred[0:s_size[0], 0:s_size[1]]
            cv2.imwrite(os.path.join(save_dir, "preds", name), pred)

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

def check_output_directory(path:str, clear:bool=True) -> None:
    """
    Ensures an output directory exists and is empty.

    If the specified directory exists, a message will print and all files within it are deleted (optional).
    If the directory does not exist, it is created.

    Parameters:
        path (str): The path to the directory to check or create.
        clear (bool): If True, clears the directory if it exists. If False, does not clear existing files.

    Returns:
        None

    Example:
        check_directory("/path/to/output_folder")
    """
    if os.path.exists(path):
        print(f"Output directory already exists: {path}")
        if clear:
            print(f"Clearing output directory: {path}")
            try:
                # Use shutil.rmtree instead of shell rm command - much more reliable
                for entry in os.scandir(path):
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path, ignore_errors=True)
                    else:
                        try:
                            os.unlink(entry.path)
                        except FileNotFoundError:
                            pass  # Another process may have deleted it
            except Exception as e:
                print(f"Warning: Error clearing directory contents: {e}")
    else:
        # Create with exist_ok to handle race conditions
        os.makedirs(path, exist_ok=True)
        
def _ensure_empty_dir(path:str) -> None:
    """
    Recursively removes a directory if it exists and recreates it empty.
    Uses a rename-and-delete strategy that's more reliable with concurrent access.
    """
    if os.path.exists(path):
        try:
            # Rename strategy - much safer for parallel operations
            tmp_path = f"{path}.{os.getpid()}.{time.time()}.old"
            os.rename(path, tmp_path)
            # Create the new directory
            os.makedirs(path, exist_ok=True)
            # Remove the old directory in background
            try:
                shutil.rmtree(tmp_path, ignore_errors=True)
            except:
                pass  # We don't care if cleanup fails
        except OSError:
            # If rename failed, fall back to recreating
            try:
                shutil.rmtree(path, ignore_errors=True)
                # Sleep a tiny bit to let filesystem catch up
                time.sleep(0.1)
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not fully clear directory {path}: {e}")
                # Last resort - ensure it exists
                os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
        
def create_dataset_2d(imgs_dir, output_dir, seg_dir=None, img_size=512, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, create_overlap=False, seg_ignore=None, test=False):
    """
    Create a 2D training/validation dataset by tiling full-section EM images (and optional annotations).

    This function reads full-size images from imgs_dir (and optional segmentations from seg_dir),
    splits each section into tiles of size img_size, and writes the tiles to output_dir under
    subfolders "imgs" and, if not test mode, "gts". Optionally creates overlapping tiles
    (half-stride) when create_overlap=True, and supports copying additional auxiliary directories.

    Key behaviors
    - If test is False, seg_dir must be provided (unless a custom image_to_seg_name_map handles mapping).
      Ground-truth tiles will be converted to binary masks (non-zero -> 255).
    - If create_overlap is True, tiles are created with a stride of img_size//2 (useful for prediction stitching).
    - If add_dir and add_dir_maps are provided, additional per-image data directories are tiled and saved.
    - If output_dir exists it will be removed and recreated (guarded by a printed warning).
    - Returns tiling metadata useful for later stitching: (max_y, max_x, max_section_size, last_image_shape).

    Parameters
    - imgs_dir (str): Directory containing full-section images.
    - output_dir (str): Directory to write tiled dataset. Subfolders "imgs" and "gts" (if not test) are created.
    - seg_dir (str | None): Directory containing segmentation images. Required for non-test mode unless a mapping function is used.
    - img_size (int): Tile size in pixels (square). Default 512.
    - image_to_seg_name_map (callable | None): Function mapping an image filename to its segmentation filename.
      If None and test==False a default mapping is assumed (replacing 'img' with 'seg').
    - add_dir (list[str] | None): Additional directories whose files should be included (each will get its own output subfolder).
    - add_dir_maps (dict | None): Mapping from each add_dir path to a function mapping an image filename to that add_dir filename.
    - create_overlap (bool): If True, create overlapping tiles with stride img_size//2.
    - seg_ignore (iterable | None): Iterable of label values to set to 0 in ground-truth before binarization.
    - test (bool): If True, only create image tiles (no ground-truth output).

    Returns
    - tuple: (max_y, max_x, max_section_size, image_shape)
        - max_y (int): maximum number of tile rows produced for a section.
        - max_x (int): maximum number of tile columns produced for a section.
        - max_section_size (int): number of processed sections (index of last processed + 1).
        - image_shape (tuple): shape of the last processed full-section image (H, W).

    Notes and examples
    - Example:
        max_y, max_x, n_sections, img_shape = create_dataset_2d(
            imgs_dir='/data/sections',
            output_dir='/data/tiles',
            seg_dir='/data/labels',
            img_size=512,
            create_overlap=False,
            test=False
        )
    - Be careful: calling this on a large dataset will delete output_dir if it exists.
    - When using create_overlap=True, downstream stitching must account for the cropping strategy used during assembly.
    """
    #Check additional directories
    assert (add_dir is None and add_dir_maps is None or add_dir is not None and add_dir_maps is not None) 
    if not test and image_to_seg_name_map is None:
        print("WARNING: No image to segmentation name mapping provided, assuming the default naming convention")
        image_to_seg_name_map = lambda x: x.replace('img', 'seg')

    #Safely remove/recreate the output directory
    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, recreating it")
        _ensure_empty_dir(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    #Make subdirectories
    os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
    if not test:
        os.makedirs(os.path.join(output_dir, "gts"), exist_ok=True)
        if add_dir:
            for i in (add_dir):
                os.makedirs(os.path.join(output_dir, os.path.split(i)[-1]), exist_ok=True)

    imgs = sorted(os.listdir(imgs_dir))

    def split_subroutine(img, gt, overlap=False):
        #Split the sections into tiles
        img_imgs, img_names, max_img_sizes = split_img(img, overlap=overlap, tile_size=img_size, names=True, xysizes=True)
        #Split the ground truth sections into masks
        if not test: 
            gt_imgs = split_img(gt, overlap=overlap, tile_size=img_size)
            add_data = [] if add_dir else None
            if add_dir:
                for j in range(len(add_dir)):
                    dat = cv2.imread(os.path.join(add_dir[j], add_dir_maps[add_dir[j]](imgs[i])))
                    dat = split_img(dat, overlap=overlap, tile_size=img_size)
                    add_data.append(dat)
        #Write the images and masks to the output directory
        for j in range(len(img_imgs)):
            cv2.imwrite(os.path.join(output_dir, f"imgs/{imgs[i].replace('.png', '_'+img_names[j])}.png"), img_imgs[j])
            #Write the ground truth images if they exist
            if not test:
                cv2.imwrite(os.path.join(output_dir, f"gts/{imgs[i].replace('.png', '_'+img_names[j])}.png"), gt_imgs[j])
                if add_dir:
                    for k in range(len(add_data)):
                        assert cv2.imwrite(os.path.join(output_dir, f"{os.path.split(add_dir[k])[-1]}/{imgs[i].replace('.png', '_'+img_names[j])}.png"), add_data[k][j])

        #Get how many Y and X tiles were created
        return max_img_sizes[0], max_img_sizes[1] #Y, X
        
    for i in tqdm(range(len(imgs))):
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]), 0)
        
        if not test: 
            gt = cv2.imread(os.path.join(seg_dir, image_to_seg_name_map(imgs[i])), 0)

            #make changes in the gt
            for ig in seg_ignore:
                gt[gt == ig] = 0

            gt[gt!=0] = 255
        else: gt = None
        
        #Create and write tiles for every image in dataset, also returns how many Y and X tiles were created
        if create_overlap:
            max_y, max_x = split_subroutine(img, gt, overlap=True)  
        else:
            max_y, max_x = split_subroutine(img, gt)

        max_section_size = i + 1
        
    return max_y, max_x, max_section_size, img.shape
            
#Example Usage
#gs = None if args.add_dir is None else {i: lambda x: x.replace(args.img_template, args.add_dir_templates[j]) for j, i in enumerate(args.add_dir)}
#create_dataset_2d_from_full(args.imgs_dir, args.output_dir, seg_dir=args.seg_dir, img_size=args.img_size, image_to_seg_name_map= f, add_dir=args.add_dir, add_dir_maps=gs, seg_ignore=args.seg_ignore, create_overlap=args.create_overlap, test=args.test)
            
def create_dataset_3d(flat_dataset_dir, output_dir, window=(0, 1, 0, 0), image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, depth_pattern=r's\d\d\d', test=False):
    """
    Function to create a 3d dataset from a 2d, aka flat, dataset
    @param flat_dataset_dir: the directory of the flat dataset
    @param output_dir: the directory to output the 3d dataset
    @param window: the context window to use for the 3d dataset, only one dimension can be set to 1
    @param image_to_seg_name_map: the mapping from image to segmentation name
    @param add_dir: the additional directories to include in the dataset
    @param add_dir_maps: the mapping from image to additional data directory
    @param depth_pattern: the pattern to match the depth
    @param test: whether to run in test mode - just an images dataset
    @return: None (creates and saves your dataset to specified directory)
    """
    assert window.count(1) == 1, "Only one dimension can be set to 1"

    imgs = os.listdir(os.path.join(flat_dataset_dir, "imgs"))

    flat_imgs, flat_segs, flat_adds = [], [], []
    seq_imgs, seq_segs, seq_adds = [], [], []

    min_depth = min([int(re.findall(depth_pattern, img)[0][1:]) for img in imgs])
    max_depth = max([int(re.findall(depth_pattern, img)[0][1:]) for img in imgs])

    central_pos = window.index(1)

    def helper_for_another(img, name_map):
        temp_seq = []
        for i in range(len(window)):
            if i < central_pos:
                temp_seq.append(get_another(name_map(img), i-central_pos))
            elif i == central_pos:
                temp_seq.append(name_map(img))
            else:
                temp_seq.append(get_another(name_map(img), i-central_pos))
        return temp_seq

    for img in tqdm(imgs):
        depth = int(re.findall(depth_pattern, img)[0][1:])
        if depth < min_depth + central_pos or depth > max_depth - len(window) + central_pos+1: continue
                
        # flat_masks += [get_mask(img)]
        if not test: flat_segs += [image_to_seg_name_map(img)]

        img = (os.path.join(flat_dataset_dir, "imgs", img))
        flat_imgs += [img]

        if not test: seq_segs.append(helper_for_another(img, image_to_seg_name_map))
        seq_imgs.append(helper_for_another(img, lambda x: x))
        
        if not test:
            for i in add_dir_maps:
                flat_adds.append(add_dir_maps[i](img))
                seq_adds.append(helper_for_another(img, add_dir_maps[i]))

    for i in tqdm(range(len(seq_imgs))):
        os.makedirs(os.path.join(output_dir, "imgs", os.path.split(seq_imgs[i][1])[-1][:-4]))
        if not test:
            os.makedirs(os.path.join(output_dir, "gts", os.path.split(seq_imgs[i][1])[-1][:-4]))
            for j in add_dir_maps:
                os.makedirs(os.path.join(output_dir, os.path.split(j)[-1], os.path.split(seq_imgs[i][1])[-1][:-4]))

        for j in range(4):
            shutil.copy(os.path.join(flat_dataset_dir, "imgs", seq_imgs[i][j]), os.path.join(output_dir, "imgs", os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_imgs[i][j])[-1]))
            if not test:
                shutil.copy(os.path.join(flat_dataset_dir, "gts", seq_segs[i][j]), os.path.join(output_dir, "gts", os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_segs[i][j])[-1]))
                for k in add_dir:
                    shutil.copy(os.path.join(flat_dataset_dir, os.path.split(k)[-1], seq_adds[i][j]), os.path.join(output_dir, os.path.split(k)[-1], os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_adds[i][j])[-1]))

#Example Usage
#gs = None if args.add_dir is None else {i: lambda x: x.replace(args.img_template, args.add_dir_templates[j]) for j, i in enumerate(args.add_dir)}
#output_dir = args.output_dir if not args.make_twoD else args.output_dir+"_3d"
#create_dataset_3d(args.flat_dataset_dir, output_dir, depth_pattern=r's\d\d\d', window=args.window, test=args.test, image_to_seg_name_map=f, add_dir_maps=gs, add_dir=args.add_dir)

def create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir, filter=False, train_size=0.8, val_size=0.1, test_size=0.1, random_state=40, three=False):
    """
    Split a dataset into train, validation, and test sets.

    Args:
        source_img_dir: Directory containing all source images
        source_gt_dir: Directory containing all ground truth masks
        output_base_dir: Base directory where train/val/test folders will be created
        filter: Applies sobel_filter to exclude poor images
        train_size, val_size, test_size: Proportions for the splits (should sum to 1)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with paths to the created datasets
    """
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'val')
    test_dir = os.path.join(output_base_dir, 'test')

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(directory, 'imgs'), exist_ok=True) if not three else os.makedirs(os.path.join(directory, 'vols'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'gts'), exist_ok=True)

    # Get all image filenames
    all_images_or_vols = sorted([f for f in os.listdir(source_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if not three else sorted([f for f in os.listdir(source_img_dir) if f.endswith(('.npy', '.npz'))])
    all_images_or_vols_copy = all_images_or_vols.copy()

    # If filtering is enabled, apply the sobel filter to each image  
    if filter and not three:
        count = 0
        for img_name in tqdm(all_images_or_vols_copy, desc="Filtering images"):
            img_path = os.path.join(source_img_dir, img_name)
            if sobel_filter(img_path) == 'exclude':
                all_images_or_vols_copy.remove(img_name)
                count += 1
                
        print(f"Filtered {count} images out of {len(all_images_or_vols)} based on Sobel filter criteria.")
    
    # First split: train vs (val+test)
    train_images, remaining_images = train_test_split(
        all_images_or_vols_copy, 
        train_size=train_size, 
        random_state=random_state
    )

    # Second split: val vs test (from the remaining)
    val_ratio = val_size / (val_size + test_size)
    val_images, test_images = train_test_split(
        remaining_images, 
        train_size=val_ratio, 
        random_state=random_state
    )

    # Copy the files to their respective directories
    for image_list, target_dir in [
        (train_images, train_dir), 
        (val_images, val_dir), 
        (test_images, test_dir)
    ]:
        for img_name in image_list:
            # Copy image
            shutil.copy(
                os.path.join(source_img_dir, img_name),
                os.path.join(target_dir, 'imgs', img_name) if not three else os.path.join(target_dir, 'vols', img_name)
            )
            
            # Copy ground truth 
            gt_name = os.path.splitext(img_name)[0] + "_label.png"  
            shutil.copy(
                os.path.join(source_gt_dir, gt_name),
                os.path.join(target_dir, 'gts', gt_name)
            )

    print(f"Dataset split completed: {len(train_images)} training, {len(val_images)} validation, {len(test_images)} test images")

    key = 'imgs' if not three else 'vols'
    return {
        'train': {key: os.path.join(train_dir, key), 'gts': os.path.join(train_dir, 'gts')},
        'val': {key: os.path.join(val_dir, key), 'gts': os.path.join(val_dir, 'gts')},
        'test': {key: os.path.join(test_dir, key), 'gts': os.path.join(test_dir, 'gts')}
    }
    
def filter_by_overlay(image_folder, mask_folder, output_folder):
    """
    Displays each image with its segmentation mask and overlay and allows the user to manually filter images.

    For every image in the specified folder, this function overlays the corresponding mask (colored blue)
    on the image and displays the result. The user is prompted to accept or reject each image.
    Accepted images and their masks are copied to the output folder in separate directories.

    Parameters:
        image_folder (str): Path to the folder containing images.
        mask_folder (str): Path to the folder containing segmentation masks.
        output_folder (str): Path to the folder where accepted images and masks will be saved.

    Returns:
        None
    """
    #Get images
    imgs = os.listdir(image_folder)
    
    #Check if output folder exists, if not create it
    check_output_directory(output_folder, clear=False)
    check_output_directory(Path(output_folder) / "img", clear=False)
    check_output_directory(Path(output_folder) / "gts", clear=False)
    
    for img in imgs:
        img_mask = re.sub(r'.png$', r'_label.png', str(img))
        img_path = os.path.join(image_folder, img)
        mask_path = os.path.join(mask_folder, img_mask)
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask from (0 and 1) to (0 and 255)
        mask[mask > 0] = 255
        
        # Create an overlay of the mask_copy on top of base image in blue
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_rgb[mask_rgb[:, :, 0] == 255] = [0, 0, 255]
        overlay = cv2.addWeighted(image, 1, mask_rgb, 0.5, 0)
        
        # Display the original image, mask, and overlay
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Truth')
        plt.subplot(133)
        plt.imshow(overlay, cmap='gray' )
        plt.title('Overlay')
        plt.show()
        
        user_input = input(f"Do you want to accept image {Path(img).name}? (Y/N): ")
        
        # Save the image and mask if user accepts
        if user_input.lower() == 'y' or user_input.lower() == 'yes':
            output_img_path = os.path.join(output_folder, "img", img)
            output_mask_path = os.path.join(output_folder, "gts", img_mask)
            subprocess.run(f"cp {img_path} {output_img_path}", shell=True)
            subprocess.run(f"cp {mask_path} {output_mask_path}", shell=True)
            print(f"Transferred image and mask for {img} to {output_folder}")

def filter_pixels(img:np.ndarray, size_threshold:int=8) -> np.ndarray:
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
    labeled, _ = label(img > 0, structure=structure)
    # For each pixel, check if its component has at least size_threshold pixels
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

def overlay_img(img:str, pred:str, alpha:float=0.4) -> plt.figure:
    """
    Overlay a prediction mask on top of a grayscale image for visualization.

    This function loads a grayscale image and a prediction mask, then overlays the prediction
    on the image with transparency for easy visual inspection.

    Parameters:
        img (str): Path to the grayscale image file.
        pred (str): Path to the prediction mask file (should be same size as img)
        alpha (float): Defines transparency of overlay.

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
    plt.imshow(pred, cmap="gray", alpha=alpha)
    return plot

def plot_3D_intensity_projection(file:str, title:str='Title', xlab:str='X', ylab:str='Y', zlab:str='Z', elev:int=20, azim:int=90, figx:int=10, figy:int=7, dpi:int=None):
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

def predict_multiple_models(model1_path, model2_path, model3_path, data_dir):
    """
    Compare predictions from three different UNet models on a randomly selected image. 
    Use as a base template and edit to suit your visualization needs.
    
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
    
def seed_everything(seed: int = 40):
    """
    Set the random seed for reproducibility.
    """
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
def single_image_inference(image:np.ndarray, model_path:str, model, augmentation=None):
    """
    Performs inference on a single grayscale image using a trained PyTorch model.
    
    This function loads a trained model from a checkpoint, preprocesses the input image,
    runs inference, and returns a binary segmentation mask.
    
    Args:
        image (np.ndarray): Input grayscale image as a NumPy array with shape (H, W).
                           Expected to have pixel values in range [0, 255].
        model_path (str): Path to the saved PyTorch model checkpoint (.pt file).
        model: PyTorch model instance (e.g., UNet) to load the state dict into.
               The model architecture should match the saved checkpoint.
        augmentation (callable, optional): Optional augmentation function to apply to the image.
    
    Returns:
        np.ndarray: Binary segmentation mask as uint8 NumPy array with shape (H, W).
                   Values are 0 (background) or 1 (foreground/gap junction).
    
    Note:
        - Input image is automatically normalized to [0, 1] range.
        - Uses sigmoid activation with 0.5 threshold for binarization.
        - Model is automatically set to evaluation mode.
    
    Example:
        >>> import cv2
        >>> from models import UNet
        >>> 
        >>> # Load image and model
        >>> image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
        >>> model = UNet()
        >>> 
        >>> # Run inference
        >>> mask = single_image_inference(image, 'model.pt', model, valid_augmentation)
        >>> 
        >>> # Save result
        >>> cv2.imwrite('mask.png', mask * 255)
    """
    #Setup model
    model = model
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda"if torch.cuda.is_available else 'cpu') #Send to gpu
    model.eval() 
   
    #Prepare image
    if augmentation:
        augmented = augmentation(image=image) # Augmentation does normalization + standardization and converts to tensor
        image = augmented['image']
        image = image.unsqueeze(0) # Add shape: (1, 1, H, W) for batch and channel
    else:
        image = torch.ToTensor()
        image = image.unsqueeze(0).unsqueeze(0)

    #Inference
    with torch.no_grad():
        pred = model(image.to('cuda')) 
        pred = nn.Sigmoid()(pred) >= 0.5 #Binarize with sigmoid activation function
        pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype("uint8") #Convert from tensor back to image
    
    return pred
    
def single_volume_inference(volume:np.ndarray, model_path:str, model, augmentation=None):
    """
    Performs inference on a single volume using a trained PyTorch model.
    
    This function loads a trained model from a checkpoint, preprocesses the input volume,
    runs inference, and returns a binary segmentation mask.
    
    Args:
        image (np.ndarray): Input grayscale image as a NumPy array with shape (H, W).
                           Expected to have pixel values in range [0, 255].
        model_path (str): Path to the saved PyTorch model checkpoint (.pt file).
        model: PyTorch model instance (e.g., UNet) to load the state dict into.
               The model architecture should match the saved checkpoint.
        augmentation (callable, optional): Optional augmentation function to apply to the image.
    
    Returns:
        np.ndarray: Binary segmentation mask as uint8 NumPy array with shape (H, W).
                   Values are 0 (background) or 1 (foreground/gap junction).
    
    Note:
        - Input image is automatically normalized to [0, 1] range.
        - Uses sigmoid activation with 0.5 threshold for binarization.
        - Model is automatically set to evaluation mode.
    
    Example:
        >>> import cv2
        >>> from models import UNet
        >>> 
        >>> # Load image and model
        >>> image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
        >>> model = UNet()
        >>> 
        >>> # Run inference
        >>> mask = single_image_inference(image, 'model.pt', model, valid_augmentation)
        >>> 
        >>> # Save result
        >>> cv2.imwrite('mask.png', mask * 255)
    """
    #Setup model
    model = model
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda"if torch.cuda.is_available else 'cpu') #Send to gpu
    model.eval() 
   
    #Prepare volume
    if augmentation:
        #Make additional targets dict
        additional_targets = {}
        for i in range(1, volume.shape[0]):
            target_key = f'image{i}'
            additional_targets[target_key] = 'image'
            
        #Update albumentations pipeline with additional targets for all slices in volume
        augmentation.add_targets(additional_targets)

        #Prepare data dictionary with all slices, adding an extra channel dimension at the end
        #Note: albumentations Compose is supposed to add a channel dimension automatically and then remove it after 
        #augmentation, but it keeps crashing the script so I do it manually here.
        aug_data = {'image': volume[0][..., None]}  # First slice as main image
        for i in range(1, volume.shape[0]):
            target_key = f'image{i}'
            aug_data[target_key] = volume[i][..., None]

        # Apply augmentation once to all slices
        augmented = augmentation(**aug_data)

        print("Shape of augmented image:", augmented['image'].shape)

        # Reconstruct volume from augmented slices
        augmented_slices = [np.squeeze(augmented['image'], 0)]  # First slice, remove channel dimension
        for i in range(1, volume.shape[0]):
            augmented_slices.append(np.squeeze(augmented[f'image{i}'], 0))

        volume = np.stack(augmented_slices, axis=0)
        volume = torch.from_numpy(volume.astype(np.float32))
        volume = volume.unsqueeze(0).unsqueeze(0)          # Add channel and batch dimension: (1, 1, D, H, W)
    else:
        volume = torch.ToTensor()
        volume = volume.unsqueeze(0).unsqueeze(0)

    #Inference
    with torch.no_grad():
        print("Shape of volume before model:", volume.shape)
        pred = model(volume.to('cuda')) 
        pred = nn.Sigmoid()(pred) >= 0.5 #Binarize with sigmoid activation function
        pred = pred.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy().astype("uint8") #Convert from tensor back to image
    
    return pred

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
    
def worker_init_fn(worker_id):
    """
    Initialize the worker with a unique seed based on the worker ID.
    """
    seed = GLOBAL_SEED + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

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
            
def zoom_out_and_pad(image: np.ndarray) -> np.ndarray:
    """
    Downsamples a 512x512 image by a factor of 2 (to 256x256), 
    pastes it centered onto a black 512x512 image, and returns the result.

    Parameters:
        image (np.ndarray): Input 512x512 image (grayscale or RGB).

    Returns:
        np.ndarray: 512x512 image with the downsampled image centered and black padding.
    """
    # Downsample image to 256x256
    downsampled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Create a black 512x512 background
    if len(image.shape) == 2:  # Grayscale
        padded = np.zeros((512, 512), dtype=image.dtype)
    else:  # Color
        padded = np.zeros((512, 512, image.shape[2]), dtype=image.dtype)
    
    # Compute top-left corner for centering
    y_offset = (512 - 256) // 2
    x_offset = (512 - 256) // 2
    
    # Paste downsampled image onto black background
    padded[y_offset:y_offset+256, x_offset:x_offset+256] = downsampled
    
    return padded
