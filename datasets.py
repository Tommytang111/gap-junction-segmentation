import os
import cv2
import numpy as np;
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from resize_image import resize_image
from pathlib import Path
# import segmentation_models_pytorch as smp
# import albumentations as album
import joblib

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.ops import DropBlock2D, DropBlock3D
from collections import OrderedDict
from PIL import Image
import re, math, random
import pickle as p

from resnet import ResNet, BasicBlock

class CaImagesDataset(torch.utils.data.Dataset):

    """Calcium imaging images dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            dataset_dir, 
            finetune_dirs=[],
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512),
            mask_neurons=None,
            mask_mito=None,
            gen_gj_entities=False,
            split=0 #0 for train, 1 for valid
    ):
        prefix = "train" if not split else "valid"
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        masks_dir = os.path.join(dataset_dir, f"{prefix}_gts")
        
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts"))) if "DS" not in image_id and (True and self.centers_and_contours(cv2.imread(os.path.join(masks_dir, image_id)), filter_mode=True))]
        self.image_paths = [os.path.join(images_dir, os.path.split(image_id.replace("sem2dauer_gj_2d_training.vsseg_export_", "SEM_dauer_2_image_export_"))[-1]) for image_id in self.mask_paths if "DS" not in image_id]

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = []
            for neuron_id in self.image_paths:
                neuron_id = os.path.split(neuron_id)[-1]
                item = os.path.join(os.path.join(dataset_dir, f"{prefix}_neuro"), os.path.split(neuron_id.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_"))[-1].replace(".png.png", ".png"))
                self.neuron_paths.append(item)

        else: self.neuron_paths = None
        if finetune_dirs:
            for f_dir in finetune_dirs:
                t = [os.path.join(os.path.join(f_dir, "gts"), image_id) for image_id in sorted(os.listdir(os.path.join(f_dir, f"gts"))) if "DS" not in image_id]
                self.mask_paths += t
                self.image_paths += [os.path.join(os.path.join(f_dir, "imgs"), os.path.split(image_id)[-1]) for image_id in t]
        
        if mask_neurons and finetune_dirs: 
            for f_dir in finetune_dirs:
                self.neuron_paths += [os.path.join(os.path.join(f_dir, "neurons"), image_id) for image_id in self.mask_paths if "DS" not in image_id]

        if mask_mito:
            self.mito_mask = [os.path.join(os.path.join(dataset_dir, f"{prefix}_mito"), neuron_id.replace("png", "tiff") if not "tiny" in dataset_dir else neuron_id) for neuron_id in sorted(os.listdir(images_dir))]
            assert len(self.mito_mask) == len(self.image_paths), f"{len(self.mito_mask)} {len(self.image_paths)}"
        else:
            self.mito_mask=None
            
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
        self.gen_gj_entities=gen_gj_entities

    def centers_and_contours(self, gt, filter_mode=False):
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(gray, dtype=np.uint8)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
         # Calculate center of each contour
        centers, contour_arr = [], []
        for i, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                cnt = cnt.squeeze(1)
                cont_arr = torch.zeros(gray.shape)
                cont_arr[[cnt[:, 1], cnt[:, 0]]] = 1

                #check for intersection
                get_out = False
                for c in contour_arr:
                    if torch.count_nonzero((c +cont_arr) >= 2) != 0: 
                        get_out = True
                        break
                if get_out: continue # bad contour identification safeguard

                centers.append((cx, cy))
                contour_arr.append(cont_arr)    

                out = cv2.circle(out, (cx, cy), 5, 255, 2)
                out = cv2.drawContours(out, contours, i, color=255, thickness=2)
                
        if not len(centers) and filter_mode:
            return False
        elif filter_mode: return True

        contour_arr = torch.stack(contour_arr, axis=0)
        assert contour_arr.shape[0] == len(centers)

        center_indices = [list(range(len(centers)))] + [list(list(zip(*centers))[0]), list(list(zip(*centers))[1])]
        center_arr = torch.zeros((len(centers), gt.shape[0], gt.shape[1]))
        try:
            center_arr[list(range(len(centers))), list(list(zip(*centers))[0]), list(list(zip(*centers))[1])] = 1
        except:
            print(center_indices, center_arr.shape)
            raise Exception

        #sanity check
        if len(torch.unique(contour_arr.sum(dim=0))) > 2:
            #save the image somehwere
            cv2.imwrite("example.png", out)
            cv2.imwrite("example_gt.png", gt)
        assert len(torch.unique(contour_arr.sum(dim=0))) <=2, f"{torch.unique(contour_arr.sum(dim=0))} {torch.unique(contour_arr)} {(len(centers))}"


        return center_arr.to(dtype=torch.float32), contour_arr.to(dtype=torch.float32)

    def __getitem__(self, i):
        
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        try:
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
        except Exception as e:
            print(self.image_paths[i])
            raise Exception(self.image_paths[i])
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
        if self.gen_gj_entities: centers, contours = self.centers_and_contours(cv2.imread(self.mask_paths[i]))

        # make sure each pixel is 0 or 255
        
        mask_labels, counts = np.unique(mask, return_counts=True)
        # if (len(mask_labels)>2):
        #     print("More than 2 labels found for mask")
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        mask_ref = mask.copy()
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
            image = v2.RandomAutocontrast()(image)
            image = v2.GaussianBlur(5)(image)

        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        # img_size = 512
        # width, height = self.image_dim
        # max_dim = max(img_size, width,    )
        # pad_left = (max_dim-width)//2
        # pad_right = max_dim-width-pad_left
        # pad_top = (max_dim-height)//2
        # pad_bottom = max_dim-height-pad_top
        # _transform.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
        #                                 padding_mode='edge'))
        # _transform.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))  

        mask = transforms.Compose(_transform)(mask)
        if len(mask_labels) == 1:
            mask[:] = 0
        else:
            mask[mask != 0] = 1
        ont_hot_mask = mask
        # ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()

        _transform_img = _transform.copy()
        # _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)

        #get the coresponding neuron mask
        if self.neuron_paths: neurons = cv2.cvtColor(cv2.imread(self.neuron_paths[i]), cv2.COLOR_BGR2GRAY)
        if self.mito_mask: mitos = np.array(Image.open(self.mito_mask[i]))
        
        if not self.gen_gj_entities:
            return image, ont_hot_mask, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        else:
            return image, ont_hot_mask, centers, contours, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class SectionsDataset(torch.utils.data.Dataset):
    
    def __init__(
            self, 
            dataset_dir ,
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512),
            chain_length=4,
            mask_neurons=None,
            mask_mito=None,
            split=0 #0 if train 1 if valid
    ):    
        prefix = "train" if not split else "valid"
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(os.path.join(dataset_dir, f"{prefix}_gts"), image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts")))]

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = [path.replace("imgs", "neuro") for path in self.image_paths]
        else: self.neuron_paths = None

        # if mask_mito:
        #     self.mito_mask = [os.path.join(mask_mito, neuron_id) for neuron_id in sorted(os.listdir(images_dir))]
        # else:
        #     self.mito_mask=None
        self.mito_mask = None
            
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
        self.chain_length = chain_length

        # self.make_chain_paths()
        print("Dataset has {} samples".format(len(self.image_paths)))

    def make_chain_paths(self):
        chain_img_paths, chain_seg_paths, chain_neuron_paths, chain_mito_paths = [], [], [] if self.neuron_paths else None, [] if self.neuron_paths else None
        for i in range(len(self.image_paths)):
            imgs, segs, neurons, mitos = [], [], [], []
            img_path, seg_path = self.image_paths[i], self.mask_paths[i]
            layer = re.findall(r's\d\d\d_', img_path)[0]

            skip = False
            for j in range(self.chain_length):
                next_layer_1 = "s" + ("00" if int(layer[1:-1]) < 9-j else "0") + str(int(layer[1:-1])+j) + "_"
                img_f_1 = img_path.replace(layer, next_layer_1)
                seg_f_1 = seg_path.replace(layer, next_layer_1)
                if self.neuron_paths:
                    neuron_path = self.neuron_paths[i]
                    neurons_f_1 = neuron_path.replace(layer, next_layer_1)
                if self.mito_mask:
                    mito_path = self.mito_mask[i]
                    mito_f_1 = mito_path.replace(layer, next_layer_1)

                if not os.path.isfile(img_f_1) or not os.path.isfile(seg_f_1) or (self.neuron_paths is not None and not os.path.isfile(neurons_f_1)) \
                    or (self.mito_mask is not None and not os.path.isfile(mito_f_1)) : 
                    skip = True
                    break
                # print("Not skip!")
                imgs.append(img_f_1)
                segs.append(seg_f_1)
                if self.neuron_paths: neurons.append(neurons_f_1)
                if self.mito_mask: mitos.append(mito_f_1)

            if skip: continue

            chain_img_paths.append(imgs)
            chain_seg_paths.append(segs)

            if self.neuron_paths: chain_neuron_paths.append(neurons)
            if self.mito_mask: chain_mito_paths.append(mitos)  
            # assert len(sum(chain_img_paths, start=[]))%3 == 0

        self.image_paths, self.mask_paths = chain_img_paths, chain_seg_paths
        if self.neuron_paths: self.neuron_paths = chain_neuron_paths
        if self.mito_mask: self.mito_mask = chain_mito_paths

    def __getitem__(self, i):

        # directory is arranged as before, current, future1, future2

        images = sorted(os.listdir(self.image_paths[i]))
        masks = sorted(os.listdir(self.mask_paths[i]))
        

        img = []
        ns = []
        if self.neuron_paths: neurons = sorted(os.listdir(self.neuron_paths[i]))
        if self.mito_mask: mitos = []

        mask = cv2.cvtColor(cv2.imread(os.path.join(self.mask_paths[i], masks[1])), cv2.COLOR_BGR2GRAY)

        for j in range(4):
            im = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
            img.append(im)
            if self.neuron_paths:
                neuron = cv2.cvtColor(cv2.imread(os.path.join(self.neuron_paths[i], neurons[j])), cv2.COLOR_BGR2GRAY)
                ns.append(neuron)
        
        image = np.stack(img, axis=0)
        if self.neuron_paths: neurons = np.stack(ns, axis=0)
        mitos=[]
        

        # make sure each pixel is 0 or 255
        
        mask_labels = np.unique(mask)
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        
        # apply augmentations
        flippant = random.random()
        if self.augmentation and flippant < 0.25:
            if flippant < 0.55:
                #random rotation plus gaussian blur
                sample = self.transform1(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            elif flippant < 0.85:
                #horizontal and vertical flip + gaussian blur
                sample = self.transform2(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                #everything
                sample = self.transform3(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        mask = transforms.Compose(_transform)(mask)
        if len(mask_labels) == 1:
            mask[:] = 0
        else:
            mask[mask != 0] = 1
        ont_hot_mask = mask

        _transform_img = _transform.copy()
        image = transforms.Compose(_transform_img)(image)

        image = torch.permute(image, (1, 2, 0))  

        # ont_hot_mask = torch.permute(ont_hot_mask, (1, 2, 0))    
            
        return image.unsqueeze(0), ont_hot_mask, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], torch.from_numpy(mitos[np.newaxis, :]) if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, aug=False):
        images = os.path.join(dataset_dir, "imgs")
        gts = os.path.join(dataset_dir, "gts")
        mems = os.path.join(dataset_dir, "neurons")

        self.mask_paths = []
        self.image_paths = []
        self.neuron_paths = []
        names = []

        print("--Filtering")
        # for image_id in tqdm(sorted(os.listdir(images))):
        #     if "DS" in image_id:
        #         continue
        #     image_id = os.path.join(gts, image_id)
        #     gt = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2GRAY)
        #     gt[gt == 2] = 0
        #     gt[gt == 15] = 0
        #     if len(np.unique(gt)) < 2: continue
        #     image_name = os.path.split(image_id)[-1]
        #     names.append(image_name)
        #     self.image_paths.append(os.path.join(images, image_name))
        #     self.neuron_paths.append(os.path.join(mems, image_name))
        #     self.mask_paths.append(os.path.join(gts, image_id))
        
        #COMMENTED BELOW WAS REMOVED BY TOMMY
        #with open("/home/mishaalk/projects/def-mzhen/mishaalk/gapjncsegmentation/names.p", "rb") as  f:
        #    names = p.load(f)
        for image_name in sorted(os.listdir(images)):
            if "SEM" in image_name:
                continue
            self.image_paths.append(os.path.join(images, image_name))
            self.neuron_paths.append(os.path.join(mems, image_name))
            self.mask_paths.append(os.path.join(gts, image_name))
        
        self.augmentation = aug
        

        
    def __getitem__(self, i):

        # directory is arranged as before, current, future1, future2
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY)
        neurons = cv2.cvtColor(cv2.imread(self.neuron_paths[i]), cv2.COLOR_BGR2GRAY) 
        

        # make sure each pixel is 0 or 255
        # print(np.unique(mask))
        
        mask_labels = np.unique(mask)
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
            image = v2.RandomAutocontrast()(image)
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        mask = transforms.Compose(_transform)(mask)
        if len(mask_labels) == 1:
            mask[:] = 0
        else:
            mask[mask != 0] = 1
        ont_hot_mask = mask

        _transform_img = _transform.copy()
        image = transforms.Compose(_transform_img)(image)

        # image = torch.permute(image, (1, 2, 0))     
        if len(image.shape) >= 5: print(image.shape)
        # print( torch.from_numpy(neurons[np.newaxis, :]).unsqueeze(0).shape)
            
        return image, ont_hot_mask, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], []

    def __len__(self):
        # return length of 
        return len(self.image_paths)
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_dir ,
            image_dim = (512, 512),
            td=False,
            membrane=False
    ):      
        self.image_paths = [os.path.join(Path(dataset_dir) / "imgs", image_id) for image_id in sorted(os.listdir(Path(dataset_dir) / "imgs")) if "DS" not in image_id]
        self.mask_paths = [os.path.join(Path(dataset_dir) / "gts", image_id) for image_id in sorted(os.listdir(Path(dataset_dir) / "gts")) if "DS" not in image_id]
        if membrane:
            self.membrane_paths = [os.path.join(membrane, image_id) for image_id in sorted(os.listdir(dataset_dir)) if "DS" not in image_id]
        else: self.membrane_paths = None
        self.td = td
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

    def __getitem__(self, i):
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        if not self.td:
            try:
                image0 = cv2.imread(self.image_paths[i])
                #Padding and Cropping
                if (image0.shape[-1] != 512) or (image0.shape[-2] != 512): #Current image should be 2D: (lxw)
                    image0 = resize_image(image0, 512, 512, (0,0,0))
                image = cv2.cvtColor(np.array(image0), cv2.COLOR_BGR2GRAY) 
                mask = np.zeros_like(image)#cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
                if self.membrane_paths: memb = cv2.imread(self.membrane_paths[i].replace("sem_dauer_2_em_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_"), -1) == 0
            except Exception as e:
                print(self.image_paths[i])
                raise Exception(self.image_paths[i])
        else:
            images = sorted(os.listdir(self.image_paths[i]))
            img, mem = [], []
            for j in range(4):
                im = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
                if self.membrane_paths: 
                    memb = cv2.cvtColor(cv2.imread(os.path.join(self.membrane_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
                    mem.append(memb)
                img.append(im)
            
            image = np.stack(img, axis=0)
            memb = np.stack(mem, axis=0)
            
        """ REVISIT FOR MASK AND MEMBRANE WHEN NEEDED
        if image.shape[-1] != 512: 
            image = torch.zeros(image.shape[:-1]+(512,))
            mask = torch.zeros(image.shape[:-1]+(512,))
            if self.membrane_paths: memb = torch.zeros(memb.shape[:-1]+(512,))
        if image.shape[-2] != 512: 
            image = torch.zeros(image.shape[:-2]+(512,512))
            mask = torch.zeros(image.shape[:-1]+(512,))
            if self.membrane_paths: memb = torch.zeros(memb.shape[:-2]+(512,512))
        """

        #Transform to tensor
        _transform = []
        _transform.append(transforms.ToTensor()) 
        if self.membrane_paths: memb = transforms.Compose(_transform)(memb)
        _transform_img = _transform.copy()
        # _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)
        mask = transforms.Compose(_transform_img)(mask)
        if self.td: 
            image = torch.permute(image, (1, 2, 0)) 
            if self.membrane_paths: memb = torch.permute(memb, (1, 2, 0)) 
        else: 
            image = image.squeeze(0) 
            if self.membrane_paths: memb = memb.squeeze(0)
            mask = mask.squeeze(0)
        #Add batch dimension to image, mask
        return image.unsqueeze(0), mask.unsqueeze(0), image.unsqueeze(0) if not self.membrane_paths else memb.unsqueeze(0).float(), image.unsqueeze(0)

class ExtendDataset(torch.utils.data.Dataset):

    """Calcium imaging images dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            dataset_dir,  
            image_dim = (512, 512),
            split=0 #0 for train, 1 for valid
    ):
        print("--extend dataset--")
        prefix = "train" if not split else "valid"
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        masks_dir = os.path.join(dataset_dir, f"{prefix}_gts")
        
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir)) if "DS" not in image_id]

        self.image_paths = [os.path.join(images_dir, os.path.split(image_id)[-1]) for image_id in self.mask_paths if "DS" not in image_id]

        self.image_dim = image_dim
        self.split = split

    def __getitem__(self, i):

        # read the directories
        s = int(re.findall("s\d\d\d", self.image_paths[i])[0][1:])
        img_files = os.listdir(self.image_paths[i])
        mask_files = os.listdir(self.mask_paths[i])

        # read the input EM image and its target label
        try:
            s = re.findall(r"s\d\d\d", self.image_paths[i])[0][1:]
            if int(s) <= 50: extra= ".png.png"
            else: extra = ".png"
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], os.path.split(self.image_paths[i])[-1].replace("_before", "").replace("_after", "")+extra)), cv2.COLOR_BGR2GRAY) # each pixel is 
            img_files.remove(os.path.split(self.image_paths[i])[-1].replace("_before", "").replace("_after", "")+extra)
        except Exception as e:
            print(self.image_paths[i])
            raise Exception(e)
            
        mask = cv2.cvtColor(cv2.imread(os.path.join(self.mask_paths[i], os.path.split(self.mask_paths[i])[-1].replace("_before", "").replace("_after", "")+extra)), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
        mask_files.remove(os.path.split(self.mask_paths[i])[-1].replace("_before", "").replace("_after", "")+extra)
        mask_files.remove(os.path.split(self.mask_paths[i])[-1].replace("_before", "").replace("_after", "")+"og.png")

        #read the previous/next section's prediction and EM
        pred_image = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], img_files[0])), cv2.COLOR_BGR2GRAY)
        pred_mask = cv2.cvtColor(cv2.imread(os.path.join(self.mask_paths[i], mask_files[0])), cv2.COLOR_BGR2GRAY)

        mask[mask == 0] = 0
        mask[mask == 255] = 1
        pred_mask[pred_mask == 0] = 0
        pred_mask[pred_mask == 255] = 1
        # apply augmentations
        # if random.random() >= 0.5:
        #     # image, mask = self.augmentation(image, mask)
        #     # image = v2.RandomAutocontrast()(image)
        #     augmentation = v2.Compose([
        #         v2.RandomHorizontalFlip(p=0.5),
        #         v2.RandomVerticalFlip(p=0.5),
        #         v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.5),
        #         v2.ColorJitter(contrast=(0, 0.5))
        #     ])
        #     image, mask, pred_image, pred_mask = augmentation(image, mask, pred_image, pred_mask)
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        mask = torch.from_numpy(mask)
        ont_hot_mask = mask
        

        _transform_img = _transform.copy()
        image = transforms.Compose(_transform_img)(image)
        pred_mask = torch.from_numpy(pred_mask)
        pred_image = transforms.Compose(_transform_img)(pred_image)
        
        # print(torch.unique(image), torch.unique(pred_image), torch.unique(mask).to(torch.float32), torch.unique(pred_mask))
        if self.split == -1: return image, pred_image, pred_mask.to(torch.float32).unsqueeze(0), mask.to(torch.float32).unsqueeze(0)
        return image, pred_image, pred_mask.to(torch.float32).unsqueeze(0), mask.to(torch.float32).unsqueeze(0)
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class FinetuneFNDataset(CaImagesDataset):
    def __init__(
            self, 
            dataset_dir, 
            finetune_dirs=[],
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512),
            mask_neurons=None,
            mask_mito=None,
            gen_gj_entities=False,
            split=0):
        
        # super(FinetuneFNDataset, self).__init__(dataset_dir=dataset_dir, finetune_dirs=finetune_dirs, augmentation=augmentation, preprocessing=preprocessing, image_dim=image_dim, mask_neurons =mask_neurons, mask_mito = mask_mito, gen_gj_entities=gen_gj_entities, split=split)

        # print(len(self.image_paths))
        prefix = "train" if not split else "valid"
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        masks_dir = os.path.join(dataset_dir, f"{prefix}_gts")
        
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts")))]
        self.image_paths = [os.path.join(images_dir, os.path.split(image_id.replace("sem2dauer_gj_2d_training.vsseg_export_", "SEM_dauer_2_image_export_"))[-1]) for image_id in self.mask_paths ]
        print(len(self.image_paths), images_dir, masks_dir)

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = []
            for neuron_id in self.image_paths:
                neuron_id = os.path.split(neuron_id)[-1]
                item = os.path.join(os.path.join(dataset_dir, f"{prefix}_neuro"), os.path.split(neuron_id.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_"))[-1].replace(".png.png", ".png"))
                self.neuron_paths.append(item)

        else: self.neuron_paths = None
        if finetune_dirs:
            for f_dir in finetune_dirs:
                t = [os.path.join(os.path.join(f_dir, "gts"), image_id) for image_id in sorted(os.listdir(os.path.join(f_dir, f"gts"))) if "DS" not in image_id]
                self.mask_paths += t
                self.image_paths += [os.path.join(os.path.join(f_dir, "imgs"), os.path.split(image_id)[-1]) for image_id in t]
        
        if mask_neurons and finetune_dirs: 
            for f_dir in finetune_dirs:
                self.neuron_paths += [os.path.join(os.path.join(f_dir, "neurons"), image_id) for image_id in self.mask_paths if "DS" not in image_id]

        if mask_mito:
            self.mito_mask = [os.path.join(os.path.join(dataset_dir, f"{prefix}_mito"), neuron_id.replace("png", "tiff") if not "tiny" in dataset_dir else neuron_id) for neuron_id in sorted(os.listdir(images_dir))]
            assert len(self.mito_mask) == len(self.image_paths), f"{len(self.mito_mask)} {len(self.image_paths)}"
        else:
            self.mito_mask=None
            
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
        self.gen_gj_entities=gen_gj_entities
    
    def __getitem__(self, i):
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        try:
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
        except Exception as e:
            print(self.image_paths[i])
            raise Exception(self.image_paths[i])
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
        if self.gen_gj_entities: centers, contours = self.centers_and_contours(cv2.imread(self.mask_paths[i]))

        mask_ref = mask.copy()
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
            image = v2.RandomAutocontrast()(image)
            image = v2.GaussianBlur(5)(image)

        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        # mask = transforms.Compose(_transform)(mask)
        mask = torch.from_numpy(mask).to(torch.float32)
        ont_hot_mask = mask
        # ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()

        _transform_img = _transform.copy()
        # _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)

        #get the coresponding neuron mask
        if self.neuron_paths: neurons = cv2.cvtColor(cv2.imread(self.neuron_paths[i]), cv2.COLOR_BGR2GRAY)
        if self.mito_mask: mitos = np.array(Image.open(self.mito_mask[i]))
        
        if not self.gen_gj_entities:
            return image, ont_hot_mask.unsqueeze(0), torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        else:
            return image, ont_hot_mask, centers, contours, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)