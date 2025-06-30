#LIBRARIES
import torch
from torch import nn
from utilities import UpBlock, DownBlock, DoubleConv, GenDLoss, FocalLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import cv2
from tqdm import tqdm 
import copy
import wandb
import random
import shutil
from sklearn.model_selection import train_test_split
#Custom Libraries
from resize_image import resize_image

#DATASET CLASS
#Class can load any mask as long as the model corresponds to the mask type
class TrainingDataset(Dataset):
    def __init__(self, images, labels, masks=None, augmentation=None, data_size=(512, 512), train=True):
        self.image_paths = sorted([os.path.join(images, img) for img in os.listdir(images)])
        self.label_paths = sorted([os.path.join(labels, lbl) for lbl in os.listdir(labels)])
        self.mask_paths = sorted([os.path.join(masks, mask) for mask in os.listdir(masks)]) if masks else None
        self.augmentation = augmentation
        self.data_size = data_size
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #Read image, label, and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) if self.mask_paths else None
        
        #Apply resizing with padding if image is not expected size and then convert back to ndarray
        if (image.shape[0] != self.data_size[0]) or (image.shape[1] != self.data_size[1]): 
            image = np.array(resize_image(image, self.data_size[0], self.data_size[1], (0,0,0)))
            label = np.array(resize_image(label, self.data_size[0], self.data_size[1], (0,0,0)))
            if mask is not None:
                mask = np.array(resize_image(mask, self.data_size[0], self.data_size[1], (0,0,0)))

        #Convert mask/label to binary for model classification
        label[label > 0] = 1
        if mask is not None:
            mask[mask > 0] = 1
        
        #Apply augmentation if provided
        if self.augmentation and self.train:
            if mask is not None:
                #Use mask in augmentation
                augmented = self.augmentation(image=image, mask=label, label=mask)
                image = augmented['image']
                label = augmented['mask']
                mask = augmented['label']
            else:
                #Without mask
                augmented = self.augmentation(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']

        #Add entity recognition clause later if needed
        
        # Convert to tensors if not already converted from augmentation
        if not torch.is_tensor(image):
            image = ToTensor()(image).float()
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long()
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        elif mask is None:
            mask = torch.zeros_like(label)

        return image, label, mask

#MODEL CLASS
class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, residual=False, scale=False, spatial=False, dropout=0, classes=2):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three, spatial=False, residual=residual) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three, spatial=spatial, dropout=self.dropout, residual=residual) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256, spatial=spatial, dropout=self.dropout, residual=residual) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512, spatial=spatial, dropout=self.dropout, residual=residual) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,spatial=spatial, dropout=self.dropout, residual=residual)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, residual=residual) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1 if classes == 2 else classes, kernel_size=1)
        self.attend = attend
        if scale:
            self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling

    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)    
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        if self.three: x = x.squeeze(-3)   
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

#FUNCTIONS
def seed_everything(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    """
    Initialize the worker with a unique seed based on the worker ID.
    """
    seed = 42 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
<<<<<<< HEAD
def create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
=======
def create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir, train_size=0.8, val_size=0.1, test_size=0.1, random_state=14):
>>>>>>> 0d9e5d5 (Update)
    """
    Split a dataset into train, validation, and test sets.

    Args:
        source_img_dir: Directory containing all source images
        source_gt_dir: Directory containing all ground truth masks
        output_base_dir: Base directory where train/val/test folders will be created
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
        os.makedirs(os.path.join(directory, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'gts'), exist_ok=True)

    # Get all image filenames
    all_images = sorted([f for f in os.listdir(source_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # First split: train vs (val+test)
    train_images, remaining_images = train_test_split(
        all_images, 
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
                os.path.join(target_dir, 'imgs', img_name)
            )
            
            # Copy ground truth 
            gt_name = os.path.splitext(img_name)[0] + "_label.png"  
            shutil.copy(
                os.path.join(source_gt_dir, gt_name),
                os.path.join(target_dir, 'gts', gt_name)
            )

    print(f"Dataset split completed: {len(train_images)} training, {len(val_images)} validation, {len(test_images)} test images")

    return {
        'train': {'imgs': os.path.join(train_dir, 'imgs'), 'gts': os.path.join(train_dir, 'gts')},
        'val': {'imgs': os.path.join(val_dir, 'imgs'), 'gts': os.path.join(val_dir, 'gts')},
        'test': {'imgs': os.path.join(test_dir, 'imgs'), 'gts': os.path.join(test_dir, 'gts')}
    }
    
#Add augmentation functions
# Custom augmentation
def get_custom_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.8,1.2), rotate=360, translate_percent=0.15, shear=(-45, 45), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        A.Resize(512, 512),
        ToTensorV2()
    ])

# Light augmentation for gap junction segmentation
def get_light_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=0.0, std=1.0),  # For grayscale
        ToTensorV2()
    ])

# Medium augmentation
def get_medium_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.5
        ),
        A.ElasticTransform(
            alpha=1, 
            sigma=50, 
            alpha_affine=50, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
    
# Heavy augmentation
def get_heavy_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15, 
            scale_limit=0.3, 
            rotate_limit=25, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.6
        ),
        A.ElasticTransform(
            alpha=1, 
            sigma=50, 
            alpha_affine=50, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.4
        ),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(p=0.3),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
    
#Define training function
def train(dataloader, model, loss_fn, optimizer, recall, precision, f1, device='cuda'):
    """
    Training logic for the epoch.
    """
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    
    #Reset metrics for each epoch
    recall.reset()
    precision.reset()
    f1.reset()
    
    for batch, (X, y, _) in tqdm(enumerate(dataloader), total=num_batches, desc="Training Batches"):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Calculate metrics after converting predictions to binary
        pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1) #Remove channel dimension to match y
        
        #Update metrics
        recall.update(pred_binary, y)
        precision.update(pred_binary, y)
        f1.update(pred_binary, y)
        
        train_loss += loss.item()

    #Compute final metrics per epoch
    train_recall = recall.compute().item()
    train_precision = precision.compute().item()
    train_f1 = f1.compute().item()
    train_loss_per_epoch = train_loss / num_batches 
    
    return train_loss_per_epoch, train_recall, train_precision, train_f1
    
def validate(dataloader, model, loss_fn, recall, precision, f1, device='cuda'):
    """
    Validation logic for the epoch.
    """
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)
    
    #Reset metrics for each epoch
    recall.reset()
    precision.reset()
    f1.reset()
    
    with torch.no_grad():
        for X, y, _ in tqdm(dataloader, desc="Validation Batches"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
        #Calculate metrics
        pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1) #Remove channel dimension to match y
        
        #Update metrics
        recall.update(pred_binary, y)
        precision.update(pred_binary, y)
        f1.update(pred_binary, y)
            
        #Compute final metrics per epoch
        val_recall = recall.compute().item()
        val_precision = precision.compute().item()
        val_f1 = f1.compute().item()
        val_loss_per_epoch = test_loss / num_batches

    return val_loss_per_epoch, val_recall, val_precision, val_f1
    print(f"Avg loss: {test_loss:>7f}\n")

def test(model, dataloader, loss_fn, device='cuda'):
    """
    Evaluate the model on the test dataset
    
    Args:
        model: Trained PyTorch model
        test_dataloader: DataLoader for test dataset
        loss_fn: Loss function
        device: Device to run evaluation on
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)
    
    # Initialize metrics
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1 = BinaryF1Score().to(device)
    
    with torch.no_grad():
        for X, y, _ in tqdm(dataloader, desc="Test Evaluation"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # Calculate metrics
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1)
            
            # Update metrics
            recall.update(pred_binary, y)
            precision.update(pred_binary, y)
            f1.update(pred_binary, y)
            
    # Compute final metrics
    test_metrics = {
        "test_loss": test_loss / num_batches,
        "test_recall": recall.compute().item(),
        "test_precision": precision.compute().item(),
        "test_f1": f1.compute().item()
    }
    
    print(f"Test Results | Loss: {test_metrics['test_loss']:.4f}, "
          f"Recall: {test_metrics['test_recall']:.4f}, "
          f"Precision: {test_metrics['test_precision']:.4f}, "
          f"F1: {test_metrics['test_f1']:.4f}")
    
    return test_metrics

def wandb_init(run_name):
    """
    WandB Initialization
    """
    #wandb.login(key="04e003d2c64e518f8033ab016c7a0036545c05f5")
    run = wandb.init(project="gap-junction-segmentation", 
            entity="zhen_lab",
            name=run_name,
            dir="/home/tommy111/projects/def-mzhen/tommy111",
            reinit=True,
            config={
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 200,
                "image_size": (512, 512),
                "loss_function": "Generalized Dice Loss",
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau"
            }
    )
    return run
    
def main():
    """
    Main function to run training and validation loop.
    """
    #Initialize wandb
    run = wandb_init("unet_base_pooled_2695imgs_sem_dauer_2_516imgs_sem_adult")

    #Set seed for reproducibility
    seed_everything(14)

    # Create dataset splits (uncomment and run once to create the splits)
    source_img_dir = "/home/tommy111/projects/def-mzhen/tommy111/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult/imgs"
    source_gt_dir = "/home/tommy111/projects/def-mzhen/tommy111/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult/gts"
    output_base_dir = "/home/tommy111/projects/def-mzhen/tommy111/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult_split"

    #Create the splits (comment out after first run)
    dataset_paths = create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir)
    
    #Alternatively, if splits already exist, define paths manually
    # dataset_paths = {
    #     'train': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/train/imgs', 
    #               'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/train/gts'},
    #     'val': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/val/imgs', 
    #             'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/val/gts'},
    #     'test': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/test/imgs', 
    #              'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/test/gts'}
    # }

    # Create dataset splits (uncomment and run once to create the splits)
    source_img_dir = "/home/tommytang111/gap-junction-segmentation/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult/imgs"
    source_gt_dir = "/home/tommytang111/gap-junction-segmentation/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult/gts"
    output_base_dir = "/home/tommytang111/gap-junction-segmentation/data/pooled_2695imgs_sem_dauer_2_516imgs_sem_adult_split"
    
    #Create the splits (comment out after first run)
    dataset_paths = create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir)
    
    #Alternatively, if splits already exist, define paths manually
    # dataset_paths = {
    #     'train': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/train/imgs', 
    #               'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/train/gts'},
    #     'val': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/val/imgs', 
    #             'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/val/gts'},
    #     'test': {'imgs': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/test/imgs', 
    #              'gts': '/home/tommytang111/gap-junction-segmentation/data/pilot2_split/test/gts'}
    # }

    #Set data augmentation type
    train_augmentation = get_custom_augmentation()  # Change to get_medium_augmentation() or get_heavy_augmentation() as needed

    #For validation without augmentation
    valid_augmentation = A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
    
    #Initialize datasets
    train_dataset = TrainingDataset(
        images=dataset_paths['train']['imgs'],
        labels=dataset_paths['train']['gts'],
        augmentation=train_augmentation,
        train=True,
    )

    valid_dataset = TrainingDataset(
        images=dataset_paths['val']['imgs'],
        labels=dataset_paths['val']['gts'],
        augmentation=valid_augmentation,
        train=False
    )
    
    test_dataset = TrainingDataset(
        images=dataset_paths['test']['imgs'],
        labels=dataset_paths['test']['gts'],
        augmentation=valid_augmentation,
        train=False
    )

    #Load datasets into DataLoader
<<<<<<< HEAD
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
=======
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
>>>>>>> 0d9e5d5 (Update)
    
    #Set device and model
    device = torch.device("cuda")    
    model = UNet().to(device)
    
    #Set loss function, optimizer, and scheduler
    loss_fn = GenDLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    #Send evaluation metrics to device
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1 = BinaryF1Score().to(device)
    
    #Clear GPU memory before training
    torch.cuda.empty_cache()
    
    #Initialize training variables
    epochs = 200
    best_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        #Training
        train_loss, train_recall, train_precision, train_f1 = train(train_dataloader, model, loss_fn, optimizer, recall, precision, f1)

        #Validation
        val_loss, val_recall, val_precision, val_f1 = validate(valid_dataloader, model, loss_fn, recall, precision, f1)

        #Update learning rate scheduler
        scheduler.step(val_loss)
        
        #Print metrics
        print(f"Train | Loss: {train_loss:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}, F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}")
        print("-----------------------------")
        
        #Log best model state based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        #Log best model state based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
        #Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "val_f1": val_f1,
            "best_val_f1": best_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })

    print("Training Complete!")
    
    #Save the best logged model state
<<<<<<< HEAD
    model_save_path = f"/home/tommytang111/gap-junction-segmentation/models/{run.name}.pt"
=======
    model_save_path = f"/home/tommy111/projects/def-mzhen/tommy111/models/{run.name}_{run.id}.pt"
>>>>>>> 0d9e5d5 (Update)
    torch.save(best_model_state, model_save_path)
    print(f"Saved PyTorch Model to {model_save_path}")
    
    # Load the best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = test(model, test_dataloader, loss_fn, device)
    
    # Log test metrics to wandb
    wandb.log(test_metrics)
    
    wandb.finish()
        
if __name__ == "__main__":
    main()
