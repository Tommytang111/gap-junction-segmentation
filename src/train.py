#Training script for cloud use
#July 2025
#Tommy Tang

#LIBRARIES
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm 
import copy
import wandb
import cv2
#Custom Libraries
from utils import seed_everything, worker_init_fn, create_dataset_splits
from models import TrainingDataset, TrainingDataset3D, UNet, GenDLoss

#Set Global Seed
GLOBAL_SEED = 40

#DATASET CLASS
#TrainingDataset from src/models.py

#MODEL CLASS
#Unet from src/models.py

#FUNCTIONS
def train(dataloader, model, loss_fn, optimizer, recall, precision, f1, device='cuda', three=False):
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
    
    for batch, (X, y) in tqdm(enumerate(dataloader), total=num_batches, desc="Training Batches"):
        X, y = X.to(device), y.to(device)
        
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Calculate metrics after converting predictions to binary
        if three:
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1).squeeze(1) #Remove channel dimension and depth dimension to match y
        else:
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
    
def validate(dataloader, model, loss_fn, recall, precision, f1, device='cuda', three=False):
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
        for X, y in tqdm(dataloader, desc="Validation Batches"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            #Calculate metrics
            if three:
                pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1).squeeze(1) #Remove channel dimension and depth dimension to match y
            else:
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

def test(model, dataloader, loss_fn, device='cuda', three=False):
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
        for X, y in tqdm(dataloader, desc="Test Evaluation"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # Calculate metrics
            if three:
                pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1).squeeze(1) #Remove channel dimension and depth dimension to match y
            else:
                pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1) #Remove channel dimension to match y
            
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

def wandb_init(run_name, epochs, batch_size, data, augmentations):
    """
    WandB Initialization
    """
    def extract_augmentations(augmentations):
        """Extract augmentation details from A.Compose."""
        aug_list = []
        for aug in augmentations.transforms:
            aug_list.append({aug.__class__.__name__ : aug.get_params})
        return aug_list

    # Extract augmentations
    train_aug_details = extract_augmentations(augmentations)
    
    wandb.login(key="04e003d2c64e518f8033ab016c7a0036545c05f5")
    
    run = wandb.init(project="gap-junction-segmentation", 
            entity="zhen_lab",
            name=run_name,
            dir="/home/tommy111/projects/def-mzhen/tommy111",
            reinit=True,
            config={
                "dataset": data,
                "model": "UNet3D-2D",
                "learning_rate": 0.01,
                "batch_size": batch_size,
                "epochs": epochs,
                "image_size": (512, 512),
                "loss_function": "Generalized Dice Loss",
                "optimizer": "SGD",
                "momentum": 0.9,
                "scheduler": "ReduceLROnPlateau",
                "augmentation": train_aug_details,
                "Unet upsample mode": "conv_transpose"
            }
    )
    return run
    
def main(run_name:str, data_dir:str, output_path:str, batch_size:int=16, epochs:int=200, seed:int=40, three=False, dropout=0):
    """
    Main function to run training, validation, and test loop.
    """
    #Set seed for reproducibility
    seed_everything(seed)

    # Create dataset splits (uncomment and run once to create the splits)
    if three:
        source_img_dir = f"{data_dir}/vols"
    else:
        source_img_dir = f"{data_dir}/imgs"
    source_gt_dir = f"{data_dir}/gts"
    output_base_dir = f"{data_dir}_split"

    #Create the splits (does not overwrite existing splits)
    dataset_paths = create_dataset_splits(source_img_dir, source_gt_dir, output_base_dir, random_state=seed, filter=True, three=three)

    #Set data augmentation type
    #ORIGINAL AUGMENTATION
    # train_augmentation = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.Affine(scale=(0.8, 1.2), rotate=360, translate_percent=0.15, shear=(-15, 15), p=0.9),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #     A.GaussNoise(p=0.3),
    #     A.Normalize(mean=0.0, std=1.0) if not three else A.NoOp(),
    #     A.Resize(512, 512) if not three else A.NoOp(),
    #     A.ToTensorV2() if not three else A.NoOp()
    # ], seed=GLOBAL_SEED, p=0.9)
    
    #AUGMENTATION THAT PRODUCES OVERFITTING
    # train_augmentation = A.Compose([
    #     A.Affine(scale=(0.9, 1.1), rotate=0, translate_percent=0.15, p=0.9),
    #     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    #     A.GaussNoise(var_limit=(1.0, 5.0), p=0.5),
    #     A.Normalize(mean=0.0, std=1.0),
    #     A.Resize(512, 512) if not three else A.NoOp(),
    #     A.ToTensorV2() if not three else A.NoOp()
    # ], seed=GLOBAL_SEED, p=0.9)
    
    #mean=137.0 (0.54 normalized), std=46.2 (0.18 normalized) for 516imgs_sem_adult
    
    #NEW AUGMENTATION
    train_augmentation = A.Compose([
        #A.HorizontalFlip(p=0.5),
        A.SquareSymmetry(p=0.5),
        A.CoarseDropout(num_holes_range=[1,8], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.2], fill=0, fill_mask=0, p=0.5), 
        A.Affine(scale=(0.9, 1.1), rotate=360, translate_percent=0.15, p=0.5),
        A.ElasticTransform(alpha=300, sigma=10, interpolation=cv2.INTER_LINEAR, p=0.5),
        A.Normalize(mean=0, std=1),
        A.ToTensorV2() if not three else A.NoOp()
    ], seed=GLOBAL_SEED)

    #For validation without augmentation
    valid_augmentation = A.Compose([
        #A.CenterCrop(height=256, width=256),
        #A.PadIfNeeded(min_height=512, min_width=512, position="center", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=0, std=1), #Specific to the dataset
        A.ToTensorV2()
    ])
    
    #For 3D validation without augmentation
    valid_augmentation3D = A.Compose([
        #A.CenterCrop(height=256, width=256), CURRENTLY UNNECESSARY, IS NOT THE SAME AS POSTPROCESSING CROP AND STITCH
        #A.PadIfNeeded(min_height=512, min_width=512, position="center", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=0, std=1) #Specific to the dataset
    ])
    
    #Initialize wandb
    run = wandb_init(run_name, epochs, batch_size, Path(data_dir).stem, train_augmentation)
    
    #Initialize datasets
    if three:
        train_dataset = TrainingDataset3D(
            volumes=dataset_paths['train']['vols'],
            labels=dataset_paths['train']['gts'],
            augmentation=train_augmentation,
            train=True,
        )

        valid_dataset = TrainingDataset3D(
            volumes=dataset_paths['val']['vols'],
            labels=dataset_paths['val']['gts'],
            augmentation=valid_augmentation3D,
            train=False
        )
        
        test_dataset = TrainingDataset3D(
            volumes=dataset_paths['test']['vols'],
            labels=dataset_paths['test']['gts'],
            augmentation=valid_augmentation3D,
            train=False
        )
        
    else:
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    #Set device and model
    device = torch.device("cuda")    
    model = UNet(three=three, n_channels=1, classes=1, up_sample_mode='conv_transpose', dropout=dropout).to(device)
    
    #Set loss function, optimizer, and scheduler
    loss_fn = GenDLoss()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    #Send evaluation metrics to device
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1 = BinaryF1Score().to(device)
    
    #Clear GPU memory before training
    torch.cuda.empty_cache()
    
    #Initialize training variables
    epochs = epochs
    best_train_f1 = 0.0
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        #Training
        train_loss, train_recall, train_precision, train_f1 = train(train_dataloader, model, loss_fn, optimizer, recall, precision, f1, three=three)

        #Validation
        val_loss, val_recall, val_precision, val_f1 = validate(valid_dataloader, model, loss_fn, recall, precision, f1, three=three)

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
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
        
        #Log best model state based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
            "best_train_f1": best_train_f1,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })

    print("Training Complete!")
    
    #Save the best logged model state
    model_save_path = f"{output_path}/{run.name}_{run.id}.pt"
    torch.save(best_model_state, model_save_path)
    print(f"Saved PyTorch Model to {model_save_path}")
    
    # Load the best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = test(model, test_dataloader, loss_fn, device, three=three)
    
    # Log test metrics to wandb
    wandb.log(test_metrics)
    
    wandb.finish()
        
if __name__ == "__main__":
    main(run_name="unet_3D2D_516vols_sem_adult",
         data_dir="/home/tommy111/projects/def-mzhen/tommy111/data/516vols_sem_adult",
         seed=40,
         epochs=200,
         batch_size=4,
         output_path="/home/tommy111/projects/def-mzhen/tommy111/models",
         three=True,  # Set to True for 3D-2D U-Net, False for 2D U-Net
         dropout=0) 
