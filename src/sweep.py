#A script for hyperparameter sweeps using Weights & Biases
#To use, setup sweep_config with desired hyperparameters and call sweep() function with appropriate arguments.
#Tommy Tang
#July 2025

#Libraries
import sys
# Add parent directory to sys.path
parent_dir = '/home/tommytang111/gap-junction-segmentation/code/src'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import copy
import wandb
#Custom Libraries
from utils import seed_everything, worker_init_fn
from models import TrainingDataset, UNet, GenDLoss, FocalLoss

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Functions
def get_custom_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), rotate=10, translate_percent=0.15, shear = (-5, 5), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        A.Resize(512, 512),
        ToTensorV2()
    ])
    
def get_custom_augmentation2():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.8,1.2), rotate=360, translate_percent=0.15, shear=(-15, 15), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        A.Resize(512, 512),
        ToTensorV2()
    ])

def train(dataloader, model, loss_fn, optimizer, recall, precision, f1):
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    
    # Reset metrics for each epoch
    recall.reset()
    precision.reset()
    f1.reset()
    
    for batch, (X, y, _) in tqdm(enumerate(dataloader), total=num_batches, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        # Special handling for BCEWithLogitsLoss
        if y.dim() == 3:
            y = y.unsqueeze(1).float()
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate metrics after converting predictions to binary
        pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1)
        
        # Update metrics
        if y.dim() == 4 and y.size(1) == 1:
            y = y.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        recall.update(pred_binary, y)
        precision.update(pred_binary, y)
        f1.update(pred_binary, y)
        
        train_loss += loss.item()

    # Compute final metrics per epoch
    train_recall = recall.compute().item()
    train_precision = precision.compute().item()
    train_f1 = f1.compute().item()
    train_loss_per_epoch = train_loss / num_batches 
    
    return train_loss_per_epoch, train_recall, train_precision, train_f1

def validate(dataloader, model, loss_fn, recall, precision, f1):
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)
    
    # Reset metrics for each epoch
    recall.reset()
    precision.reset()
    f1.reset()
    
    with torch.no_grad():
        for X, y, _ in tqdm(dataloader, desc="Validation", leave=False):
            X, y = X.to(device), y.to(device)
            #Special handling for BCEWithLogitsLoss
            if y.dim() == 3:
                y = y.unsqueeze(1).float()
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # Calculate metrics
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze(1)
            
            # Update metrics
            if y.dim() == 4 and y.size(1) == 1:
                y = y.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            recall.update(pred_binary, y)
            precision.update(pred_binary, y)
            f1.update(pred_binary, y)
            
    # Compute final metrics per epoch
    val_recall = recall.compute().item()
    val_precision = precision.compute().item()
    val_f1 = f1.compute().item()
    val_loss_per_epoch = test_loss / num_batches

    return val_loss_per_epoch, val_recall, val_precision, val_f1

def sweep(data_dir:str, output_dir:str, seed:int=40, epochs:int=50):
    #Read WandB API key from secrets.txt
    with open("/home/tommytang111/gap-junction-segmentation/code/secrets.txt", "r") as file:
        lines = file.readlines()
        #WandB API key is on the fourth line
        wandb_api_key = lines[3].strip()
    
    #Initialize wandb run
    wandb.login(key=wandb_api_key)
    wandb.init(dir="/home/tommytang111/gap-junction-segmentation/wandb")
    
    #Get hyperparameters from wandb config
    config = wandb.config
    
    #Set seeds
    seed_everything(seed)
    
    #Get augmentation strategy from config, default to 'medium'
    aug_strategy = config.get('augmentation', 'custom2')

    if aug_strategy == 'custom1':
        train_aug = get_custom_augmentation()
    elif aug_strategy == 'custom2':
        train_aug = get_custom_augmentation2()

    valid_aug = A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()])

    #Initialize datasets with config batch size
    train_dataset = TrainingDataset(
        images=f"{data_dir}/train/imgs",
        labels=f"{data_dir}/train/gts",
        augmentation=train_aug,
        train=True
    )
    
    valid_dataset = TrainingDataset(
        images=f"{data_dir}/val/imgs",
        labels=f"{data_dir}/val/gts",
        augmentation=valid_aug,
        train=False
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    #Initialize model with config dropout
    model = UNet(dropout=config.dropout).to(device)
    
    #Loss function mapping
    if config.loss_function == "GenDLoss":
        loss_fn = GenDLoss()
    #elif config.loss_function == "BCEWithLogitsLoss":
        #loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    elif config.loss_function == "FocalLoss":
        loss_fn = FocalLoss(alpha=torch.Tensor([0.08, 0.92]), device=device)

    #Optimizer mapping
    if config.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    elif config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6
    )
    
    # Initialize metrics
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1 = BinaryF1Score().to(device)
    
    # Training loop
    torch.cuda.empty_cache()
    epochs = epochs
    best_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # Training
        train_loss, train_recall, train_precision, train_f1 = train(
            train_dataloader, model, loss_fn, optimizer, recall, precision, f1
        )
        
        # Validation
        val_loss, val_recall, val_precision, val_f1 = validate(
            valid_dataloader, model, loss_fn, recall, precision, f1
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train | Loss: {train_loss:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}, F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}")
        print("-----------------------------")

        # Log best model state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            # Save best model for this run
            model_path = f"{output_dir}/sweep_model_{wandb.run.id}.pt"
            
        # Log metrics to wandb
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
    torch.save(best_model_state, model_path)
    print("Saved PyTorch Model to ", model_path)
    torch.cuda.empty_cache()
    wandb.finish()

#Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'random', 'bayes'
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [8]
        },
        'optimizer': {
            'values': ['AdamW']
        },
        'loss_function': {
            'values': ['GenDLoss']
        },
        'dropout': {
            'values': [0, 0.1]
        },
        'augmentation': {
            'values': ['custom1', 'custom2']
        },
    }
}

if __name__ == "__main__":
    #Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="gap-junction-segmentation")
    print(f"Sweep ID: {sweep_id}")

    #Start the sweep agent 
    #wandb recommends calling agent with no arguments for sweep() function but I bypass this with a lambda function
    wandb.agent(sweep_id=sweep_id, function=lambda: sweep(data_dir="/home/tommytang111/gap-junction-segmentation/data/pilot1", 
                                                          output_dir="/home/tommytang111/gap-junction-segmentation/models/sweeps", 
                                                          seed=40,
                                                          epochs=50
                                                          )
    )