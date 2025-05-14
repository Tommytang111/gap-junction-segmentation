# import packages
import os
import cv2
import numpy as np;
import random, tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import segmentation_models_pytorch as smp
# import albumentations as album
import joblib
import torchvision.ops.focal_loss as focal
from torchvision.transforms import v2

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from utilities import *
# from models import *
from datasets import *
# import segmentation_models_pytorch.utils.metrics
import wandb
import random
import torch
import time
from tqdm import tqdm
import signal
import sys, csv

model_folder = "/mnt/e/Mishaal/GapJunction/models/"
sample_preds_folder = "home/tommytang111/results/"
table, class_labels = None, None #wandb stuff

torch.autograd.set_detect_anomaly(True)
DATASETS = {
    "new": "/mnt/e/Mishaal/sem_dauer_2/final_jnc_only_split", 
    "tiny": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_tiny_jnc_only128",
    "new3d": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split3d", 
    "test": "/mnt/e/Mishaal/sem_dauer_2/jnc_only_dataset_test", 
    "extend": r"/home/mishaalk/scratch/gapjunc/train_datasets/0_50_extend_train",
    "new_rwt": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split_rwt"
}
                
def make_dataset_new(dataset_dir, aug=False, neuron_mask=False, mito_mask=False, chain_length=False, gen_gj_entities=False, finetune_dirs=[]):
    """
    Makes a preprocessed dataset from an existing directory. Directory name must have "extend", "test", 
    """
    try:
        height, width = cv2.imread(os.path.join(x_new_dir, os.listdir(x_new_dir)[0])).shape[:2]
    except:
        height, width = 512, 512

    # Get train and val dataset instances
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4)
        v2.RandomZoomOut(fill=0, side_range=(1,2), p=0.5)
        v2.RandomAffine(degrees=0, scale=(1.0, 2.0), fill=0)
    ])
    if chain_length:
        train = SectionsDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            chain_length=chain_length,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask
        )
        valid = SectionsDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            chain_length=chain_length,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask, 
            split=1
        )
    elif "extend" in dataset_dir:
        train = ExtendDataset(dataset_dir)
        valid = ExtendDataset(dataset_dir, split=1)
    elif "test" in dataset_dir:
        train = DebugDataset(dataset_dir)
        valid = None
    else:
        dset_cls = CaImagesDataset if "rwt" not in dataset_dir else FinetuneFNDataset
        train = dset_cls(
            dataset_dir,
            finetune_dirs=finetune_dirs,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask,
            gen_gj_entities=gen_gj_entities
        )
        valid = dset_cls(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask, 
            split=1,
            gen_gj_entities=gen_gj_entities
        )

    return train, valid

def write_valid_imgs(batch, batch_gt, batch_no, epoch):
    global model_folder
    sample_folder = model_folder + "images"
    os.makedirs(sample_folder, exist_ok=True)

    for im in range(batch.size(0)):
        assert cv2.imwrite(os.path.join(sample_folder, f"epoch{epoch}_b{batch_no}_{im}.png"), batch[im].cpu().expand(3, -1, -1).permute(1,2,0).detach().numpy() * 255)
        assert cv2.imwrite(os.path.join(sample_folder, f"epoch{epoch}_b{batch_no}_{im}_gt.png"), batch_gt[im].cpu().expand(3, -1, -1).permute(1,2,0).detach().numpy() * 255)


def train_loop(model, train_loader, criterion, optimizer, valid_loader=None, mem_feat=False, epochs=30, decay=None, gen_gj_entities=True, fn_rwt=False):
    global table, class_labels, model_folder, DEVICE, args
    
    print(f"Using device: {DEVICE}")
    model_name = "model5"
    def sigint_handler(sig, frame):
        if table is not None:
            print("logging to WANDB")
            wandb.log({"Table" : table})
            joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGUSR1, sigint_handler)

    recall_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(gt[gt == 1])
    precision_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(pred[pred == 1])

    for epoch in range(epochs):
        pbar = tqdm(total=len(train_loader), position=0, leave=True)
        for i, data in enumerate(train_loader):
            # pbar.set_description("Progress: {:.2%}".format(i/len(train_loader)))
            if not gen_gj_entities: inputs, labels, neuron_mask, mito_mask = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
            else:
                inputs, labels, label_centers, label_contours, pad_mask, neuron_mask, mito_mask = data
                label_centers, label_contours, pad_mask = label_centers.to(DEVICE), label_contours.to(DEVICE), pad_mask.to(DEVICE)

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if neuron_mask != []: neuron_mask = neuron_mask.to(DEVICE)
            if mito_mask != []: mito_mask = mito_mask.to(DEVICE)

            pred = model(inputs) if not mem_feat else model(inputs, neuron_mask.to(torch.float32))

            if args.pred_mem:
                assert neuron_mask != []
                neuron_mask = neuron_mask.long()
                neuron_mask[neuron_mask == 0] = 2
                neuron_mask[neuron_mask == 1] = 0
                labels = labels + neuron_mask
                labels[labels == 3] = 1 # make it a gapjnc
                labels = F.one_hot(labels.long()).to(torch.float32).permute((0, 1, 4, 2, 3))
                loss_f_ = F.cross_entropy
            else: loss_f_ = F.binary_cross_entropy_with_logits

            
            if args.focalweight == 0:
                loss = criterion(pred.squeeze(1), labels.squeeze(1), neuron_mask.squeeze(1) if neuron_mask != [] and not mem_feat else [], mito_mask.squeeze(1) if mito_mask != [] else [], model.s1, model.s2, loss_fn=loss_f_)
            else: loss = criterion(pred.squeeze(1), labels.squeeze(1) if not gen_gj_entities else (label_centers, label_contours, pad_mask), neuron_mask.squeeze(1) if neuron_mask != [] and not mem_feat else [], mito_mask.squeeze(1) if mito_mask != [] else [], loss_fn = loss_f_, fn_reweight=fn_rwt)
            
            loss.backward() # calculate gradients (backpropagation)
            
            # log some metrics
            # print(torch.unique(torch.argmax(pred.squeeze(1), dim=1), return_counts=True))
            wandb.log({"train_precision": precision_f(pred >= 0, labels != 0)}) if not args.pred_mem else wandb.log({"train_precision": precision_f(torch.argmax(pred.squeeze(1), dim=1) == 1, labels.squeeze(1).argmax(dim=1) != 0)})
            wandb.log({"train_recall": recall_f(pred >= 0, labels != 0)}) if not args.pred_mem else wandb.log({"train_recall": recall_f(torch.argmax(pred.squeeze(1), dim=1) == 1, labels.squeeze(1).argmax(dim=1) != 0)})

            if args.mask_neurons and args.gendice: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() # update model weights (values for kernels)
            pbar.set_postfix(step = f"Step: {i}", loss = f"Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
            pbar.update(1)
            pbar.refresh()
            # break

        val_loss_avg, val_count = [], []
        val_prec_avg, val_recall_avg = [], []
        with torch.no_grad():
                    
            epoch_non_empty = False

            for i, data in enumerate(valid_loader):            
                if gen_gj_entities:
                    valid_inputs, valid_labels, valid_label_centers, valid_label_contours, valid_pad_mask, valid_neuron_mask, valid_mito_mask = data
                    valid_label_centers, valid_label_contours, valid_pad_mask = valid_label_centers.to(DEVICE), valid_label_contours.to(DEVICE), valid_pad_mask.to(DEVICE)
                else:
                     valid_inputs, valid_labels, valid_neuron_mask, valid_mito_mask = data
                valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)

                if valid_neuron_mask != []: valid_neuron_mask = valid_neuron_mask.to(DEVICE)
                if valid_mito_mask != []: valid_mito_mask = valid_mito_mask.to(DEVICE)
                valid_pred = model(valid_inputs) if not mem_feat else model(valid_inputs, valid_neuron_mask.to(torch.float32))

                if args.pred_mem:
                    assert valid_neuron_mask != []
                    valid_neuron_mask = valid_neuron_mask.long()
                    valid_neuron_mask[valid_neuron_mask == 0] = 2
                    valid_neuron_mask[valid_neuron_mask == 1] = 0
                    valid_labels = valid_labels + valid_neuron_mask
                    valid_labels[valid_labels == 3] = 1 # make it a gapjnc
                    valid_labels = F.one_hot(valid_labels.long()).to(torch.float32).permute((0, 1, 4, 2, 3))
                    loss_f_ = F.cross_entropy
                else: loss_f_ = F.binary_cross_entropy_with_logits

                if args.focalweight == 0:
                    valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1), valid_neuron_mask.squeeze(1) if valid_neuron_mask != [] and not mem_feat else [], valid_mito_mask if valid_mito_mask !=[] else [], model.s1, model.s2, loss_fn=loss_f_)
                else:   
                    valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1) if not gen_gj_entities else (valid_label_centers, valid_label_contours, valid_pad_mask), valid_neuron_mask.squeeze(1) if valid_neuron_mask != [] and not mem_feat else [], valid_mito_mask if valid_mito_mask !=[] else [], loss_fn=loss_f_,fn_reweight=fn_rwt )
                
                #--ADD FOR CEDAR IF WANT --
                # mask_img = wandb.Image(valid_inputs[0].squeeze(0).cpu().numpy()[0] if args.td else valid_inputs[0].squeeze(0).cpu().numpy(), 
                #                         masks = {
                #                             "predictions" : {
                #                 "mask_data" : (torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))) * 255).cpu().detach().numpy(),
                #                 "class_labels" : class_labels
                #             },
                #             "ground_truth" : {
                #                 "mask_data" : (valid_labels[0].squeeze(0) * 255).cpu().numpy(),
                #                 "class_labels" : class_labels
                #             }}
                # )
                # table.add_data(f"Epoch {epoch} Step {i}", mask_img)
                # wandb.log({"valid_loss": valid_loss})
                val_loss_avg.append(valid_loss.detach().item())
                val_count.append(valid_inputs.shape[0])
                uniques = np.unique(torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))).detach().cpu().numpy())
                if len(uniques) == 2:
                    if not epoch_non_empty:
                        epoch_non_empty = True
                        print("UNIQUE OUTPUTS!")
                else:
                    epoch_non_empty = False
                if not args.pred_mem:
                    val_prec_avg.append(precision_f(valid_pred >= 0, valid_labels).detach().item())
                    val_recall_avg.append(recall_f(valid_pred >= 0, valid_labels).detach().item())
                else:
                    val_prec_avg.append(precision_f(torch.argmax(valid_pred.squeeze(1), dim=1) == 1, valid_labels.squeeze(1).argmax(dim=1) == 1).detach().item())
                    val_recall_avg.append(recall_f(torch.argmax(valid_pred.squeeze(1), dim=1) == 1, valid_labels.squeeze(1).argmax(dim=1) == 1).detach().item())

                # #save it locally:
                # if not args.pred_mem:
                #     write_valid_imgs(valid_pred, valid_labels, i, epoch)
                # else:
                #     # print(torch.unique(torch.argmax(valid_pred.squeeze(1), dim=1), return_counts=True))
                #     write_valid_imgs(valid_pred.squeeze(1).argmax(dim=1) == 1, valid_labels.squeeze(1).argmax(dim=1) * 0.5, i, epoch)
            # log some metrics
            valid_loss = np.sum(np.array(val_loss_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_loss": val_loss_avg})
            val_prec_avg = np.sum(np.array(val_prec_avg) * np.array(val_count))/sum(val_count)
            val_recall_avg = np.sum(np.array(val_recall_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_precision": val_prec_avg})
            wandb.log({"valid_recall": val_recall_avg})
        if decay is not None: decay.step(valid_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss} | Valid Prec: {val_prec_avg} | Valid Recall: {val_recall_avg}")
        print(f"Time elapsed: {time.time() - start} seconds")
        temp_name = model_name+"_epoch"+str(epoch)
        joblib.dump(model, os.path.join(model_folder, f"{temp_name}.pk1"))
    print(f"Total time: {time.time() - start} seconds")
    wandb.log({"Table" : table})
    joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
    wandb.finish()
    try:
        joblib.dump(loss_list, os.path.join(model_folder, "loss_list_1.pkl"))
    except:
        print("Failed to save loss list")

def extend_train_loop(model, train_loader, criterion, optimizer, valid_loader=None, mem_feat=False, epochs=30, decay=None, gen_gj_entities=True, fn_rwt=False):
    global table, class_labels, model_folder, DEVICE, args
    
    print(f"Using device: {DEVICE}")
    model_name = "model5"
    def sigint_handler(sig, frame):
        if table is not None:
            print("logging to WANDB")
            wandb.log({"Table" : table})
            joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGUSR1, sigint_handler)

    recall_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(gt[gt == 1])
    precision_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(pred[pred == 1])

    for epoch in range(epochs):
        pbar = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            pbar.set_description("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, pred_image, pred_mask, labels = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            pred_image, pred_mask = pred_mask.to(DEVICE), pred_mask.to(DEVICE)

            pred = model(inputs, pred_image, pred_mask)
            
            loss = criterion(pred.squeeze(1), labels.squeeze(1))            
            loss.backward() 
            
            # log some metrics
            wandb.log({"train_precision": precision_f(pred >= 0.5, labels)})
            wandb.log({"train_recall": recall_f(pred >= 0.5, labels)})

            if args.mask_neurons and args.gendice: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() 
            pbar.set_postfix(step = f"Step: {i}", loss = f"Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
            pbar.update(1)
            pbar.refresh()
            # break

        val_loss_avg, val_count = [], []
        val_prec_avg, val_recall_avg = [], []
        with torch.no_grad():
                    
            epoch_non_empty = False

            for i, data in enumerate(valid_loader):            
                valid_inputs, valid_pred_image, valid_pred_mask, valid_labels = data
                valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
                valid_pred_mask, valid_pred_image = valid_pred_mask.to(DEVICE), valid_pred_image.to(DEVICE)

                valid_pred = model(valid_inputs, valid_pred_image, valid_pred_mask)
                valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1))
                mask_img = wandb.Image(valid_inputs[0].squeeze(0).cpu().numpy()[0] if args.td else valid_inputs[0].squeeze(0).cpu().numpy(), 
                                        masks = {
                                            "predictions" : {
                                "mask_data" : (torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))) * 255).cpu().detach().numpy(),
                                "class_labels" : class_labels
                            },
                            "ground_truth" : {
                                "mask_data" : (valid_labels[0].squeeze(0) * 255).cpu().numpy(),
                                "class_labels" : class_labels
                            }}
                )
                table.add_data(f"Epoch {epoch} Step {i}", mask_img)
                # wandb.log({"valid_loss": valid_loss})
                val_loss_avg.append(valid_loss.detach().item())
                val_count.append(valid_inputs.shape[0])
                uniques = np.unique(torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))).detach().cpu().numpy())
                if len(uniques) == 2:
                    if not epoch_non_empty:
                        epoch_non_empty = True
                        print("UNIQUE OUTPUTS!")
                else:
                    epoch_non_empty = False
                val_prec_avg.append(precision_f(valid_pred >= 0.5, valid_labels).detach().item())
                val_recall_avg.append(recall_f(valid_pred >= 0.5, valid_labels).detach().item())

                #save it locally:
                # write_valid_imgs(valid_pred, valid_labels, i, epoch)
            # log some metrics
            valid_loss = np.sum(np.array(val_loss_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_loss": val_loss_avg})
            val_prec_avg = np.sum(np.array(val_prec_avg) * np.array(val_count))/sum(val_count)
            val_recall_avg = np.sum(np.array(val_recall_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_precision": val_prec_avg})
            wandb.log({"valid_recall": val_recall_avg})
        if decay is not None: decay.step(valid_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss} | Valid Prec: {val_prec_avg} | Valid Recall: {val_recall_avg}")
        print(f"Time elapsed: {time.time() - start} seconds")
        temp_name = model_name+"_epoch"+str(epoch)
        joblib.dump(model, os.path.join(model_folder, f"{temp_name}.pk1"))
    print(f"Total time: {time.time() - start} seconds")
    wandb.log({"Table" : table})
    joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
    wandb.finish()
    try:
        joblib.dump(loss_list, os.path.join(model_folder, "loss_list_1.pkl"))
    except:
        print("Failed to save loss list")

def inference_save(model, train_dataset, valid_dataset):
    """
    This function is deprecated, needs to be rewritten
    """
    global DEVICE, model_folder, sample_preds_folder

    sample_train_folder = sample_preds_folder+"/train_res"
    os.makedirs(sample_train_folder, exist_ok=True)
    model_file = args.model_name if args.model_name is not None else "model5_epoch89.pk1"
    # Load model
    model = joblib.load(os.path.join(model_folder, model_file))
    #Run model on gpu
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    #Set model to evaluation mode
    model.eval()
    for i in tqdm(range(len(train_dataset))):
        data = train_dataset[i] # image and ground truth from test dataset
        image, gt_mask = data[0], data[1]
        # print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        # print(image)
        suffix = "_1_{}".format(i)
        #plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        #plt.savefig(os.path.join(sample_train_folder, f"sample_pred_{suffix}.png"))
        cv2.imwrite(os.path.join(sample_train_folder, f"sample_pred_{suffix}.png"))
        # plt.show()
        # plt.imshow(gt_mask.squeeze(0).detach().numpy(), cmap="gray")
        #plt.savefig(os.path.join(sample_train_folder, f"sample_gt_{suffix}.png"))
        cv2.imwrite(os.path.join(sample_train_folder, f"sample_gt_{suffix}.png"))
        # plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        # print(pred_mask.shape)
        # pred_mask_binary = pred_mask.squeeze(0).detach()
        pred_mask_binary = torch.round(nn.Sigmoid()(pred_mask)) * 255
        #plt.imshow(pred_mask_binary.cpu().detach().squeeze(0).squeeze(0).numpy(), cmap="gray")
        cv2.imwrite(os.path.join(sample_train_folder, f"sample_pred_binary_{suffix}.png"))
        #plt.savefig(os.path.join(sample_train_folder, f"sample_pred_binary_{suffix}.png"))
        # plt.show()

    sample_val_folder = sample_preds_folder+"/valid_res"
    os.makedirs(sample_val_folder, exist_ok=True)
    for i in tqdm(range(len(valid_dataset))):
        image, gt_mask = valid_dataset[i] # image and ground truth from test dataset
        # print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        # print(image)
        suffix = "_1_{}".format(i)
        plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        plt.savefig(os.path.join(sample_val_folder, f"sample_pred_{suffix}.png"))
        # plt.show()
        plt.imshow(gt_mask.squeeze(0).detach().numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_val_folder, f"sample_gt_{suffix}.png"))
        # plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        
        # pred_mask_binary = torch.argmax(pred_mask.squeeze(0).detach(), 0)
        pred_mask_binary = torch.round(nn.Sigmoid()(pred_mask)) * 255
        plt.imshow(pred_mask_binary.cpu().detach().squeeze(0).squeeze(0).numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_val_folder, f"sample_pred_binary_{suffix}.png"))

"""
def inference_save(model, train_dataset, valid_dataset):
    global DEVICE, model_folder, sample_preds_folder
    print(f"Using device: {DEVICE}")
    
    sample_train_folder = sample_preds_folder + "/train_results"
    sample_val_folder = sample_preds_folder+"/valid_results"
    #Make directories if not already existing
    os.makedirs(sample_train_folder, exist_ok=True)
    os.makedirs(sample_val_folder, exist_ok=True)
    #Load model
    model_file = args.model_name if args.model_name is not None else "model5_epoch89.pk1"
    model = joblib.load(os.path.join(model_folder, model_file))
    #Send model to gpu and run in evaluation mode
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    model.eval()

    # Use DataLoader for batching
    batch_size = args.batch_size if hasattr(args, "batch_size") else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    with torch.no_grad():
        #Predict on training data
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Unpack batch (adjust if your dataset returns more items)
            images, gt_masks = batch[0], batch[1]
            images = images.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)

            preds = model(images)
            pred_masks_binary = torch.round(nn.Sigmoid()(preds)) * 255

            for i in range(images.size(0)):
                suffix = f"_1_{batch_idx * batch_size + i}"
                # Save prediction
                cv2.imwrite(os.path.join(sample_train_folder, f"pred_binary_{suffix}.png"), pred_masks_binary)
"""
def extend_inference_sequence(model, img_dir, preds_dir, new_preds_dir, y_range, x_range, s_min, s_max, img_template, preds_template, off=False):

    os.makedirs(new_preds_dir, exist_ok=True)
    model = model.to("cuda")
    model.eval()

    imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if "DS" not in img]
    imgs = sorted(imgs, key=lambda x: (int(re.findall("Y\d+", x)[0][1:]), int(re.findall("X\d+", x)[0][1:]), int(re.findall("s\d\d\d", x)[0][1:])))

    masks = [os.path.join(preds_dir, img) for img in os.listdir(img_dir) if "DS" not in img]

    wrote_files = 0

    for y in tqdm(y_range):
        for x in x_range:
            cur_s = s_min
            
            prev_pred, prev_img = None, None
            while cur_s <= s_max:
                img_name = img_template + f"s{cur_s}_Y{y}_X{x}{'' if not off else 'off'}.png"
                if not os.path.isfile(os.path.join(img_dir, img_name)):
                    while not (os.path.isfile(os.path.join(img_dir, img_name))) and cur_s != s_max:
                        cur_s = cur_s + 1
                        img_name = img_template + f"s{cur_s}_Y{y}_X{x}{'' if not off else 'off'}.png"
                    if cur_s == s_max: break
                    if cur_s < s_max: cur_s +=1
                    prev_pred, prev_img = None, None

                img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2GRAY)
                if img.shape[0] != 512 or img.shape[1] != 512: 
                    break
                if prev_pred is None:
                    prev_img = img
                    prev_pred = torch.nn.Sigmoid()(torch.from_numpy(np.load(os.path.join(preds_dir, img_name.replace(img_template, preds_template) + ".npy")))).unsqueeze(0).unsqueeze(0) >= 0.5
                else:
                    # predict this based on the previous pred
                    _transform = []
                    _transform.append(transforms.ToTensor())
                    new_pred = torch.nn.Sigmoid()(model(transforms.Compose(_transform)(img).to("cuda").unsqueeze(0), transforms.Compose(_transform)(prev_img).to("cuda").unsqueeze(0), prev_pred.to(dtype=torch.float32).to("cuda"))) >= 0.5

                    #write it in 
                    assert cv2.imwrite(os.path.join(new_preds_dir, img_name), new_pred.detach().cpu().numpy().squeeze(0).squeeze(0) * 255)
                    prev_pred = new_pred
                    prev_img = img
                    wrote_files += 1
                cur_s += 1
    # move it backwards:
    for y in tqdm(y_range):
        for x in x_range:
            cur_s = s_max

            after_img, after_pred = None, None
            while cur_s >= s_min:
                img_name = img_template + f"s{cur_s}_Y{y}_X{x}{'' if not off else 'off'}.png"
                if not os.path.isfile(os.path.join(img_dir, img_name)):
                    while not (os.path.isfile(os.path.join(img_dir, img_name))) and cur_s !=s_min:
                        cur_s = cur_s - 1
                        img_name = img_template + f"s{cur_s}_Y{y}_X{x}{'' if not off else 'off'}.png"

                    if cur_s == s_min: break
                    if cur_s > s_min: cur_s -=1
                    prev_pred, prev_img = None, None
                img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2GRAY)
                if img.shape[0] != 512 or img.shape[1] != 512: 
                    break
                if prev_pred is None:
                    prev_img = img
                    if not os.path.isfile(os.path.join(new_preds_dir, img_name.replace(img_template, preds_template))):
                        prev_pred = torch.nn.Sigmoid()(torch.from_numpy(np.load(os.path.join(preds_dir, img_name.replace(img_template, preds_template))+".npy"))).unsqueeze(0).unsqueeze(0) >= 0.5
                    else:
                        prev_pred = cv2.cvtColor(cv2.imread(os.path.join(new_preds_dir, img_name.replace(img_template, preds_template)))).unsqueeze(0) == 255
                else:
                    # predict this based on the previous pred
                    _transform = []
                    _transform.append(transforms.ToTensor())

                    new_pred = torch.nn.Sigmoid()(model(transforms.Compose(_transform)(img).to("cuda").unsqueeze(0), transforms.Compose(_transform)(prev_img).to("cuda").unsqueeze(0), prev_pred.to(dtype=torch.float32).to("cuda"))) >= 0.5

                    #write it in 
                    assert cv2.imwrite(os.path.join(new_preds_dir, img_name[:-4]+ "after.png"), new_pred.detach().cpu().numpy().squeeze(0).squeeze(0) * 255)
                    prev_pred = new_pred
                    prev_img = img
                cur_s -= 1   

def setup_wandb(epochs, lr):
    global table, class_labels
    WANDB_API_KEY = "42a2147c44b602654473783bde1ecd15579cc313"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="celegans",
        entity="mishaalkandapath",
        dir="/home/tommy/scratch/gapjunc",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        }
    )
    class_labels = {
        0: "background",
        1: "gapjunction",
    }

    table = wandb.Table(columns=['ID', 'Image'])

def new_collate(batch):
    inputs, labels, label_centers, label_contours, neuron_mask, mito_mask = zip(*batch) # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])label_centers, label_contours

    label_centers_batched = torch.nn.utils.rnn.pad_sequence(label_centers, batch_first=True, padding_value = 15)
    label_contours_batched = torch.nn.utils.rnn.pad_sequence(label_contours, batch_first=True, padding_value = 15)

    labels = torch.stack(labels, axis=0)

    pad_mask = label_centers_batched == 15

    label_centers_batched[pad_mask] = 0
    label_contours_batched[pad_mask] = 0


    pad_mask = pad_mask.sum(dim=-1).sum(dim=-1) >= 1

    if neuron_mask[0] != []:
        neuron_mask = torch.stack(neuron_mask, axis=0)
    else:
        neuron_mask = []
    if mito_mask[0] != []:
        mito_mask = torch.stack(mito_mask, axis=0)
    else: mito_mask = []

    assert len(torch.unique(label_centers_batched.sum(dim=1))) <= 2, f"{torch.unique(label_centers_batched.sum(dim=1))} {(torch.unique(label_contours_batched))}"

    return torch.stack(inputs, axis=0), labels, label_centers_batched.to(dtype=torch.bool), label_contours_batched.to(dtype=torch.bool), pad_mask, neuron_mask, mito_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mask_neurons", action="store_true")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--mask_mito", action="store_true")
    parser.add_argument("--special", action="store_true")
    parser.add_argument("--td", action="store_true")
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--gendice", action="store_true")
    parser.add_argument("--ss", action="store_true")
    parser.add_argument("--dicefocal", action="store_true")
    parser.add_argument("--tversky", action="store_true")
    parser.add_argument("--customloss", action="store_true")
    parser.add_argument("--focalweight", default=0.5, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--mem_feat", action="store_true")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout for neural network training")
    parser.add_argument("--spatial", action="store_true", help="Spatial Pyramid")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--pred_mem", action="store_true")
    parser.add_argument("--lr", default=0, type=float)
    parser.add_argument("--extendinfer", action="store_true")
    parser.add_argument("--finetune_dirs", action="append", type=str)
    parser.add_argument("--fn_rwt", action="store_true")

    args = parser.parse_args()

    if not args.special and not args.extendinfer:

        if args.seed:
            SEED = 12
            np.random.seed(SEED)
            random.seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.manual_seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)

        batch_size = args.batch_size

        
        if args.dataset is not None:
            if args.fn_rwt: dset = args.dataset + "_rwt"
            else: dset = args.dataset
            train_dir = (DATASETS[dset if not args.extend else "extend"])
            train_dataset, valid_dataset =  make_dataset_new(train_dir, aug=args.aug,neuron_mask=args.mask_neurons or args.mem_feat or args.pred_mem, mito_mask=args.mask_mito, chain_length=args.td, gen_gj_entities=args.customloss, finetune_dirs=args.finetune_dirs)
        # else: make_dataset_old(args.aug)

        if args.dataset is None: print("----WARNING: RUNNING OLD DATASET----")

        print("ARGS: ", sys.argv)

        #set the model and results path
        model_folder += args.dataset+f"{'_residual' if args.residual else ''}"+"/" if not args.spatial else args.dataset+f"spatial{'_residual' if args.residual else ''}"+"/"
        sample_preds_folder += args.dataset + "/" if not args.spatial else args.dataset+f"spatial{'_residual' if args.residual else ''}"+"/"

        #extend?
        model_folder = model_folder[:-1]+("extend/" if args.extend else "/")
        sample_preds_folder = sample_preds_folder[:-1]+("extend/" if args.extend else "/")

        if "test" in args.dataset:
            print(len(train_dataset))
            train_old, _ = make_dataset_new(DATASETS["new"], aug=args.aug,neuron_mask=args.mask_neurons or args.mem_feat, mito_mask=args.mask_mito, chain_length=args.td)
            print(len(train_old))
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [math.ceil(0.08*len(train_dataset)), math.ceil(0.12*len(train_dataset)), len(train_dataset) - math.ceil(0.08*len(train_dataset)) - math.ceil(0.12*len(train_dataset))])
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_old]) 
            print(len(train_dataset))  

        if not args.customloss:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=12)      
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=new_collate)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=new_collate)        
        
        print("Data loaders created.")

        DEVICE = torch.device("cuda") if torch.cuda.is_available() or not args.cpu else torch.device("cpu")

        if args.extend:
            model = ExtendUNet().to(DEVICE)        
        elif args.model_name is None and not args.resnet:
            model = UNet(three=args.td, scale=args.focalweight == 0, spatial=args.spatial, dropout=args.dropout, residual=args.residual, classes=2 if not args.pred_mem else 3).to(DEVICE) if not args.mem_feat else MemUNet().to(DEVICE)
        elif args.model_name is None and args.resnet:
            model = ResUNet(three=args.td).to(DEVICE) if not args.mem_feat else MemResUNet(three=args.td).to(DEVICE)
        else: model = joblib.load(os.path.join("/mnt/d/mishaal_to_move/2d_gd_mem_run1/", args.model_name)) # load model

        if not args.infer:# and not args.new_preds_dir:

            print("Current dataset {}".format(args.dataset))

            if args.dataset != "tiny" and args.dataset != "erased" and args.dataset != "new" and args.dataset != "new3d" and args.dataset != "test" and not args.extend:
                print("MASKING NEURONS IS SET TO {}".format(args.mask_neurons))

                #calc focal weighting:
                smushed_labels = None
                if args.mask_neurons: nm = None
                for i in tqdm(range(len(train_dataset))):
                    if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
                    else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
                    if args.mask_neurons and nm is None: nm = torch.tensor(train_dataset[i][2], dtype=torch.bool)
                    elif args.mask_neurons: nm = torch.concat([nm,  torch.tensor(train_dataset[i][2])], dtype=torch.bool)

                smushed_labels = smushed_labels.flatten() if not args.mask_neurons else smushed_labels.flatten()[nm.flatten()]
                class_counts = torch.bincount(smushed_labels)
                total_samples = len(train_dataset) * 512 * 512
                w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
                cls_weights = torch.Tensor([w1, w2/2]) #soften it a bit
                print(cls_weights)
            elif args.dataset == "tiny":
                cls_weights = torch.Tensor([1, 90])
                print(cls_weights)
            elif args.dataset == "erased":
                cls_weights = torch.Tensor([1, 15])
            else:
                cls_weights = torch.Tensor([0.08, 0.92]) if not args.pred_mem else torch.Tensor([0.4, 0.85, 0.12])

            #init optimizers
            if args.customloss:
                criterion = SpecialLoss()
                model_folder = model_folder[:-1]+"custom"
            elif args.gendice:
                criterion = GenDLoss() if not args.pred_mem else MultiGenDLoss()
                model_folder= model_folder[:-1] + "gendice"
            elif args.dice:
                criterion = DiceLoss()
                model_folder = model_folder[:-1] + "dice"
            elif args.ss:
                criterion = SSLoss()
                model_folder =model_folder[:-1] +"ssloss"
            elif args.dicefocal:
                floss, gloss =  FocalLoss(alpha=cls_weights, device=DEVICE), GenDLoss()
                if args.focalweight == 0:
                    criterion = lambda a,b,c,d,s1, s2: floss(s1*a, b, c, d) + gloss(s2*a, b, c, d)
                else: criterion = lambda a,b,c,d: args.focalweight * floss(a, b, c, d) + (1 - args.focalweight) *gloss(a, b, c, d)
                model_folder =model_folder[:-1] +"dicefocalloss"
            elif args.tversky:
                criterion = TverskyFocal()
                model_folder =model_folder[:-1] +"tvrsky"
            else:
                criterion = FocalLoss(alpha=cls_weights, device=DEVICE)#torch.nn.BCEWithLogitsLoss()
                model_folder= model_folder[:-1] +"focal"

            optimizer = torch.optim.Adam(model.parameters(),lr=1e-6 if not args.lr else args.lr)
            decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            loss_list = [] 

            if args.split:
                classifier_criterion = torch.nn.BCEWithLogitsLoss()
            

            setup_wandb(30, 0.0001)

            print("Starting training...")
            start = time.time()

            #determine save folder
            counter = 0
            model_folder+="mito" if args.mask_mito else ""
            model_folder += "neuro" if args.mask_neurons else ""
            while (os.path.isdir(model_folder)):
                model_folder += str(counter)
                counter+=1
                continue
            os.mkdir(model_folder)

            print("SAVING MODELS TO {}".format(model_folder))
    
            print(f"running for {300 if not args.epochs else args.epochs} epochs")
            
            train_wrapper = train_loop if not args.extend else extend_train_loop
            train_wrapper(model, train_loader, criterion, optimizer, valid_loader, epochs=300 if not args.epochs else args.epochs, mem_feat=args.mem_feat, gen_gj_entities=args.customloss, fn_rwt=args.fn_rwt)

        else:
            inference_save(model, train_dataset, valid_dataset)
    elif args.extendinfer:
        model_folder += "newextendgendice01" # and 0 are the best ones. 01 is some tweak on the learning rate i believe
        model = joblib.load(os.path.join(model_folder, "model5_epoch56.pk1"))

        model = model.to("cuda")
        model.eval()

        extend_inference_sequence(model, "/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split/imgs", sample_preds_folder+"2d_gd_mem_run1_full_front/", sample_preds_folder+"extend_preds_off/", range(0, 16), range(0, 18), 100, 120, "sem_dauer_2_em_",  "SEM_dauer_2_export_", off=True)

    else:
        #--- Personal debug area ---
        #Use only argument --special to access 
        #MODEL
        model_folder = "/mnt/d/mishaal_to_move/2d_gd_mem_run1/" #"2d_gd_mem_run1" IS THE BEST HERE OKAY??
        #OUTPUT
        sample_preds_folder = "/home/tommytang111/results/"
        #DATA
        dataset_dir="/mnt/e/Mishaal/sem_dauer_2/jnc_only_dataset_test/imgs"
        #LOAD MODEL
        model = joblib.load(os.path.join(model_folder, "model5_epoch149.pk1")) #3d gendice epoch 95, focal epoch 108, 2d mask 115, df_0.2 R1 73, dyna R1 82
        # for flat dyna we have 125, 2wt we have 142, 2d_gd_mem_run2 best was 36, 2d_membrane_aug_low is 83, 2d_membrane_noaug is 140
        #resnet w memebrane 67 or 84, 
        #resnet 3d 131 or 130
        #resnet focal 126
        model = model.to("cuda")
        model.eval()

        # dataset_dir = "/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split/valid_imgs"
        # dataset_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/3d_test_new/imgs/"
        # dataset_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/full_front_split/imgs/"
        # mem_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split/neurons/imgs"
        # dataset_dir="/home/mishaalk/scratch/gapjunc/test_datasets/dauer1_test/imgs"

        # dataset = CaImagesDataset(dataset_dir, preprocessing=None, augmentation=None, image_dim=(512, 512), split=1)
        dataset = TestDataset(dataset_dir, td=False, membrane=False)#, membrane="/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split/neurons/")
        imgs_files, gt_files = [i for i in sorted(os.listdir(dataset_dir)) if "DS" not in i], [i for i in sorted(os.listdir(dataset_dir)) if "DS" not in i]
        def collate_fn(batch):
            return (
                torch.stack([x[0] for x in batch]),
                torch.stack([x[1] for x in batch]),
                torch.stack([x[2] for x in batch]),
                torch.stack([x[3] for x in batch]),
            )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

        if not os.path.isdir(sample_preds_folder):
            os.mkdir(sample_preds_folder)

        # assert len(imgs_files) == 2695
        recall_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(gt[gt == 1])
        precision_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(pred[pred == 1])

        i = 0
        with torch.no_grad():
            class_list = []
            rec_acc, prec_acc = [], []
            for data in tqdm(dataloader):
                image, gt, membrane, _ = data

                #infer:
                x_tensor = image.to("cuda")
                memb = membrane.to("cuda")
                pred_mask = model(x_tensor) # [1, 2, 512, 512]
                # print(pred_mask.shape)
                # pred_mask_binary = pred_mask.squeeze(0).detach()
                pred_mask_binary = pred_mask
                # classified = torch.round(nn.Sigmoid()(classified))
                pred_mask_binary = pred_mask_binary.to("cpu")
                for j in range(pred_mask_binary.shape[0]):
                    file_name = imgs_files[i]
                    # prfed_og = cv2.cvtColor(cv2.imread(sample_preds_folder+file_name), cv2.COLOR_BGR2GRAY)
                    # print(np.count_nonzero(prfed_og == pred_mask_binary[j].squeeze(0).numpy()))
                    # raise Exception
                    # if os.path.isfile(sample_preds_folder+file_name): continue
                    # cv2.imwrite(sample_preds_folder+file_name+".png", pred_mask_binary[j].squeeze(0).numpy())
                    #save it as a numpy file
                    f = open(sample_preds_folder+file_name+".npy", "wb")
                    np.save(f, pred_mask_binary[j].squeeze(0).numpy())
                    f.close()
                    i+=1
                    # class_list.append([file_name, classified[j].item()])
                    # pred_mask_binary1 = nn.Sigmoid()(pred_mask_binary) >= 0.5
                    # gt[gt == 255] = 1
        #             rec_acc.append(recall_f(pred_mask_binary1, gt).detach().item())
        #             prec_acc.append(precision_f(pred_mask_binary1, gt).detach().item())
        # print(np.nanmean(prec_acc))
        # print(np.nanmean(rec_acc))



        print(f"I wrote {i} files")
        
        # with open(sample_preds_folder+"classes.csv", "w") as f:
        #     obj = csv.writer(f)
        #     obj.writerows(class_list)
            

