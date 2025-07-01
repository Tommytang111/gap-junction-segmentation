import cv2
import os
import numpy as np
from tqdm import tqdm 
import re
import csv
from PIL import Image
import matplotlib.pyplot as plt
import threading, random
import glob, traceback

from copy import deepcopy
BASE = "/Volumes/Normal/gapjnc/"
def only_junc_images_datast():
    #compile a list of images and their corresponding segments that have a gap junction annotation:
    jnc_files = os.listdir(BASE+"seg_export0507_2\\")

    new_img_files, new_seg_files = [], []
    for file in tqdm(jnc_files):
        seg = cv2.cvtColor(cv2.imread(BASE+"seg_export0507_2\\"+file), cv2.COLOR_BGR2GRAY)
        if len(np.unique(seg)) >= 2:
            new_img_files.append(file.replace("sem2dauer_gj_2d_training.vsseg_export", "SEM_dauer_2_image_export"))
            new_seg_files.append(file)

    new_img_files = [file + "\n" for file in new_img_files]
    new_seg_files = [file + "\n" for file in new_seg_files]
    with open(BASE + "jnc_only_images.txt", "w") as f:
        f.writelines(new_img_files)
    with open(BASE + "jnc_only_seg.txt", "w") as f:
        f.writelines(new_seg_files)

def neuron_present_images_dataset():
    jnc_files = os.listdir(BASE+"sem_dauer_2/seg_export/")

    new_img_files = []
    for file in tqdm(jnc_files):
        new_img_files.append(file.replace("sem_dauer_2_gj_segmentation_", "SEM_dauer_2_image_export_"))
    
    new_img_files = [file + "\n" for file in new_img_files]
    with open(BASE + "sem_dauer_2/seg_export_images.txt", "w") as f:
        f.writelines(new_img_files)

def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return A


def erase_neurons():
    img_files = os.listdir(BASE+"jnc_only_images\\")
    seg_files = os.listdir(BASE+"jnc_only_seg\\")
    neuron_files = os.listdir("E:\Mishaal\GapJunction\seg_export\\")
    os.mkdir(BASE+"erased_neurons\\")
    os.mkdir(BASE+"erased_neurons\\gts\\")
    os.mkdir(BASE+"erased_neurons\\imgs\\")

    for i in tqdm(range(len(neuron_files))):
        img_file = neuron_files[i].replace("20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_", "SEM_dauer_2_image_export_")
        seg_file = neuron_files[i].replace("20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_", "sem2dauer_gj_2d_training.vsseg_export_")
        if not os.path.isfile(BASE+"jnc_only_images\\" + img_file) or not os.path.isfile(BASE+"jnc_only_seg\\" + seg_file): continue
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_file), cv2.COLOR_BGR2GRAY)
        neuron_mask = cv2.cvtColor(cv2.imread("E:\Mishaal\GapJunction\seg_export\\"+neuron_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_file), cv2.COLOR_BGR2GRAY)
        seg[seg != 1] = 0
        seg[seg == 1] = 255
        
        img[neuron_mask != 0] = 255

        #write the new image to directory along with a seg directory 
        cv2.imwrite(BASE+"erased_neurons\\imgs\\"+img_file, img)
        cv2.imwrite(BASE+"erased_neurons\\gts\\"+seg_file, seg)

def shorten_pics():
    counter = 0
    img_files = sorted(os.listdir(BASE+"jnc_only_images\\"))
    seg_files = sorted(os.listdir(BASE+"jnc_only_seg\\"))
    # os.mkdir(BASE+"tiny_jnc_only\\")
    os.mkdir(BASE+"tiny_jnc_only\\gts\\")
    os.mkdir(BASE+"tiny_jnc_only\\imgs\\")
    os.mkdir(BASE+"tiny_jnc_only\\masks\\")
    os.mkdir(BASE+"tiny_jnc_only\\mito_masks\\")

    for i in tqdm(range(len(img_files))):
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(cv2.imread(("E:\\Mishaal\\GapJunction\\seg_export\\"+seg_files[i]).replace("sem2dauer_gj_2d_training.vsseg_export", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export")), cv2.COLOR_BGR2GRAY)
        mito_mask = np.array(Image.open("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+img_files[i]+".tiff"))
        seg[seg != 1] = 0
        seg[seg == 1] = 255

        #split the image into 32x32 bits
        imgs, segs, masks, mito_masks = ressample(img, 128), ressample(seg, 128), ressample(mask, 128), ressample(mito_mask, 128)
        # save these 
        counter=0
        for j in range(len(segs)):
            cv2.imwrite(BASE+"tiny_jnc_only\\gts\\{}.png".format(seg_files[i][:-4] + "_"+str(counter)), segs[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\imgs\\{}.png".format(img_files[i][:-4] + "_"+str(counter)), imgs[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\masks\\{}.png".format(seg_files[i][:-4].replace("sem2dauer_gj_2d_training.vsseg_export", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export")
                                                                    + "_"+str(counter)), masks[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\mito_masks\\{}.png".format(img_files[i][:-4] + "_"+str(counter)), mito_masks[j])
            counter+=1

def stitch_short(train_dir, preds_dir, s_dir):
    pred_files = os.listdir(preds_dir)
    pred_files.remove("classes.csv")
    img_files = sorted(os.listdir(train_dir+"imgs/"), key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))
    seg_files = sorted(os.listdir(train_dir + "gts/"), key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))
    pred_files = sorted(pred_files, key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))

    # all_saves_list = ThreadSafeList()

    def helper_inside(img_files, seg_files, pred_files, save_dir):
        img_acc, seg_acc, pred_acc = [], [], []
        base_count = None
        for i in tqdm(range(0, len(img_files))):
            if i % 16 == 0 and i != 0:
                #np.array(img_acc).reshape((512, 512))
                img_acc = np.concatenate([np.concatenate(img_acc[:4], axis=1), np.concatenate(img_acc[4:8], axis=1), np.concatenate(img_acc[8:12], axis=1), np.concatenate(img_acc[12:16], axis=1)], axis=0) 
                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img."))
                #all_saves_list.append((img_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img."))))
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img.")), img_acc)
                img_acc = []

                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_sef."))
                seg_acc = np.concatenate([np.concatenate(seg_acc[:4], axis=1), np.concatenate(seg_acc[4:8], axis=1), np.concatenate(seg_acc[8:12], axis=1), np.concatenate(seg_acc[12:16], axis=1)], axis=0) 
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_seg.")), seg_acc)
                #all_saves_list.append((seg_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_seg."))))
                seg_acc = []

                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))
                pred_acc = np.concatenate([np.concatenate(pred_acc[:4], axis=1), np.concatenate(pred_acc[4:8], axis=1), np.concatenate(pred_acc[8:12], axis=1), np.concatenate(pred_acc[12:16], axis=1)], axis=0) 
                #all_saves_list.append((pred_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))))
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred.")), pred_acc)
                pred_acc = []

            img = cv2.cvtColor(cv2.imread(train_dir+"imgs/"+img_files[i]), cv2.COLOR_BGR2GRAY)
            img_acc.append(img)
            base_count = img_files[i]

            seg = cv2.cvtColor(cv2.imread(train_dir+"gts/"+seg_files[i]), cv2.COLOR_BGR2GRAY)
            seg_acc.append(seg)

            pred = cv2.cvtColor(cv2.imread(preds_dir+pred_files[i]), cv2.COLOR_BGR2GRAY)
            pred_acc.append(pred)
        print("I did {} files".format(len(img_files)))

    offsets = len(img_files)//16
    threads = []
    for j in range(11):
        t = threading.Thread(target=helper_inside, args=(img_files[j*245*16 : (j+1)*245*16], seg_files[j*245*16 : (j+1)*245*16], pred_files[j*245*16 : (j+1)*245*16], s_dir))
        threads.append(t)

    [t.start() for t in threads]
    [t.join() for t in threads]
    print(all_saves_list.length())



def mask_acc(gt, pred, mask=None):
    good_mask = np.logical_and(gt != 2, gt != 15).flatten()
    if mask is not None:
        good_mask = np.logical_and(good_mask, mask.flatten())

    gt = gt.flatten()[good_mask]
    pred = pred.flatten()[good_mask]
    gt[gt != 0] = 255  

    # if mask is not None:
    #     print(gt.shape, mask.shape)
    #     gt = gt.flatten()[mask.flatten()]
    #     pred = pred.flatten()[mask.flatten()]

    return np.sum(gt == pred) / (gt.shape[0])
    # return np.sum(gt == pred) / (gt.shape[0] * gt.shape[1])

def mask_precision(gt, pred, mask=None):
    good_mask = np.logical_and(gt != 2, gt != 15).flatten()
    if mask is not None: good_mask = np.logical_and(good_mask, mask.flatten())

    gt = gt.flatten()[good_mask]
    pred = pred.flatten()[good_mask]
    gt[gt != 0] = 255  

    # if mask is not None:
    #     gt = gt.flatten()[mask.flatten()]
    #     pred = pred.flatten()[mask.flatten()]

    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(pred == 255)

def move_generous(img, dims, itr):
    accum = img.copy()
    for i in range(min(itr, 0), max(itr, 0)):
        accum = accum | np.roll(img, i+1, dims)
    return accum

def mask_precision_generous(gt, pred, mask=None):
    new_gt = np.pad(gt, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
    new_gt = move_generous(new_gt, 0, 10) | move_generous(new_gt, 1, 10) | move_generous(new_gt, 1, -10) | move_generous(new_gt, 0, -10)
    new_gt = new_gt[10:-10, 10:-10]

    good_mask = np.logical_and(gt != 2, gt != 15).flatten()
    if mask is not None: good_mask = np.logical_and(good_mask, mask.flatten())
    gt = gt.flatten()[good_mask]
    pred = pred.flatten()[good_mask]
    new_gt = new_gt.flatten()[good_mask]
    new_gt[new_gt != 0] = 255
    gt[gt != 0] = 255     

    num = (np.logical_and(gt == 255, pred == 255) | np.logical_and(new_gt == 255, pred == 255)) >= 1
    denom = np.sum(pred == 255)
    #remove mask items.
    # print(num.flatten().shape, mask.flatten().shape)
    # if mask is not None:
    #     num = num.flatten()[mask.flatten()]
    #     denom = np.count_nonzero(pred.flatten()[mask.flatten()] == 255)
    return np.sum(num)/denom 

def mask_acc_generous(gt, pred, mask=None):
    new_gt = np.pad(gt, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
    new_gt = move_generous(new_gt, 0, 10) | move_generous(new_gt, 1, 10) | move_generous(new_gt, 1, -10) | move_generous(new_gt, 0, -10)
    new_gt = new_gt[10:-10, 10:-10]
    
    good_mask = np.logical_and(gt != 2, gt != 15).flatten()
    if mask is not None: good_mask = np.logical_and(good_mask, mask.flatten())
    gt = gt.flatten()[good_mask]
    pred = pred.flatten()[good_mask]
    new_gt = new_gt.flatten()[good_mask]
    new_gt[new_gt != 0] = 255
    gt[gt != 0] = 255         

    num = ((new_gt == pred) +(gt == pred)) >= 1
    denom = (gt.shape[0])
    # print(num.flatten().shape, mask.flatten().shape)
    #remove mask items.
    # if mask is not None:
    #     num = num.flatten()[mask.flatten()]
    #     denom = np.count_nonzero(mask)
    return np.sum(num)/denom

def iou_accuracy(gt, pred, mask=None):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(np.logical_or(gt == 255, pred == 255))

def mask_recall(gt, pred, mask=None):
    good_mask = np.logical_and(gt != 2, gt != 15).flatten()
    if mask is not None: good_mask = np.logical_and(good_mask, mask.flatten())
    gt = gt.flatten()[good_mask]
    pred = pred.flatten()[good_mask]
    gt[gt != 0] = 255   
    
    # if mask is not None:
    #     gt = gt.flatten()[mask.flatten()]
    #     pred = pred.flatten()[mask.flatten()]

    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(gt == 255)


def mask_acc_split(data_dir, results_dir, classes=False, test=False):
    # mask_dir = "/Volumes/Normal/masks/"
    # segs = sorted(glob.glob(data_dir+"/gts/*seg*"))
    # train_res = sorted(glob.glob(data_dir+"/imgs/*image*"))
    # assert train_res != []
    train_classes, train_preds, train_gt, train_gcls = [], [], [], []
    segs = sorted(glob.glob(data_dir+"*gj*"))
    train_res = sorted(os.listdir(results_dir))

    if classes:
        t_cls_file = open(results_dir + "classes.txt", "r")
        t_cls = csv.reader(t_cls_file)

    t_recall, t_acc = [], []
    t_acc_gen = []
    t_prec, t_prec_gen = [], []
    t_bacc = []

    t_45_recall, t_3_recall, t_2_recall, t_1_recall = [], [], [], []

    weird_shape_count = 0

    for i, file in tqdm(enumerate(train_res), total=len(train_res)): 
        if "classes" in file: continue
        if "off" in file: continue
        if "DS" in file: continue
        try:
            segs[i] = os.path.join(data_dir, file.replace("SEM_dauer_2_export_", "sem_dauer_2_gj_seg_"))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY)
            no = re.findall(r'_\d+', file)[0]
            pred = cv2.cvtColor(cv2.imread(results_dir+os.path.split(file)[-1]), cv2.COLOR_BGR2GRAY)

            #area mask 
            mask = cv2.cvtColor(cv2.imread("/Volumes/Normal/gapjnc/nr_inout1/"+os.path.split(file)[-1].replace("SEM_dauer_2_export_" ,"nr_mask_").replace("png.png", "png").replace("_pred", "")), cv2.COLOR_BGR2GRAY)
            mask = mask == 21
            if test:
                test_gt = gt.copy()

                if len(np.unique(gt)) >= 2:
                    new_gt = np.pad(gt, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
                    new_gt = move_generous(new_gt, 0, 10) | move_generous(new_gt, 1, 10) | move_generous(new_gt, 1, -10) | move_generous(new_gt, 0, -10)
                    new_gt = new_gt[10:-10, 10:-10]
                    # plt.imshow(new_gt, cmap="gray")
                    # plt.show()
                    # plt.imshow(gt, cmap="gray")
                    # plt.show()
                    # plt.imshow(pred, cmap="gray")
                    # plt.show()
                    # print(mask_recall(gt, pred, mask=mask), mask_precision_generous(gt, pred, mask=mask))
                    # print(segs[i])
                # gt = (np.where(test_gt <= 11, 1, 0) | np.where(test_gt >= 3, 1, 0)) * 255
                # gt = np.ma.masked_array(gt, mask=(gt == 2))
                # gt = np.ma.masked_array(gt, mask=(gt == 15))
                # pred = np.ma.masked_array(pred, mask=(gt == 2))
                # pred = np.ma.masked_array(pred, mask=(gt == 15))
            # mask = cv2.cvtColor(cv2.imread(mask_dir + file.replace("_img","").replace("SEM_dauer_2_image_export_" ,"20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_").replace(data_dir, "")), cv2.COLOR_BGR2GRAY)
            # mask = mask == 0
            #     continue
            if classes:
                cls = 1 if "True" in t_cls[no] else 0
                true_cls = np.any(gt == 255).item()

            t_acc.append(mask_acc(gt, pred, mask=mask))
            t_recall.append(mask_recall(gt, pred, mask=mask))
            t_acc_gen.append(mask_acc_generous(gt, pred, mask=mask))
            t_prec_gen.append(mask_precision_generous(gt, pred, mask=mask))
            t_prec.append(mask_precision(gt, pred, mask=mask))

            # if t_recall[-1] < 0.9:
            #     gt[(gt == 2) | (gt == 15)] = 0
            #     gt[gt!=0] = 255
            #     plt.imshow(gt, cmap="gray")
            #     print(segs[i])
            #     print(train_res[i])
            #     plt.show()
            #     plt.imshow(pred, cmap="gray")
            #     plt.show()

            #extra recall:

            t_45_recall.append(mask_recall((np.where(test_gt == 11, 1, 0) | np.where(test_gt == 9, 1, 0)) * 255, pred, mask=mask))
            t_3_recall.append(mask_recall(np.where(test_gt == 7, 255, 0), pred, mask=mask))
            t_2_recall.append(mask_recall(np.where(test_gt == 5, 255, 0), pred, mask=mask))
            t_1_recall.append(mask_recall(np.where(test_gt == 3, 255, 0), pred, mask=mask))
            
            train_preds.append(pred)
            train_gt.append(gt)
            if classes:
                train_classes.append(cls)
                train_gcls.append(true_cls)
        except Exception:
            segs[i] = os.path.join(data_dir, file.replace("SEM_dauer_2_em_", "sem_dauer_2_gj_gt_"))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY) 
            assert not np.count_nonzero(gt[(gt != 2) & (gt != 15)])
            weird_shape_count +=1

            print(traceback.format_exc())
            # print(file, segs[i])
    

    print("Training set classifier accuracy {}".format(np.nanmean((np.array(train_classes) == np.array(train_gcls)))))
    print("Training set mask accuracy {}".format(np.nanmean(t_acc)))
    print("Training set mask recall {}".format(np.nanmean(t_recall)))
    print("Training set mask accuracy generous {}".format(np.nanmean(t_acc_gen)))
    print("Training set mask prec {}".format(np.nanmean(t_prec)))
    print("Training set mask prec generous {}".format(np.nanmean(t_prec_gen)))

    print("Training set mask recall 45 {}".format(np.nanmean(t_45_recall)))
    print("Training set mask recall 3 {}".format(np.nanmean(t_3_recall)))
    print("Training set mask recall 2 {}".format(np.nanmean(t_2_recall)))
    print("Training set mask recall 1 {}".format(np.nanmean(t_1_recall)))


def mask_acc_split_intersect(data_dir, results_dir, classes=False):
    mask_dir = "/Volumes/Normal/masks/"
    segs = sorted(glob.glob(data_dir+"*seg*"))
    train_res = sorted(glob.glob(data_dir+"*img*"))
    train_classes, train_preds, train_gt, train_gcls = [], [], [], []

    if classes:
        t_cls_file = open(results_dir + "classes.txt", "r")
        t_cls = csv.reader(t_cls_file)

    t_recall, t_acc = [], []
    t_acc_gen, t_acc_iou = [], []
    t_prec, t_prec_gen = [], []

    bad_count = 0

    for i, file in tqdm(enumerate(train_res), total=len(train_res)):
        if "classes" in file: continue
        try:
            pred_acc, gt_acc = None, None
            for j in range(3):
                sint = int(re.findall(r's\d+', file)[0][1:])
                strint = "s"+ "0" + ("" if (sint + j) >= 10 else "0")+str(sint+j)
                try:
                    gt = cv2.cvtColor(cv2.imread(segs[i].replace(re.findall(r's\d+', file)[0], strint)), cv2.COLOR_BGR2GRAY) == 255
                    no = re.findall(r'_\d+', file)[0]
                    pred = cv2.cvtColor(cv2.imread(file.replace("img", "pred").replace(re.findall(r's\d+', file)[0], strint)), cv2.COLOR_BGR2GRAY) == 255
                    
                    if gt_acc is None: gt_acc = gt
                    else: gt_acc = gt_acc & gt
                    if pred_acc is None: pred_acc = pred
                    else: pred_acc = pred_acc & pred
                except: 
                    strint = "s"+ "0" + ("" if (sint - j) >= 10 else "0")+str(sint+j)
                    if os.path.isfile(segs[i].replace(re.findall(r's\d+', file)[0], strint)):
                        gt = cv2.cvtColor(cv2.imread(segs[i].replace(re.findall(r's\d+', file)[0], strint)), cv2.COLOR_BGR2GRAY) == 255
                        no = re.findall(r'_\d+', file)[0]
                        pred = cv2.cvtColor(cv2.imread(file.replace("img", "pred").replace(re.findall(r's\d+', file)[0], strint)), cv2.COLOR_BGR2GRAY) == 255
                        
                        if gt_acc is None: gt_acc = gt
                        else: gt_acc = gt_acc & gt
                        if pred_acc is None: pred_acc = pred
                        else: pred_acc = pred_acc & pred
                    else:
                        bad_count += 1

                    continue
            # if "SEM_dauer_2_image_export_s041_Y6_X12_" in file:
            #     print(np.count_nonzero(gt_acc), np.count_nonzero(pred_acc))
            
            pred = pred_acc * 255
            gt = gt_acc * 255

            mask = cv2.cvtColor(cv2.imread(mask_dir + file.replace("_img","").replace("SEM_dauer_2_image_export_" ,"20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_").replace(data_dir, "")), cv2.COLOR_BGR2GRAY)
            mask = mask == 0
            #     continue
            if classes:
                cls = 1 if "True" in t_cls[no] else 0
                true_cls = np.any(gt == 255).item()

            # if not np.isnan(mask_precision_generous(gt, pred)):
            #     print("UAUYYAYAYAHSKDJAFHLKFHJSJDKLFHSLJKDHFSKLJDHFK")

            t_acc.append(mask_acc(gt, pred))
            t_recall.append(mask_recall(gt, pred))
            t_acc_gen.append(mask_acc_generous(gt, pred, mask=mask))
            t_acc_iou.append(iou_accuracy(gt, pred))
            t_prec_gen.append(mask_precision_generous(gt, pred, mask=mask))
            t_prec.append(mask_precision(gt, pred))


            if "SEM_dauer_2_image_export_s041_Y6_X12_" in file:
                print(mask_recall(gt, pred), mask_precision(gt, pred), mask_precision_generous(gt, pred))

            # if mask_precision_generous(gt, pred) < 0.9:
            #     print(file, mask_precision_generous(gt, pred))

            
            train_preds.append(pred)
            train_gt.append(gt)
            if classes:
                train_classes.append(cls)
                train_gcls.append(true_cls)
        except Exception as E: 
            print(E)
            print(file, segs[i])
    

    print("Training set classifier accuracy {}".format(np.nanmean((np.array(train_classes) == np.array(train_gcls)))))
    print("Training set mask accuracy {}".format(np.nanmean(t_acc)))
    print("Training set mask recall {}".format(np.nanmean(t_recall)))
    print("Training set mask accuracy generous {}".format(np.nanmean(t_acc_gen)))
    print("Training set mask accuracy iou {}".format(np.nanmean(t_acc_iou)))
    print("Training set mask prec {}".format(np.nanmean(t_prec)))
    print("Training set mask prec generous {}".format(np.nanmean(t_prec_gen)))
    print("Sdfsdf", bad_count)


def assemble_predictions(images_dir, preds_dir, gt_dir, overlay=True):
    # superimpose the predictions on the image 
    red = (0, 0, 255) # FP
    green = (0, 255, 0) #TP
    blue = (255, 0, 0) #FN

    preds_colors = [green, red, blue]

    # magenta:
    # magenta = (255, 0, 255) #conf 5
    # cyan = (255, 255, 0) #conf 4
    # brown = (0, 255, 255) #conf 3
    # orange = (0, 165, 255) #conf 2
    # light_blue = (255, 165, 0) #conf 1
    conf5 = (255, 0, 255) #conf 5
    conf4 = (255, 255, 0) #conf 4
    conf3 = (0, 255, 255) #conf 3
    conf2 = (0, 165, 255) #conf 2
    conf1 = (255, 165, 0) #conf 1

    gt_colors = [conf1, conf2, conf3, conf4, conf5]

    def assemble(img, y, x):
        ys = []
        for i in range(0, x):
            ys.append(np.concatenate(img[i*19:(i+1)*19], axis=1))
        assert len(ys) == 17
        img = np.concatenate(ys, axis=0)
        return img

    for s in range(100, 106):
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        for y in tqdm(range(0, 17)):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            for x in range(0, 19):
                suffix = r"s{}_Y{}_X{}".format(s, y, x)
                img = cv2.cvtColor(cv2.imread(images_dir + "SEM_dauer_2_export_" + suffix + ".png", cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
                #color the borders yellow
                img[0:2, :] = [0, 255, 255]
                img[-1:-3, :] = [0, 255, 255]
                img[:, 0:2] = [0, 255, 255]
                img[:, -1:-3] = [0, 255, 255]
                y_acc_img += [img]
                try:
                    pred = cv2.cvtColor(cv2.imread(preds_dir + "SEM_dauer_2_export_" + suffix + ".png"), cv2.COLOR_BGR2GRAY)
                    gt = cv2.cvtColor(cv2.imread(gt_dir + "sem_dauer_2_gj_seg_" + suffix + ".png"), cv2.COLOR_BGR2GRAY)

                    cond = np.logical_and(gt == 2, gt == 15)

                    masked_gt = deepcopy(gt)
                    masked_gt[cond] = 0
                    masked_gt[masked_gt != 0] = 255

                    #make preds color
                    # masked_gt = np.repeat(masked_gt[:, :, np.newaxis], 3, axis=-1)
                    pred_c= np.repeat(pred[:, :, np.newaxis], 3, axis=-1)
                    for m in range(3):
                        # print(np.count_nonzero(np.logical_and(pred[:, :, m] == 255, masked_gt[:, :, m] == 0)))
                        # print(np.count_nonzero(np.logical_and(pred[:, :, m] == 0, masked_gt[:, :, m] == 255)))
                        pred_c[(pred == 255) & (masked_gt == 255)] = green
                        pred_c[np.logical_and(pred == 255, masked_gt == 0)] = red
                        pred_c[np.logical_and(pred == 0, masked_gt == 255)] = blue
                    pred = pred_c
                    # if np.any(pred != 0):
                    #     print(np.unique(pred))
                    #     plt.imshow(pred)
                    #     plt.show()
                        
                    #color borders
                    pred[0:2, :] = [0, 255, 255]
                    pred[-1:-3, :] = [0, 255, 255]
                    pred[:, 0:2] = [0, 255, 255]
                    pred[:, -1:-3] = [0, 255, 255]

                    y_acc_pred += [pred]
                    
                    #make gt color based on confidence
                    colored_gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                    for m in range(3):
                        colored_gt[gt == 3] = conf1
                        colored_gt[gt == 5] = conf2
                        colored_gt[gt == 7] = conf3
                        colored_gt[gt == 9] = conf4
                        colored_gt[gt == 11] = conf5

                    #color borders:
                    colored_gt[0:2, :] = [0, 255, 255]
                    colored_gt[-1:-3, :] = [0, 255, 255]
                    colored_gt[:, 0:2] = [0, 255, 255]
                    colored_gt[:, -1:-3] = [0, 255, 255]

                    y_acc_gt += [colored_gt]

                except:
                    print(suffix)

                    y_acc_pred += [np.zeros_like(img)]
                    #paint borders
                    y_acc_pred[-1][0:2, :] = [0, 255, 255]
                    y_acc_pred[-1][-1:-3, :] = [0, 255, 255]
                    y_acc_pred[-1][:, 0:2] = [0, 255, 255]
                    y_acc_pred[-1][:, -1:-3] = [0, 255, 255]

                    y_acc_gt += [np.zeros_like(img)]
                    #paint borders
                    y_acc_gt[-1][0:2, :] = [0, 255, 255]
                    y_acc_gt[-1][-1:-3, :] = [0, 255, 255]
                    y_acc_gt[-1][:, 0:2] = [0, 255, 255]
                    y_acc_gt[-1][:, -1:-3] = [0, 255, 255]

            s_acc_img += [np.concatenate(y_acc_img, axis=1)]
            s_acc_pred += [np.concatenate(y_acc_pred, axis=1)]
            s_acc_gt += [np.concatenate(y_acc_gt, axis=1)]
        
        # new_img = assemble(s_acc_img, 17, 19)
        # new_pred = assemble(s_acc_pred, 17, 19)
        # new_gt = assemble(s_acc_gt, 17, 19)
        
        new_img = np.concatenate(s_acc_img, axis=0)
        new_pred = np.concatenate(s_acc_pred, axis=0)
        new_gt = np.concatenate(s_acc_gt, axis=0)

        def write_legend(text, color, img, org):
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 

            
            # fontScale 
            fontScale = 10
            
            # Line thickness of 2 px 
            thickness = 20

            return cv2.putText(img, text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
        

        #write them all in
        save_dir = "/Volumes/Normal/gapjnc/resuklts/assembledpredsnew512focal/"
        save_dir += "SEM_dauer_2_image_export_" + suffix + "/"

        if overlay:
            new_pred = cv2.addWeighted(new_img, 0.5, new_pred, 0.1, 0)
            new_gt = cv2.addWeighted(new_img, 0.5, new_gt, 0.1, 0)

        for color in gt_colors:
            new_gt = write_legend(f"Confidence {gt_colors.index(color)}", color, new_gt, (450, 450+250*gt_colors.index(color)))

        for color in preds_colors:
            new_pred = write_legend(f"{['TP', 'FP', 'FN'][preds_colors.index(color)]}", color, new_pred, (450, 450+250*preds_colors.index(color)))

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_img.png", new_img)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_pred.png", new_pred)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_gt.png", new_gt)


def center_img(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Apply thresholding
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate center of each contour
    sub_images = []
    centers = []
    sub_copy = img.copy()
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(cx, cy)
            centers.append((cx, cy))

    return centers

def split_img(img, offset=256, tile_size=512, names=False):
    if offset:
        img = img[offset:-offset, offset:-offset]
    imgs = []
    names = []
    for i in range(img.shape[0]//tile_size+1):
        for j in range(img.shape[1]//tile_size +1):
            imgs.append(img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size])
            names.append("Y{}_X{}".format(i, j))
    return (imgs, names) if names else imgs 

def centered_dataset(imgs_dir, gt_dir, mito_dir=None, neuron_dir=None, save_dir=None):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, "imgs"))
    os.mkdir(os.path.join(save_dir, "gts"))
    if mito_dir: os.mkdir(os.path.join(save_dir, "mitos"))
    if neuron_dir: os.mkdir(os.path.join(save_dir, "neurons"))

    imgs = sorted(os.listdir(imgs_dir))
    gts = sorted(os.listdir(gt_dir))
    if mito_dir: mitos = sorted(os.listdir(mito_dir))
    if neuron_dir: neurons = sorted(os.listdir(neuron_dir))

    for i in tqdm(range(len(imgs))):
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]))
        gt = cv2.imread(os.path.join(gt_dir, gts[i]))
        if mito_dir: mito = cv2.imread(os.path.join(mito_dir, mitos[i]))
        if neuron_dir: neuron = cv2.imread(os.path.join(neuron_dir, neurons[i]))

        (img_imgs, img_names), (gt_imgs, _), (mito_imgs, _), (neuron_imgs, _) = split_img(img, 0, names=True), split_img(gt, 0), split_img(mito, 0) if mito_dir else ([], []), split_img(neuron, 0) if neuron_dir else ([] ,[])
        for j in range(len(img_imgs)):
            cv2.imwrite(os.path.join(save_dir, f"imgs/{imgs[i].replace('.png', '_'+img_names[j])}.png"), img_imgs[j])
            cv2.imwrite(os.path.join(save_dir, f"gts/{gts[i].replace('.png', '_'+img_names[j])}.png"), gt_imgs[j])
            if mito_dir: cv2.imwrite(os.path.join(save_dir, f"mitos/{imgs[i].replace('.png', '_'+img_names[j])}.png"), mito_imgs[j])
            if neuron_dir: cv2.imwrite(os.path.join(save_dir, f"neurons/{imgs[i].replace('.png', '_'+img_names[j])}.png"), neuron_imgs[j])
        
        (img_imgs, img_names), (gt_imgs, _), (mito_imgs, _), (neuron_imgs, _) = split_img(img, names=True), split_img(gt), split_img(mito) if mito_dir else ([], []), split_img(neuron) if neuron_dir else ([], [])

        for j in range(len(img_imgs)):
            cv2.imwrite(os.path.join(save_dir, f"imgs/{imgs[i].replace('.png', '_'+img_names[j]+'off')}.png"), img_imgs[j])
            cv2.imwrite(os.path.join(save_dir, f"gts/{gts[i].replace('.png', '_'+img_names[j]+'off')}.png"), gt_imgs[j])
            if mito_dir: cv2.imwrite(os.path.join(save_dir, f"mitos/{imgs[i].replace('.png', '_'+img_names[j]+'off')}.png"), mito_imgs[j])
            if neuron_dir: cv2.imwrite(os.path.join(save_dir, f"neurons/{imgs[i].replace('.png', '_'+img_names[j]+'off')}.png"), neuron_imgs[j])
            
def assemble_overlap(img_dir, gt_dir, pred_dir, save_dir, overlap=True):
    for s in range(101, 109):
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        s_acc_img1, s_acc_pred1, s_acc_gt1 = [], [], []
        for y in tqdm(range(0, 17)):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            y_acc_img1, y_acc_pred1, y_acc_gt1 = [], [], []
            for x in range(0, 19):
                suffix = r"s{}_Y{}_X{}".format(s, y, x)
                im =cv2.cvtColor(cv2.imread(img_dir + "SEM_dauer_2_em_" + suffix + ".png"), cv2.COLOR_BGR2GRAY)
                try:
                    gt = cv2.cvtColor(cv2.imread(gt_dir + "sem_dauer_2_gj_gt_" + suffix + ".png"), cv2.COLOR_BGR2GRAY)
                    pred = cv2.cvtColor(cv2.imread(pred_dir + "SEM_dauer_2_em_" + suffix + ".png"), cv2.COLOR_BGR2GRAY)
                    if gt.shape != pred.shape: pred = np.zeros_like(gt)  
                except:
                    gt = np.zeros_like(im)
                    pred = np.zeros_like(im)

                #just append as is
                y_acc_img.append(im)
                y_acc_pred.append(pred)
                y_acc_gt.append(gt)

                try:
                    im1 = cv2.cvtColor(cv2.imread(img_dir + "SEM_dauer_2_em_" + suffix + "off.png"), cv2.COLOR_BGR2GRAY)
                    gt1 = cv2.cvtColor(cv2.imread(gt_dir + "sem_dauer_2_gj_gt_" + suffix + "off.png"), cv2.COLOR_BGR2GRAY)
                    pred1 = cv2.cvtColor(cv2.imread(pred_dir + "SEM_dauer_2_em_" + suffix + "off.png"), cv2.COLOR_BGR2GRAY)
                    if im1 is None or gt1 is None or pred1 is None: raise Exception
                    if gt1.shape != pred1.shape: pred1 = np.zeros_like(gt1)  
                    y_acc_img1.append(im1)
                    y_acc_pred1.append(pred1)
                    y_acc_gt1.append(gt1)
                except:
                    # gt1 = np.zeros_like(im1)
                    # pred1 = np.zeros_like(im1)
                    continue

            s_acc_img.append(np.concatenate(y_acc_img, axis=1))
            s_acc_pred.append(np.concatenate(y_acc_pred, axis=1))
            s_acc_gt.append(np.concatenate(y_acc_gt, axis=1))
            try:
                s_acc_img1.append(np.concatenate(y_acc_img1, axis=1))
                s_acc_pred1.append(np.concatenate(y_acc_pred1, axis=1))
                s_acc_gt1.append(np.concatenate(y_acc_gt1, axis=1))
            except: continue

        new_img = np.concatenate(s_acc_img, axis=0)
        new_pred = np.concatenate(s_acc_pred, axis=0)
        new_gt = np.concatenate(s_acc_gt, axis=0)

        new_img1 = np.concatenate(s_acc_img1, axis=0)
        new_pred1 = np.concatenate(s_acc_pred1, axis=0)
        new_gt1 = np.concatenate(s_acc_gt1, axis=0)

        #calculate and color statsitic
        new_gt[(new_gt == 2) | (new_gt == 15)] = 0
        new_gt_conf = new_gt.copy()
        new_gt[new_gt != 0] = 1

        new_gt1[(new_gt1 == 2) | (new_gt1 == 15)] = 0
        new_gt1[new_gt1 != 0] = 1
        
        new_pred[new_pred !=0 ] = 1
        new_pred1[new_pred1 !=0 ] = 1

        new_gt[256:-256, 256:-256] = new_gt[256:-256, 256:-256] | new_gt1
        new_pred[256:-256, 256:-256] = new_pred[256:-256, 256:-256] | new_pred1

        new_img1 = np.repeat(new_img[:, :, np.newaxis], 3, axis=-1)
        new_gt1 = np.repeat(new_gt[:, :, np.newaxis], 3, axis=-1)
        new_pred1 = np.repeat(new_pred[:, :, np.newaxis], 3, axis=-1)

        
        #color statistics now
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        for m in range(3):
            new_pred1[(new_pred == 1) & (new_gt == 1)] = green
            new_pred1[(new_pred == 1) & (new_gt == 0)] = red
            new_pred1[(new_pred == 0) & (new_gt == 1)] = blue
        
        #color confidence levels
        conf5 = (255, 0, 255)
        conf4 = (255, 255, 0)
        conf3 = (0, 255, 255)
        conf2 = (0, 165, 255)
        conf1 = (255, 165, 0)

        #make labels
        for m in range(3):
            new_gt1[new_gt_conf == 3] = conf1
            new_gt1[new_gt_conf == 5] = conf2
            new_gt1[new_gt_conf == 7] = conf3
            new_gt1[new_gt_conf == 9] = conf4
            new_gt1[new_gt_conf == 11] = conf5

        #make legends
        def write_legend(text, color, img, org):
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 

            
            # fontScale 
            fontScale = 10
            
            # Line thickness of 2 px 
            thickness = 20

            return cv2.putText(img, text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
        
        #measure stats
        # assembled_stats(new_gt_conf, new_pred, )


        if overlap:
            new_pred1 = cv2.addWeighted(new_img1, 0.5, new_pred1, 0.1, 0)
            new_gt1 = cv2.addWeighted(new_img1, 0.5, new_gt1, 0.1, 0)
        
        for color in [conf1, conf2, conf3, conf4, conf5]:
            new_gt = write_legend(f"Confidence {[conf1, conf2, conf3, conf4, conf5].index(color)}", color, new_gt1, (450, 450+250*[conf1, conf2, conf3, conf4, conf5].index(color)))
        
        for color in [green, red, blue]:
            new_pred = write_legend(f"{['TP', 'FP', 'FN'][[green, red, blue].index(color)]}", color, new_pred1, (450, 450+250*[green, red, blue].index(color)))
        
        #write them all in
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_img.png", new_img1)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_pred.png", new_pred1)
        assert cv2.imwrite(save_dir + "SEM_dauer_2_image_export_" + suffix + "_gt.png", new_gt1)

def assembled_stats(gt_image, preds_image, nr_mask):
    recall = mask_recall(gt_image, preds_image, mask=nr_mask)
    precision = mask_precision(gt_image, preds_image, mask=nr_mask)
    precision_gen = mask_precision_generous(gt_image, preds_image, mask=nr_mask)
    acc = mask_acc(gt_image, preds_image, mask=nr_mask)
    acc_gen = mask_acc_generous(gt_image, preds_image, mask=nr_mask)
    
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Precision Generous: ", precision_gen)
    print("Accuracy: ", acc)
    print("Accuracy Generous: ", acc_gen)


if __name__ == "__main__":
    # save_dir = "/home/mishaalk/scratch/gapjunc/results/stitched_mito_preds/"
    # left = {'SEM_dauer_2_image_export_s023_Y7_X13_img.png', 'SEM_dauer_2_image_export_s033_Y11_X6_img.png', 'SEM_dauer_2_image_export_s042_Y9_X11_img.png', 'SEM_dauer_2_image_export_s009_Y7_X15_img.png', 'SEM_dauer_2_image_export_s047_Y10_X8_img.png', 'SEM_dauer_2_image_export_s050_Y9_X7_img.png', 'SEM_dauer_2_image_export_s038_Y4_X10_img.png', 'SEM_dauer_2_image_export_s014_Y10_X11_img.png', 'SEM_dauer_2_image_export_s004_Y8_X5_img.png', 'SEM_dauer_2_image_export_s028_Y5_X11_img.png', 'SEM_dauer_2_image_export_s018_Y6_X15_img.png'}

    # for file in tqdm(left):
    #     img_acc, pred_acc, seg_acc = [], [], []
    #     for i in range(16):
    #         im = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/imgs/"+file[:-8]+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         pred = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/results/tiny_mito_preds/"+file[:-8]+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         seg = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/gts/"+file[:-8].replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_")+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         img_acc.append(im)
    #         pred_acc.append(pred)
    #         seg_acc.append(seg)
    #     img_acc = np.concatenate([np.concatenate(img_acc[:4], axis=1), np.concatenate(img_acc[4:8], axis=1), np.concatenate(img_acc[8:12], axis=1), np.concatenate(img_acc[12:16], axis=1)], axis=0) 
    #     seg_acc = np.concatenate([np.concatenate(seg_acc[:4], axis=1), np.concatenate(seg_acc[4:8], axis=1), np.concatenate(seg_acc[8:12], axis=1), np.concatenate(seg_acc[12:16], axis=1)], axis=0) 
    #     pred_acc = np.concatenate([np.concatenate(pred_acc[:4], axis=1), np.concatenate(pred_acc[4:8], axis=1), np.concatenate(pred_acc[8:12], axis=1), np.concatenate(pred_acc[12:16], axis=1)], axis=0) 

    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_pred.png")), file
    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_img.png")), file
    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_seg.png")), file

        
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_pred.png"), pred_acc)
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_seg.png"), seg_acc)
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_img.png"), img_acc)
    # img_files = os.listdir(BASE+"sem_dauer_2_10_125/")
    # count_write = 0
    # for i in tqdm(range(len(img_files))):
    #     k = int(re.findall(r's\d\d\d', img_files[i])[0][1:])
    #     if k > 105: continue
    #     count_write+=1
    #     img = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2_10_125/"+img_files[i]), cv2.COLOR_BGR2GRAY)

    #     #split the image into 32x32 bits
    #     imgs = ressample(img, 128)
    #     # save these 
    #     counter=0
    #     for j in range(len(imgs)):
    #         assert cv2.imwrite("/Users/mishaal/tiny_test/{}.png".format(img_files[i][:-4] + "_"+str(counter)), imgs[j])
    #         counter+=1
    # print(counter, count_write)
    # mask_acc_split("/Volumes/Normal/gapjnc/gj_segmentation/", "/Volumes/Normal/gapjnc/resuklts/newgendicerun2testset/", classes=False, test=True)
    # stitch_short("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/",\
    #               "/home/mishaalk/scratch/gapjunc/results/tiny_mito_preds/",\
    #               "/home/mishaalk/scratch/gapjunc/results/stitched_mito_preds/")


    # erase_neurons()
    # shorten_pics()
    # import shutil

    # os.mkdir(BASE+"tiny_jnc_only\\gts1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\imgs1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\masks1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\mito_masks1\\")

    # files = sorted(os.listdir(BASE+"tiny_jnc_only\\gts\\"))
    # images = sorted(os.listdir(BASE+"tiny_jnc_only\\imgs\\"))
    # masks = sorted(os.listdir(BASE+"tiny_jnc_only\\masks\\"))
    # mito_masks = sorted(os.listdir(BASE+"tiny_jnc_only\\mito_masks\\"))

    # #class balancing (65-35)
    # empty_segs, present_segs = [], []
    # for i in tqdm(range(len(files))):
    #     im = cv2.cvtColor(cv2.imread(BASE+"tiny_jnc_only\\gts\\"+files[i]), cv2.COLOR_BGR2GRAY)
    #     if len(np.unique(im)) == 2: present_segs.append(i)
    #     else: empty_segs.append(i)
    # n_present = len(present_segs)
    # total = (n_present *100)//65
    # empty_segs = np.random.choice(np.array(empty_segs), size=7800, replace=False).tolist()
    # all_segs = empty_segs + present_segs

    # for i in tqdm(range(len(files))):
    #     if i not in all_segs: continue
    #     shutil.copyfile(BASE+"tiny_jnc_only\\gts\\"+files[i],BASE+"tiny_jnc_only\\gts1\\"+files[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\masks\\"+masks[i],BASE+"tiny_jnc_only\\masks1\\"+masks[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\imgs\\"+files[i].replace("sem2dauer_gj_2d_training.vsseg_export", "SEM_dauer_2_image_export"),BASE+"tiny_jnc_only\\imgs1\\"+images[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\mito_masks\\"+mito_masks[i],BASE+"tiny_jnc_only\\mito_masks1\\"+mito_masks[i])

    # print("Class balanced tiny has {} empty and {} present with a total of {} points".format(empty_segs, present_segs, len(all_segs)))


    # only_junc_images_datast()
    # neuron_present_images_dataset()

    # from shutil import copyfile 
    # for line in open(BASE+"jnc_only_images.txt", 'r'): 
    #     filename=BASE+"image_export\\" + line.strip() 
    #     dest=BASE+"jnc_only_images\\"+ line.strip() 
    #     copyfile(filename, dest)

    #make a jnc_only_dataset
    # os.mkdir(BASE+"jnc_only_dataset\\")
    # os.mkdir(BASE+"jnc_only_dataset\\gts\\")
    # os.mkdir(BASE+"jnc_only_dataset\\imgs\\")
    # for sec in ["train", "test", "valid"]:
    #     BASE = "E:\\Mishaal\\GapJunction\\small_data\\ground_truth\\"
    #     # img_files = sorted(os.listdir(BASE+"jnc_only_images\\"))
    #     seg_files = sorted(os.listdir(BASE+sec+"\\"))
    #     os.mkdir(BASE+sec+"_new\\")

    #     for i in tqdm(range(len(seg_files))):
    #         # img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
    #         seg = cv2.cvtColor(cv2.imread(BASE+sec+"\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)
    #         if len(np.unique(seg)) > 2: raise Exception(np.unique(seg))
    #         seg[seg == 30] = 0
    #         seg[seg == 215] = 255

    #         cv2.imwrite(BASE+sec+"_new\\{}".format(seg_files[i]), seg)
    #         # cv2.imwrite(BASE+"jnc_only_dataset\\imgs\\{}.png".format(img_files[i]), img)

    # preds_dir = "/Volumes/Normal/gapjnc/resuklts/tinymitobesttest/"
    # pred_files = os.listdir("/Volumes/Normal/gapjnc/resuklts/tinymitobesttest/")
    # # pred_files.remove("classes.csv")
    # pred_files = sorted(pred_files, key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))

    # # all_saves_list = ThreadSafeList()

    # def helper_inside(img_files, seg_files, pred_files, save_dir):
    #     img_acc, seg_acc, pred_acc = [], [], []
    #     base_count = None
    #     for i in tqdm(range(0, len(img_files))):
    #         if i % 16 == 0 and i != 0:
    #             # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))
    #             pred_acc = np.concatenate([np.concatenate(pred_acc[:4], axis=1), np.concatenate(pred_acc[4:8], axis=1), np.concatenate(pred_acc[8:12], axis=1), np.concatenate(pred_acc[12:16], axis=1)], axis=0) 
    #             #all_saves_list.append((pred_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))))
    #             cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred.")), pred_acc)
    #             pred_acc = []
    #         base_count = pred_files[i]
    #         pred = cv2.cvtColor(cv2.imread(preds_dir+pred_files[i]), cv2.COLOR_BGR2GRAY)
    #         pred_acc.append(pred)
    #     print("I did {} files".format(len(img_files)))

    # # offsets = len(pred_files)//16
    # # threads = []
    # # for j in range(12):
    # #     t = threading.Thread(target=helper_inside, args=(pred_files[j*245*16 : (j+1)*245*16], pred_files[j*245*16 : (j+1)*245*16], pred_files[j*245*16 : (j+1)*245*16], "/Volumes/Normal/gapjnc/resuklts/stitchedtinymmitobesttst/"))
    # #     threads.append(t)

    # # [t.start() for t in threads]
    # # [t.join() for t in threads]
    # helper_inside(pred_files, pred_files, pred_files, "/Volumes/Normal/gapjnc/resuklts/stitchedtinymmitobesttst/")


    # assemble_predictions("/Volumes/Normal/gapjnc/sem_dauer_2_10_125/", "/Volumes/Normal/gapjnc/resuklts/newgendicerun2testset/", "/Volumes/Normal/gapjnc/gj_segmentation/", False)
    # centered_dataset("/Volumes/Normal/gapjnc/sections100_125/sem_dauer_2_em/", "/Volumes/Normal/gapjnc/sections100_125/sem_dauer_2_gj_gt", save_dir="/Volumes/Normal/gapjnc/train_100_110/")         


    assemble_overlap("/Volumes/Normal/gapjnc/train_100_110/imgs/", "/Volumes/Normal/gapjnc/train_100_110/gts/", "/Volumes/Normal/gapjnc/resuklts/new3dfocaltest/", "/Volumes/Normal/gapjnc/resuklts/assembledpreds3dfocal/")
