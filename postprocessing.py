import cv2, numpy as np, re, os, sys, shutil, glob, traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, copy

import networkx as nx

#assembling function for full image assembly
"""
Function for assembling the prediction tiles into a single EM image.
@param img_dir: directory containing the original images
@param gt_dir: directory containing the ground truth images
@param pred_dir: directory containing the predicted images
@param save_dir: directory to save the assembled images
@param overlap: whether to overlap the predictions on the EM images
@param missing_dir: directory containing the missing images - when your dataset was created without certain images (e.g. images without any neurons in them)
@param img_templ: template for the image names
@param seg_templ: template for the ground truth image names
@param s_range: range of s values
@param x_range: range of x values
@param y_range: range of y values
@param offset: offset for the overlap (default 256)
@returns: None (saves the images in the save_dir)
"""
def assemble_overlap(img_dir, gt_dir, pred_dir, save_dir, extend_dir=None, overlap=True, missing_dir=None, img_templ="SEM_dauer_2_em_",seg_templ="sem_dauer_2_gj_gt_" , s_range=range(101, 109), x_range=range(0, 19), y_range=range(0, 17), offset=256, fn=None, fn_mask_dir=None, fn_cell_dir=None, fn_filter=False, plot_legend=True):
    assert (fn_filter and fn_cell_dir) or not fn_filter
    
    offset = int(offset)
    if fn: fn_stats = []
    for s in s_range:
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        if offset: s_acc_img1, s_acc_pred1, s_acc_gt1 = [], [], []
        for y in tqdm(y_range):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            if offset: y_acc_img1, y_acc_pred1, y_acc_gt1 = [], [], []
            for x in x_range:
                suffix = r"s{}_Y{}_X{}".format('0'*(3-len(str(s)))+str(s), y, x)

                if not os.path.isfile(os.path.join(img_dir, img_templ +suffix + ".png")): 
                    assert missing_dir is not None, f"Missing image {img_templ +suffix + '.png'}"
                    shutil.copy(os.path.join(missing_dir, img_templ + suffix + ".png"), os.path.join(img_dir, img_templ + suffix + ".png"))
                im =cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_templ+ suffix + ".png")), cv2.COLOR_BGR2GRAY)
                
                try:
                    if gt_dir: 
                        gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, seg_templ + suffix + "_label.png")), cv2.COLOR_BGR2GRAY) 

                    pred = cv2.imread(os.path.join(pred_dir, img_templ + suffix + "_pred.png"), cv2.IMREAD_GRAYSCALE)
                    if pred is None: raise Exception
                    
                    #pred = 1/(1+np.exp(-pred)) >= 0.5 #Apply sigmoid function WHY?

                    #softmax on channel 1 of size 3
                    # pred = np.argmax(pred, axis=0) == 1

                    if extend_dir:
                        try:
                            ext_pred = cv2.imread(os.path.join(extend_dir, img_templ + suffix + ".png"))
                            ext_pred = cv2.cvtColor(ext_pred, cv2.COLOR_BGR2GRAY)
                            ext_pred[ext_pred != 0] = 255
                            if (np.count_nonzero((pred == 0) & (ext_pred == 255))):
                                cv2.imwrite("extas.png", ((pred == 0) & (ext_pred == 255))*255)
                                print(f"EXTRA EXTENDS IN {suffix}")
                            pred = pred | (ext_pred == 255)
                            print(np.unique(pred))
                            
                        except:
                            pass
                    if gt_dir and gt.shape != pred.shape: pred = np.zeros_like(gt)  
                except Exception as e:
                    print(e)
                    if gt_dir: gt = np.zeros_like(im)
                    pred = np.zeros_like(im)

                #just append as is
                y_acc_img.append(im)
                y_acc_pred.append(pred)
                if gt_dir: y_acc_gt.append(gt)

                if offset:
                    try:
                        im1 = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_templ +suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                        if gt_dir: gt1 = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, seg_templ + suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                        if True:
                            pred1 = np.load(os.path.join(pred_dir, img_templ + suffix + "off.png")) #off.png.npy
                            pred1 = 1/(1+np.exp(-pred1)) >= 0.5

                            # pred1 = np.argmax(pred, axis=0)
                            # pred1[pred1 != 1] = 0
                        else:
                             pred1 = cv2.cvtColor(cv2.imread(os.path.join(pred_dir, img_templ + suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                             pred1 = pred1 == 255
                        if extend_dir:
                            try:
                                ext_pred1 = cv2.imread(os.path.join(extend_dir, img_templ + suffix + "off.png"))
                                ext_pred1 = cv2.cvtColor(ext_pred1, cv2.COLOR_BGR2GRAY)
                                ext_pred1[ext_pred1 != 0] = 255
                                pred1 = pred1 | (ext_pred1 == 255)
                            except:
                                print(pred1.shape)
                                pass

                        # if im1 is None or gt1 is None or pred1 is None: raise Exception
                        if gt_dir and gt1.shape != pred1.shape: pred1 = np.zeros_like(gt1)  
                        y_acc_img1.append(im1)
                        y_acc_pred1.append(pred1)
                        if gt_dir: y_acc_gt1.append(gt1)
                    except Exception as e:
                        print(e)
                        gt1 = np.zeros_like(im1)
                        pred1 = np.zeros_like(im1)
                        # continue

            s_acc_img.append(np.concatenate(y_acc_img, axis=1))
            s_acc_pred.append(np.concatenate(y_acc_pred, axis=1))
            if gt_dir: s_acc_gt.append(np.concatenate(y_acc_gt, axis=1))
            if offset:
                try:
                    s_acc_img1.append(np.concatenate(y_acc_img1, axis=1))
                    s_acc_pred1.append(np.concatenate(y_acc_pred1, axis=1))
                    if gt_dir: s_acc_gt1.append(np.concatenate(y_acc_gt1, axis=1))
                except Exception as e:
                    print(e)
                    continue

        new_img = np.concatenate(s_acc_img, axis=0)
        new_pred = np.concatenate(s_acc_pred, axis=0)
        if gt_dir: new_gt = np.concatenate(s_acc_gt, axis=0)

        if offset:
            new_img1 = np.concatenate(s_acc_img1, axis=0)
            new_pred1 = np.concatenate(s_acc_pred1, axis=0)
            if gt_dir: new_gt1 = np.concatenate(s_acc_gt1, axis=0)

        #calculate and color statistic
        if gt_dir:
            new_gt[(new_gt == 2) | (new_gt == 15)] = 0
            new_gt_conf = new_gt.copy()
            new_gt[new_gt != 0] = 1

        if offset and gt_dir:
            new_gt1[(new_gt1 == 2) | (new_gt1 == 15)] = 0
            new_gt1[new_gt1 != 0] = 1
        
        new_pred[new_pred !=0 ] = 255
        if offset: new_pred1[new_pred1 !=0 ] = 1
        if offset:
            if gt_dir: new_gt[offset:-offset, offset:-offset] = new_gt[offset:-offset, offset:-offset] | new_gt1
            try:
                new_pred[offset:-offset, offset:-offset] = new_pred[offset:-offset, offset:-offset] | new_pred1
            except:
                new_pred1 = new_pred1[:-offset, :-offset]
                # new_pred1 = np.concatenate([np.zeros((offset//128, new_pred1.shape[1])), new_pred1], axis=0) == 1
                new_pred[offset:, offset:] = new_pred[offset:,  offset:] | new_pred1
                # print(new_pred.shape, new_pred1.shape)
                # sys.exit()
                # plt.imshow(new_pred)
                # plt.show()

        if fn_filter:
            mask_membrane = glob.glob(fn_cell_dir+f"*s{'0'*(3-len(str(s)))}{s}*")[0]
            mask_membrane = cv2.imread(mask_membrane, -1)

            t = np.zeros_like(new_pred).astype(mask_membrane.dtype)
            t[:mask_membrane.shape[0], :mask_membrane.shape[1]] = mask_membrane
            mask_membrane = t

            #load nr_mask 
            mask = glob.glob(fn_mask_dir+f"*s{'0'*(3-len(str(s)))}{s}*")[0]
            mask = cv2.imread(mask, -1)
            mask = mask == 5497
            mask = mask.astype(np.uint8)    

            temp = np.zeros_like(new_pred).astype(np.uint8)
            temp[:mask.shape[0], :mask.shape[1]] = mask
            mask = temp

            new_pred = new_pred * mask
            new_pred, _ = filter_islands_and_inside(new_pred, mask_membrane)

        t = np.zeros_like(new_pred).astype(np.uint8)
        t[:new_img.shape[0], :new_img.shape[1]] = new_img
        new_img = t
        new_img1 = np.repeat(new_img[:, :, np.newaxis], 3, axis=-1)
        if gt_dir: new_gt1 = np.repeat(new_gt[:, :, np.newaxis], 3, axis=-1)
        #new_pred1 = np.repeat(new_pred[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)    
        new_pred1 = new_pred.astype(np.uint8)          

        #Update suffix to include only the section
        suffix = f"s{'0'*(3-len(str(s)))+str(s)}"

        if not fn and plot_legend:
            # else: 
            #color statistics now
            red = (0, 0, 255)
            green = (0, 255, 0)
            blue = (255, 0, 0)
            # for m in range(3):
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
            # for m in range(3):
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

            if overlap:
                new_pred1 = cv2.addWeighted(new_img1, 0.5, new_pred1, 0.6, 0)
                new_gt1 = cv2.addWeighted(new_img1, 0.5, new_gt1, 0.6, 0)
            
            for color in [conf1, conf2, conf3, conf4, conf5]:
                new_gt = write_legend(f"Confidence {[conf1, conf2, conf3, conf4, conf5].index(color)}", color, new_gt1, (450, 450+250*[conf1, conf2, conf3, conf4, conf5].index(color)))
            
            for color in [green, red, blue]:
                new_pred = write_legend(f"{['TP', 'FP', 'FN'][[green, red, blue].index(color)]}", color, new_pred1, (450, 450+250*[green, red, blue].index(color)))
            
            #write them all in
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            assert cv2.imwrite(os.path.join(save_dir, img_templ + suffix + "_img.png"), new_img1)
            assert cv2.imwrite(os.path.join(save_dir, img_templ + suffix + "_pred.png"), new_pred1)
            assert cv2.imwrite(os.path.join(save_dir, seg_templ + suffix + "_label.png"), new_gt1)
        elif not fn and not plot_legend:
            if gt_dir: new_gt1[new_gt == 1] = (255, 255, 255)
            #new_pred1[new_pred == 1] = (255, 0, 0)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            assert cv2.imwrite(os.path.join(save_dir, img_templ + suffix + "_img.png"), new_img1)
            assert cv2.imwrite(os.path.join(save_dir, img_templ + suffix + "_pred.png"), new_pred1)
            if gt_dir: assert cv2.imwrite(os.path.join(save_dir, seg_templ + suffix + "_label.png"), new_gt1)
        else: 
            mask = glob.glob(fn_mask_dir+f"*s{'0'*(3-len(str(s)))}{s}*")[0]
            mask = cv2.imread(mask, -1)
            mask = mask == 5497
            mask = mask.astype(np.uint8)

            if fn_cell_dir:
                cell_id_mask = glob.glob(fn_cell_dir+f"*s{'0'*(3-len(str(s)))}{s}*")[0]
                cell_id_mask = cv2.imread(cell_id_mask, -1)
                cell_id_mask = cell_id_mask != 0
                cell_id_mask = cell_id_mask.astype(np.uint8)
            else: cell_id_mask = None
            fn_stats.append(fn(new_gt_conf, new_pred, nr_mask=mask, cell_id_membrane=cell_id_mask, filtering=fn_filter))

    if fn: return fn_stats

#statistics
def mask_acc(gt, pred, mask=None):
    gt = (gt*mask).flatten()
    pred = (pred*mask).flatten()
    gt[gt != 0] = 255  

    return np.sum(gt == pred) / (gt.shape[0])
    # return np.sum(gt == pred) / (gt.shape[0] * gt.shape[1])

def mask_precision(gt, pred, mask=None):

    gt = (gt*mask).flatten()
    pred = (pred*mask).flatten()
    gt[gt != 0] = 255  

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

    gt = (gt*mask).flatten()
    pred = (pred*mask).flatten()
    new_gt = (new_gt*mask).flatten()
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
    
    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    new_gt = new_gt.flatten()[mask]
    new_gt[new_gt != 0] = 255
    gt[gt != 0] = 255         

    num = ((new_gt == pred) +(gt == pred)) >= 1
    denom = (gt.shape[0])
    return np.sum(num)/denom

def iou_accuracy(gt, pred, mask=None):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(np.logical_or(gt == 255, pred == 255))

def mask_recall(gt, pred, mask=None):
    print(gt.shape, pred.shape, mask.shape)
    gt = (gt*mask).flatten()
    pred = (pred*mask).flatten()
    gt[gt != 0] = 255   

    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(gt == 255)
    
"""
@params: gt_image: ground truth image
@params: preds_image: predicted image
@params: nr_mask: mask for the non-relevant areas
@params: seg_bads: tuple of segmentation labels to ignore in statistics (e.g GJs you are not sure about)
@returns: recall, precision, precision_gen, acc, acc_gen
"""
def assembled_stats(gt_image, preds_image, nr_mask=None, seg_bads=(2, 15)):
    if seg_bads:
        mask = gt_image != seg_bads[0]
        for i in seg_bads[1:]:
            mask = mask & (mask != i)
        # mask = mask.flatten()

    good_mask = np.logical_and(gt_image != 2, gt_image != 15) # TODO: for general case use above
    assert np.count_nonzero(good_mask) == np.count_nonzero(mask)

    if nr_mask is not None:
        nr_mask = np.logical_and(good_mask, nr_mask)
    else: nr_mask = np.ones_like(good_mask)

    if 1 in np.unique(preds_image):
        preds_image *= 255

    recall = mask_recall(gt_image, preds_image, mask=nr_mask)
    precision = mask_precision(gt_image, preds_image, mask=nr_mask)
    precision_gen = mask_precision_generous(gt_image, preds_image, mask=nr_mask)
    acc = mask_acc(gt_image, preds_image, mask=nr_mask)
    acc_gen = mask_acc_generous(gt_image, preds_image, mask=nr_mask)

    return recall, precision, precision_gen, acc, acc_gen

#get stats on a split dataset:
def mask_acc_split(seg_dir, results_dir, nr_mask_dir=None, td=False, breakdown=False, preds_to_seg_map=lambda x:x, preds_to_nr_map=lambda x:x):
    segs = sorted(os.listdir(seg_dir))
    preds = sorted(os.listdir(results_dir))

    t_recall, t_acc = [], []
    t_acc_gen = []
    t_prec, t_prec_gen = [], []

    t_45_recall, t_3_recall, t_2_recall, t_1_recall = [], [], [], []

    weird_shape_count = 0

    for i, file in tqdm(enumerate(preds), total=len(preds)): 
        if "DS" in file: continue #villainous
        try:
            segs[i] = os.path.join(seg_dir, preds_to_seg_map(file))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY)
            no = re.findall(r'_\d+', file)[0]
            pred = cv2.cvtColor(cv2.imread(os.path.join(results_dir, file)), cv2.COLOR_BGR2GRAY)

            #area mask 
            if nr_mask_dir:
                mask = cv2.imread(os.path.join(nr_mask_dir, preds_to_nr_map(file)), -1)
                mask = mask == 5497
            else: mask = np.ones_like(gt)
            if breakdown:
                test_gt = gt.copy()

            t_acc.append(mask_acc(gt, pred, mask=mask))
            t_recall.append(mask_recall(gt, pred, mask=mask))
            t_acc_gen.append(mask_acc_generous(gt, pred, mask=mask))
            t_prec_gen.append(mask_precision_generous(gt, pred, mask=mask))
            t_prec.append(mask_precision(gt, pred, mask=mask))

            if breakdown:
                t_45_recall.append(mask_recall((np.where(test_gt == 11, 1, 0) | np.where(test_gt == 9, 1, 0)) * 255, pred, mask=mask))
                t_3_recall.append(mask_recall(np.where(test_gt == 7, 255, 0), pred, mask=mask))
                t_2_recall.append(mask_recall(np.where(test_gt == 5, 255, 0), pred, mask=mask))
                t_1_recall.append(mask_recall(np.where(test_gt == 3, 255, 0), pred, mask=mask))

        except Exception:
            segs[i] = os.path.join(seg_dir, preds_to_seg_map(file))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY) 
            assert not np.count_nonzero(gt[(gt != 2) & (gt != 15)])
            weird_shape_count +=1

            print(traceback.format_exc())
    
    print("Training set mask accuracy {}".format(np.nanmean(t_acc)))
    print("Training set mask recall {}".format(np.nanmean(t_recall)))
    print("Training set mask accuracy generous {}".format(np.nanmean(t_acc_gen)))
    print("Training set mask prec {}".format(np.nanmean(t_prec)))
    print("Training set mask prec generous {}".format(np.nanmean(t_prec_gen)))

    print("Training set mask recall 45 {}".format(np.nanmean(t_45_recall)))
    print("Training set mask recall 3 {}".format(np.nanmean(t_3_recall)))
    print("Training set mask recall 2 {}".format(np.nanmean(t_2_recall)))
    print("Training set mask recall 1 {}".format(np.nanmean(t_1_recall)))

def center_img(img):
    # Convert to grayscale
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = copy.deepcopy(img) * 255
    out = np.zeros_like(gray, dtype=np.uint32)


    # Apply thresholding
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate center of each contour
    centers = []
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
            color = random.randint(0, 255)
            gray = cv2.drawContours(gray, contours, i, color=30, thickness=-1)
            out[gray == 30] = i+1
            gray[gray == 30] = 1
            assert not np.count_nonzero(gray == 30)
    return centers, out

def entity_recall(gt_contour, pred):
    contours = np.unique(gt_contour)
    contours = contours[contours != 0]
    num_right = 0
    for i , cnt in enumerate(contours):
        if cnt == 0: continue
        num = np.count_nonzero(gt_contour == cnt)
        num_intersect = np.count_nonzero(pred[gt_contour == cnt] == 255)
        if num_intersect/num >= 0.5: 
            num_right +=1
    return num_right/(len(contours)-1)

def recall_generous(gt, pred):
    pred_expanded = np.zeros_like(pred)
    conv_kernel = np.ones((5, 5), np.uint8)
    pred_expanded = cv2.dilate(pred, conv_kernel, iterations=1)
    plt.imshow(pred_expanded)
    plt.show()

    tp = np.sum((gt == 255) & (pred_expanded == 255))
    fn = np.sum((gt == 255) & (pred_expanded == 0))
    return tp/(tp+fn)


def filter_islands_and_inside(pred, mask_membrane):
    #dilate the predictions a bit
    pred = pred * (mask_membrane == 0)
    pred = pred.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    expanded_pred = cv2.dilate(pred, kernel, iterations=2)
    to_ret = copy.deepcopy(expanded_pred)

    # detec the entities:
    centers, painted_preds = center_img(expanded_pred)
    # remove_labels = np.unique(painted_preds[(expanded_pred + mask_membrane) == 2])
    # for label in remove_labels:
    #     expanded_pred[painted_preds == label] = 0

    for _, center in enumerate(centers):
        x, y = center
        gj_id = painted_preds[y, x]

        temp_pred = np.zeros_like(pred)
        temp_pred[painted_preds == gj_id] = 1
        #dilate 
        kernel = np.ones((5, 5), np.uint8)
        temp_pred = cv2.dilate(temp_pred, kernel, iterations=1)

        #take interseection w cellids
        connection_ids = np.unique(mask_membrane[temp_pred == 1])
        connection_ids = sorted(connection_ids)[1:]
        if len(connection_ids) > 10: continue
        if len(connection_ids) != 2: 
            #wipe it off:
            # print(connection_ids)
            # y = input("viz?")
            # if y == "y":
            #     plt.imshow(expanded_pred*0.5 + temp_pred * 0.5, cmap="gray")
            #     plt.show()
            # if len(connection_ids) == 0:
            #     plt.imshow(expanded_pred*0.5 + temp_pred * 0.5, cmap="gray")
            #     plt.show()
            expanded_pred[temp_pred == 1] = 0
    
    return expanded_pred, to_ret

def test_entity_recall():
    base = "/Volumes/Normal/gapjnc/resuklts/assembled_2d_ent_recall_test/"
    # img = cv2.imread(base+"SEM_dauer_2_image_export_s101_Y16_X18_gt.png")

    files = glob.glob(base+"SEM_dauer_2_image_export_*pred.png")
    rec = []
    # for file in tqdm(files):
    #     gt = cv2.imread(file.replace("pred", "gt"))
    #     _, gt_cont = center_img(gt)
    #     pred = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
    #     pred[pred != 0] = 255
    #     recall = entity_recall(gt_cont, pred)
    #     print(recall)
    #     rec.append(recall)
    # print(np.nanmean(rec))

    #recall generous:
    for file in tqdm(files):
        gt = cv2.cvtColor(cv2.imread(file.replace("pred", "gt")), cv2.COLOR_BGR2GRAY)
        pred = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        pred[pred != 0] = 255

        recall = recall_generous(gt, pred)
        print(recall)
        rec.append(recall)
    print(np.nanmean(rec))
    #TODO: integrate this with the pipeline and remove this function

def create_connection_matrix(pred_dir, cell_id_dir, nr_mask_dir, pred_id_template, cell_id_template, nr_mask_template, section_wise=False):
    id_txt_file = open("VAST_segmentation_metadata.txt", "r")
    id_txt = id_txt_file.readlines()[5:]
    id_txt_file.close()
    id_to_name = {}
    for line in id_txt:
        line = line.split("\t")
        id_to_name[int(line[0])] = line[-1]



    pred_files = sorted(glob.glob(pred_dir + pred_id_template))

    linkages = {}
    if section_wise: graphs = []

    for pred_file in tqdm(pred_files):
        if section_wise:
            linkages = {}
        s = int(re.findall(r's\d+', pred_file)[0][1:])
        pred = cv2.cvtColor(cv2.imread(pred_file, -1), cv2.COLOR_BGR2GRAY) != 0
        nr_mask = cv2.imread(os.path.join(nr_mask_dir, pred_file.replace(pred_id_template, nr_mask_template).split("/")[-1]), -1) == 5497
        cell_id_mask = cv2.imread(os.path.join(cell_id_dir, pred_file.replace(pred_id_template, cell_id_template).split("/")[-1]), -1)

        pred *= nr_mask
        #get the entities:
        centers, painted_preds = center_img(pred)
        for _, center in enumerate(centers):
            x, y = center
            gj_id = painted_preds[y, x]

            temp_pred = np.zeros_like(pred)
            temp_pred[painted_preds == gj_id] = 1
            #dilate 
            kernel = np.ones((5, 5), np.uint8)
            temp_pred = cv2.dilate(temp_pred, kernel, iterations=1)

            #take interseection w cellids
            connection_ids = np.unique(cell_id_mask[temp_pred == 1])
            connection_ids = sorted(connection_ids)[1:]
            if len(connection_ids) != 2: 
                continue
            #make list
            connection_ids = tuple(connection_ids)
            connection_ids = tuple([id_to_name[i] for i in connection_ids])

            if sorted(connection_ids) not in linkages:
                linkages[sorted(connection_ids)] = [s]
            else:
                linkages[sorted(connection_ids)].append(s)
        if section_wise:
            graphs.append(nx.write_graphml(nx.to_undirected(nx.from_edgelist(list(linkages.keys())))))

    if not section_wise:
        #give out one big connection graph
        G = nx.from_edgelist(list(linkages.keys()))
        G = nx.to_undirected(G)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
        plt.draw()
    else:
        return graphs

if __name__ == "__main__":

    #TEST FILTERING CODE:
    gt = cv2.cvtColor(cv2.imread("C:/finetuned_2d_unet_aug_dauer_2_full_front_assembledSEM_dauer_2_image_export_s004_Y16_X18_gt.png", -1), cv2.COLOR_BGR2GRAY)
    img = cv2.imread("C:/finetuned_2d_unet_aug_dauer_2_full_front_assembledSEM_dauer_2_image_export_s004_Y16_X18_img.png")
    pred = cv2.cvtColor(cv2.imread("C:/finetuned_2d_unet_aug_dauer_2_full_front_assembledSEM_dauer_2_image_export_s004_Y16_X18_pred.png"), cv2.COLOR_BGR2GRAY) != 0
    nr_mask = cv2.imread("E:/Mishaal/nr_in_out/nr_in_out_s111.png", -1) == 5497
    pred *= nr_mask
    mask_membrane = cv2.imread("E:/Mishaal/sem_dauer_2_cell_ids/20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_s004.png", -1)

    newpred, to_pret = filter_islands_and_inside(pred, mask_membrane)

    #compute the recall and precision on newpred and to_pret compared to gt
    kernel = np.ones((5, 5), np.uint8)

    print("Recall original: ", mask_recall(gt, pred*255, mask=nr_mask))
    print("Precision original: ", mask_precision_generous(gt, pred*255, mask=nr_mask))

    print("Recall pre-filtering: ", mask_recall(gt, to_pret*255, mask=nr_mask))
    print("Precision pre-filtering: ", mask_precision_generous(gt, cv2.erode(to_pret*255, kernel, iterations=1), mask=nr_mask))

    print("Recall post-filtering: ", mask_recall(gt, newpred*255, mask=nr_mask))
    print("Precision post-filtering: ", mask_precision_generous(gt,cv2.erode(newpred*255, kernel, iterations=1), mask=nr_mask))


    # newpred = np.stack([newpred, np.zeros_like(newpred), np.zeros_like(newpred)], axis=-1) *255
    #overlay img on pred
    # im = cv2.addWeighted(img, 0.5, newpred, 0.6, 0)
    # cv2.imwrite("overlap_fikltered.png", im)

    # pred = to_pret.astype(np.uint8) * 255
    # pred = np.stack([pred, np.zeros_like(pred), np.zeros_like(pred)], axis=-1) 
    # im1 = cv2.addWeighted(img, 0.5, pred, 0.6, 0)
    # cv2.imwrite("overlap_prefilter.png", im1)
    

    #test overlay:
    # img1 = "/Volumes/Normal/gapjnc/resuklts/assembled_finetuned_aug_2d_sem_dauer_1/SEM_dauer_2_image_export_s113_Y16_X18_pred.png"
    # img2 = "/Volumes/Normal/gapjnc/resuklts/assembled_finetuned_aug_2d_sem_dauer_1/SEM_dauer_2_image_export_s113_Y16_X18_img.png"

    # #overlap img1 on img2
    # img1 = cv2.imread(img1)
    # img2 = cv2.imread(img2)
    # img1 = cv2.addWeighted(img2, 0.5, img1, 0.6, 0)
    # cv2.imwrite("overlap1.png", img1)

