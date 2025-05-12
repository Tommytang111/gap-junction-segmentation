import cv2, re, os, sys, shutil, numpy as np
from tqdm import tqdm
from postprocessing import center_img, move_generous


"""
Split a full EM image into tiles of size @tile_size, with an optional offset from the beginning of an image of @offset pixels
@param img: the image to split
@param offset: the offset from the beginning of the image
@param tile_size: the size of the tiles
@param names: whether to return the names of the tiles
@return: a list of the tiles, and optionally the names of the tiles
"""
def split_img(img, offset=256, tile_size=512, names=False):
    if offset:
        img = img[offset:-offset, offset:-offset]
    imgs = []
    names_list = []
    for i in range(img.shape[0]//tile_size+1):
        for j in range(img.shape[1]//tile_size+1):
            imgs.append(img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size])
            names_list.append("Y{}_X{}".format(i, j))
    return (imgs, names_list) if names else imgs 

"""
Given an image at a certain cross_section, get corresponding area at a different cross-section at offset i from current depth
@param filename: the filename of the image
@param i: the offset from the current depth: <0 for before, >0 for after
@param pattern: the pattern to match the depth
@return: the filename of the image at the new depth
"""
def get_another(filename, i, pattern=r's\d\d\d'):
        i = int(re.findall(pattern, filename)[0][1:]) + i
        assert i>=0
        return filename.replace(re.findall(pattern, filename)[0], "s"+("0" if i <=9 else 
    "") + str(i))

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
def create_dataset_3d(flat_dataset_dir, output_dir, window=(0, 1, 0, 0), image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, depth_pattern=r's\d\d\d', test=False):
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
    
"""
Function to create a 2d dataset from a dataset of full EM images
@param imgs_dir: the directory of the full EM images
@param output_dir: the directory to output the 2d dataset
@param seg_dir: the directory of the segmentations
@param img_size: the size of the images
@param image_to_seg_name_map: the mapping from image to segmentation name
@param add_dir: the additional directories to include in the dataset
@param add_dir_maps: the mapping from image to additional data directory
@param create_overlap: whether to create overlapping tiles - an offset of img_size//2 is used -- helps with predictions since every GJ is given a centered context
@param seg_ignore: the values to ignore in the segmentation
@param test: whether to run in test mode - just an images dataset
@return: None (creates and saves your dataset to specified directory)
"""
def create_dataset_2d_from_full(imgs_dir, output_dir, seg_dir=None, img_size=512, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, create_overlap=False, seg_ignore=(2, 15), test=False):

    assert (add_dir is None and add_dir_maps is None or add_dir is not None and add_dir_maps is not None), "Missing additional directory name mapping for additional data directories, or vice versa"
    if not test and image_to_seg_name_map is None:
        print("WARNING: No image to segmentation name mapping provided, assuming the default naming convention")
        image_to_seg_name_map = lambda x: x.replace('img', 'seg')

    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)

    os.makedirs(output_dir)
    #make subdirs
    os.makedirs(os.path.join(output_dir, "imgs"))
    if not test:
        os.makedirs(os.path.join(output_dir, "gts"))
        if add_dir:
            for i in (add_dir):
                os.makedirs(os.path.join(output_dir, os.path.split(i)[-1]))

    imgs = sorted(os.listdir(imgs_dir))

    def split_subroutine(img, gt, offset=0):

            img_imgs, img_names = split_img(img, offset=offset, tile_size=img_size, names=True)
            if not test: 
                gt_imgs = split_img(gt, offset=offset, tile_size=img_size)
                add_data = [] if add_dir else None
                if add_dir:
                    for j in range(len(add_dir)):
                        dat = cv2.imread(os.path.join(add_dir[j], add_dir_maps[add_dir[j]](imgs[i])))
                        dat = split_img(dat, offset=offset, tile_size=img_size)
                        add_data.append(dat)

            for j in range(len(img_imgs)):
                cv2.imwrite(os.path.join(output_dir, f"imgs/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), img_imgs[j])
                if not test:
                    cv2.imwrite(os.path.join(output_dir, f"gts/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), gt_imgs[j])
                    if add_dir:
                        for k in range(len(add_data)):
                            assert cv2.imwrite(os.path.join(output_dir, f"{os.path.split(add_dir[k])[-1]}/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), add_data[k][j])

    for i in tqdm(range(len(imgs))):
        if "DS" in imgs[i]: continue
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]), -1)
        if not test: 
            gt = cv2.imread(os.path.join(seg_dir, image_to_seg_name_map(imgs[i])), -1)

            #make changes in the gt
            for ig in seg_ignore:
                gt[gt == ig] = 0

            gt[gt!=0] = 255
        else: gt = None

        split_subroutine(img, gt)
        
        # make the test overlapping mode 
        if create_overlap:
            split_subroutine(img, gt, offset=img_size//2)
    
"""
Function to create a train-val split of a dataset. Given the highly-shared structure of C. elegans, we split the dataset using defined chunks to reduce
the amount of structure shared between the train and validation sets.

@param dataset_dir: the directory of the dataset
@param output_dir: the directory to output the split
@param td: whether to copy the directories as trees
@param filter_neurons: whether to filter out images without neurons
@param filter_gj: whether to filter out images without GJs
@param image_to_seg_name_map: the mapping from image to segmentation name
@param add_dir: the additional directories to include in the dataset
@param add_dir_maps: the mapping from image to additional data directory
@return: None (creates and saves your dataset to specified directory)
"""
def create_train_val_split(dataset_dir, output_dir, td=False, filter_neurons=False, filter_gj=False, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None): 
    
    # TODO: how to specify the splitting of the worm?
    s_set = set()
    x_set = set()
    y_set = set()

    imgs = sorted(os.listdir(os.path.join(dataset_dir, "imgs/")))

    extras = add_dir
    assert len(extras) == len(add_dir_maps)

    train_imgs, train_segs, train_adds = [], [], {}
    valid_imgs, valid_segs, valid_adds = [], [], {}
    for img in imgs:
        s, y, x = re.findall("s\d\d\d", img), re.findall("Y\d+_", img), re.findall("X\d+", img)
        #open the seg and check if any gjs:
        
        if filter_gj:
            seg = cv2.imread(image_to_seg_name_map(img), cv2.COLOR_BGR2GRAY)
            if not np.any(seg != 0): continue
        
        if filter_neurons:
            key = [k for k in extras if "neur" in k][0]
            neur = cv2.imread(add_dir_maps[key](img), cv2.COLOR_BGR2GRAY)
            if not np.any(neur != 0): continue
        
        if s in s_set and y in y_set and x in x_set:
            #validation mode
            valid_imgs += [img]
            valid_segs += [image_to_seg_name_map(img)]
        else:
            train_imgs +=[img]
            train_segs+=[image_to_seg_name_map(img)]
    for j in extras:
        train_extra_dirs, valid_extra_dirs = []
        for img in imgs:
            s, y, x = re.findall("s\d\d\d", img), re.findall("Y\d+_", img), re.findall("X\d+", img)
            if s in s_set and y in y_set and x in x_set:
                #validation mode
                valid_extra_dirs += [add_dir_maps[j](img)]
            else:
                train_extra_dirs+=[add_dir_maps[j](img)]
        train_adds[j] = (train_extra_dirs)
        valid_adds[j] = (valid_extra_dirs)
    
    #write stuff
    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)
    
    os.mkdir(os.path.join(output_dir, "train_imgs/"))
    os.mkdir(os.path.join(output_dir, "train_segs/"))
    os.mkdir(os.path.join(output_dir, "valid_imgs/"))
    os.mkdir(os.path.join(output_dir, "valid_segs/"))
    for j in extras:
        os.mkdir(os.path.join(output_dir, f"train_{j}/"))
        os.mkdir(os.path.join(output_dir, f"valid_{j}/"))

    copyfn = lambda src, dest: shutil.copy(src, dest) if not td else shutil.copytree(src, dest)
    
    for i in range(len(train_imgs)):
        copyfn(os.path.join(dataset_dir, "imgs", train_imgs[i]), os.path.join(output_dir, "imgs", train_imgs[i]))
        copyfn(os.path.join(dataset_dir, "segs", train_segs[i]), os.path.join(output_dir, "segs", train_imgs[i]))
        for j in extras:
            copyfn(os.path.join(dataset_dir, f"{j}", train_adds[j][i]), os.path.join(output_dir, f"{j}", train_adds[j][i]))
    for i in range(len(valid_imgs)):
        copyfn(os.path.join(dataset_dir, "imgs", valid_imgs[i]), os.path.join(output_dir, "imgs", valid_imgs[i]))
        copyfn(os.path.join(dataset_dir, "segs", valid_segs[i]), os.path.join(output_dir, "segs", valid_imgs[i]))
        for j in extras:
            copyfn(os.path.join(dataset_dir, f"{j}", valid_adds[j][i]), os.path.join(output_dir, f"{j}", valid_adds[j][i]))
    
def create_entity_sequence_dataset(imgs_dir, output_dir, seg_dir=None, img_size=512, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, seg_ignore=(2, 15)):
    assert (add_dir is None and add_dir_maps is None or add_dir is not None and add_dir_maps is not None), "Missing additional directory name mapping for additional data directories, or vice versa"
    if image_to_seg_name_map is None:
        print("WARNING: No image to segmentation name mapping provided, assuming the default naming convention")
        image_to_seg_name_map = lambda x: x.replace('img', 'seg')

    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)
    
    os.makedirs(output_dir)
    #make subdirs
    os.makedirs(os.path.join(output_dir, "gts_mod"))
    os.makedirs(os.path.join(output_dir, "gts_mod_centers"))
    gj_set = set()
    imgs = sorted(os.listdir(imgs_dir))
    
    prev_gt = None
    for i in tqdm(range(len(imgs))):
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]))
        gt = cv2.imread(os.path.join(seg_dir, image_to_seg_name_map(imgs[i])))
        
        #filter unwanted gts
        for ig in seg_ignore:
            gt[gt == ig] = 0
        # get the contours and centers:
        gt_contours, gt_centers = center_img(gt)
        
        if i != 0:
            #trace 
            new_gt_contours = np.zeros_like(gt_contours)
            new_gt_centers = np.zeros_like(gt_centers)
            for c in gj_set:
                temp_gt = gt_contours != 0
                temp_prev_gt = prev_gt == c
                
                #which center do they belong to?
                curr_no = np.unique(gt_contours[temp_gt == temp_prev_gt])[1:]
                for element in curr_no:
                    new_gt_contours[gt_contours == element] = c
                    new_gt_centers[gt_centers == element] = c

            
            #everything has been assigned, what is leftover?
            leftover = np.unique(gt_contours * (new_gt_contours == 0))
            leftover = leftover[leftover != 0]
            if len(leftover) >= 0:
                for element in leftover:
                    gj_set.add(len(gj_set) + 1)
                    new_gt_contours[gt_contours == element] = len(gj_set) + 1
                    new_gt_centers[gt_centers == element] = len(gj_set) + 1
            cv2.imwrite(os.path.join(output_dir, f"gts_mod/{imgs[i]}"), new_gt_contours)
            cv2.imwrite(os.path.join(output_dir, f"gts_mod_centers/{imgs[i]}"), new_gt_centers)

            prev_gt = new_gt_contours
        else:
            contours = np.unique(gt_contours)
            contours = contours[contours != 0]
            gj_set = set(contours)
            prev_gt = gt_contours
            cv2.imwrite(os.path.join(gt_contours, output_dir, f"gts_mod/{imgs[i]}"))
            cv2.imwrite(os.path.join(output_dir, f"gts_mod_centers/{imgs[i]}"), gt_centers)
    f = open("gj_set.txt", "w")
    f.write(str(gj_set))
    f.close()

    #split stuff lesgo 
    os.makedirs(os.path.join(output_dir, "imgs"))
    os.makedirs(os.path.join(output_dir, "gts"))
    if add_dir:
        for i in (add_dir):
            os.makedirs(os.path.join(output_dir, os.path.split(i)[-1]))
        
    for i in range(len(gj_set)):
        os.makedirs(os.path.join(output_dir, f"gts/{i}"))
        os.make_dirs(os.path.join(output_dir, f"imgs/{i}"))
        if add_dir:
            for j in (add_dir):
                os.makedirs(os.path.join(output_dir, os.path.split(j)[-1], str(i)))
    
    for i in tqdm(range(len(imgs))):
        gt_contours, gt_centers = cv2.imread(os.path.join(output_dir, f"gts_mod/{imgs[i]}"), cv2.COLOR_BGR2GRAY), cv2.imread(os.path.join(output_dir, f"gts_mod_centers/{imgs[i]}"), cv2.COLOR_BGR2GRAY)
        #contours
        contours = np.unique(gt_contours)
        contours = contours[contours != 0]
        for c in contours:
            #center location
            row, col = np.where(gt_centers == c)
            split_img = img[row[0]-img_size//2:row[0]+img_size//2, col[0]-img_size//2:col[0]+img_size//2]
            cv2.imwrite(os.path.join(output_dir, f"imgs/{c}/{imgs[i]}"), split_img)
            split_gt = gt_contours[row[0]-img_size//2:row[0]+img_size//2, col[0]-img_size//2:col[0]+img_size//2] != 0
            cv2.imwrite(os.path.join(output_dir, f"gts/{c}/{imgs[i]}"), split_gt)
            if add_dir:
                for j in (add_dir):
                    dat = cv2.imread(os.path.join(add_dir[j], add_dir_maps[j](imgs[i])))
                    split_dat = dat[row[0]-img_size//2:row[0]+img_size//2, col[0]-img_size//2:col[0]+img_size//2]
                    cv2.imwrite(os.path.join(output_dir, os.path.split(j)[-1], str(c), imgs[i]), split_dat)

def create_unsupervised_dataset(images_dir, cell_id_dir, nr_mask_dir, img_template, id_template=None, nr_mask_template=None, out_dir="unsupervised_dataset", img_size=512):
    os.makedirs(out_dir, exist_ok=True)
    if id_template is None: id_template = img_template
    if nr_mask_template is None: nr_mask_template = img_template

    images = os.listdir(images_dir)

    write_count = 0
    skipped = 0
    
    for img in tqdm(images):
        try:
            s = re.findall("s\d\d\d", img)[0][1:]
        except:
            print(img)
            continue
        image = cv2.imread(os.path.join(images_dir, img), -1)
        neurons = cv2.imread(os.path.join(cell_id_dir, img.replace(img_template, id_template)), -1)
        nr_mask = cv2.imread(os.path.join(nr_mask_dir, img.replace(img_template, nr_mask_template)), -1)

        nr_mask[nr_mask != 5497] = 0
        nr_mask[nr_mask == 5497] = 1

        neurons *= nr_mask

        cell_ids = np.unique(neurons)

        for cell_id in cell_ids:
            if cell_id == 0: continue
            i = 0
            cell = neurons == cell_id
            cell = cell.astype(np.uint8) * 255
            contours, _ = cv2.findContours(cell, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            M = cv2.moments(contours[i])
            while M["m00"] == 0:
                if len(contours) == i+1: break
                M = cv2.moments(contours[i+1])
                i += 1
            if M["m00"] == 0: 
                print(f"Skipping {img} for cell {cell_id}")
                skipped += 1
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cell_img = image[cY-img_size//2:cY+img_size//2, cX-img_size//2:cX+img_size//2]
            
            os.makedirs(os.path.join(out_dir, f"neuron_{cell_id}"), exist_ok=True)
            assert not os.path.isfile(os.path.join(out_dir, f"neuron_{cell_id}", f"s{(3-len(str(s)))*str(0)+str(s)}.png")), f"""File already exists: {os.path.join(out_dir, f'neuron_{cell_id}', f"s{(3-len(str(s)))*'0'+str(s)}")}"""
            cv2.imwrite(os.path.join(out_dir, f"neuron_{cell_id}", f"s{(3-len(str(s)))*'0'+str(s)}.png"), cell_img)
            write_count += 1
    print(f"Written {write_count} images to {out_dir}")

def create_nerve_ring_split(img_dir, nr_dir, output_dir, img_template, nr_template, img_size=512, offset=0):
    # get the images
    imgs = os.listdir(img_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in tqdm(imgs):
        if "DS" in img_name: continue
        img = cv2.imread(os.path.join(img_dir, img_name), -1)
        nr = cv2.imread(os.path.join(nr_dir, img_name.replace(img_template, nr_template)), -1)
        nr[nr != 5497] = 0
        
        #split image into img_size
        img_imgs, img_names = split_img(img, tile_size=img_size, names=True)

        for i in range(len(img_imgs)):
            if img_imgs[i].shape[0] != img_size or img_imgs[i].shape[1] != img_size or np.count_nonzero(img_imgs[i] != 0) < 0.5*img_size**2: continue
            cv2.imwrite(os.path.join(output_dir, f"imgs/{img_names[i]}.png"), img_imgs[i])
        if offset:
            img_imgs, img_names = split_img(img, offset=offset, tile_size=img_size, names=True)
            for i in range(len(img_imgs)):
                if img_imgs[i].shape[0] != img_size or img_imgs[i].shape[1] != img_size or np.count_nonzero(img_imgs[i] != 0) < 0.5*img_size**2: continue
                cv2.imwrite(os.path.join(output_dir, f"imgs/{img_names[i]}off.png"), img_imgs[i])
    


def create_fn_dataset(img_dir, gt_dir, preds_dir , img_template, gt_template, preds_template, output_dir, npy=False):
    imgs = os.listdir(img_dir)
    os.makedirs(output_dir, exist_ok=True)

    for img in imgs:
        gt = cv2.imread(os.path.join(gt_dir, img.replace(img_template, gt_template)), cv2.COLOR_BGR2GRAY)
        if not npy: pred = cv2.imread(os.path.join(preds_dir, img.replace(img_template, preds_template)), cv2.COLOR_BGR2GRAY)
        else: 
            pred = np.load(os.path.join(preds_dir, img.replace(img_template, preds_template) + ".npy"))
            #sigmoid it
            pred = 1/(1+np.exp(-pred)) >= 0.5

        # in the gt, 0 and 1. 1 is gj. mark False Negatives as 2
        gt[gt != 0] =1 
        pred[pred != 0] = 1
        gt[(gt == 1) & (pred == 0)] = 2

        cv2.imwrite(os.path.join(output_dir, img), gt)

if __name__ == "__main__":
    # create_entity_sequence_dataset(imgs_dir="/Volumes/Normal/gapjnc/sections100_125/em/", seg_dir="/Volumes/Normal/gapjnc/sections100_125/gj_seg/", 
    #                                output_dir="/Volumes/Normal/gapjnc/sections100_125/sequence_dataset/", img_size=512, image_to_seg_name_map=lambda x: x.replace('sem_dauer_2_em_', 'sem_dauer_2_gj_gt_'))
    if False:
        split = "train"
        dirs = "/Volumes/Normal/gapjnc/final_jnc_only_split_extend/"
        masks = sorted(os.listdir(os.path.join(dirs, split+"_gts/")))
        imgs = sorted(os.listdir(os.path.join(dirs, split+"_imgs/")))
        preds = sorted(os.listdir(os.path.join(dirs, split+"_preds/")))

        os.makedirs("/Volumes/Normal/gapjnc/0_50_extend_train/",exist_ok=True)
        os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_imgs/",exist_ok=True)
        os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/",exist_ok=True)

        mask_pairs_before, mask_pairs_after = [], []
        img_pairs_before, img_pairs_after = [], []

        for i in tqdm(range(len(masks))):
            s,x, y = re.findall("s\d\d\d", masks[i]), re.findall("Y\d+", masks[i]), re.findall("X\d+", masks[i])
            before = int(s[0][1:]) - 1
            after = int(s[0][1:]) + 1
            # if before < 0 or after > 50: continue
            before = "s"+("0" * (3-len(str(before)))) + str(before)
            after = "s"+("0" * (3-len(str(after)))) + str(after)
            before_mask = preds[i].replace(s[0], before)
            after_mask = preds[i].replace(s[0], after)

            before_gt = cv2.imread(os.path.join(dirs, f"{split}_gts", masks[i]))
            if len(np.unique(before_gt)) == 1: continue # dont need this guy
            #check if they exist
            if before_mask.replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_") in masks:
                mask_pairs_before.append((before_mask, masks[i]))
                img_pairs_before.append((imgs[i].replace(s[0], before), imgs[i]))
            if after_mask.replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_") in masks:
                mask_pairs_after.append((after_mask, masks[i]))
                img_pairs_after.append((imgs[i].replace(s[0], after), imgs[i]))
            
        def subhelper(mask_list, image_list, section="before"):
            for i in tqdm(range(len(mask_list))):
                base_name = mask_list[i][1]
                if ".png.png" in base_name: off = -8
                else: off = -4
                target_name = mask_list[i][1].replace("sem2dauer_gj_2d_training.vsseg_export_", "SEM_dauer_2_image_export_")
                os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{target_name[:off]}_{section}",exist_ok=True)
                os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_imgs/{target_name[:off]}_{section}",exist_ok=True)
                shutil.copy(os.path.join(dirs, f"{split}_preds", mask_list[i][0]), f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{target_name[:off]}_{section}/{mask_list[i][0]}")
                shutil.copy(os.path.join(dirs, f"{split}_imgs", image_list[i][0]), f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_imgs/{target_name[:off]}_{section}/{image_list[i][0]}")
                shutil.copy(os.path.join(dirs, f"{split}_imgs", image_list[i][1]), f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_imgs/{target_name[:off]}_{section}/{image_list[i][1]}")

                # the new label shud only be the intersecting gap junctions
                other = cv2.imread(os.path.join(dirs, f"{split}_preds", mask_list[i][0]))
                labels = cv2.imread(os.path.join(dirs, f"{split}_gts", mask_list[i][1]))

                labels[(labels == 2) | (labels == 15)] = 0
                labels[(labels == 7) | (labels == 9) | (labels == 11)] = 255

                #get the centers and contours of each one 
                other_centers, other_contours = center_img(other)
                labels_centers, labels_contours = center_img(labels)

                new_labels = np.zeros_like(labels)
                for c in range(len(other_centers)):

                    present_mask = other_contours == c+1
                    #roll it 
                    present_mask = move_generous(present_mask, 0, 25)
                    label_contours_temp = move_generous(labels_contours, 0, 25)
                    present_contours = np.unique(label_contours_temp[present_mask])

                    for c_ in present_contours:
                        if c_ == 0: continue
                        new_labels[labels_contours == c_] = 255

                assert cv2.imwrite(f'/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{target_name[:off]}_{section}/{target_name}', new_labels)
                # print(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{target_name[:-8]}/{target_name[:-8]+'og.png'}")
                assert cv2.imwrite(f"/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{target_name[:off]}_{section}/{target_name[:off]+'og.png'}", move_generous(cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY), 0, 25) | move_generous(cv2.cvtColor(other, cv2.COLOR_BGR2GRAY), 0, 25))
        subhelper(mask_pairs_after, img_pairs_after, "after")
        subhelper(mask_pairs_before, img_pairs_before, "before")
    
    if False:
        # time to balance
        import random
        split="train"
        dirs = os.listdir(f'/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/')
        empty_gts = []
        others = []
        for d in tqdm(dirs):
            if "DS" in d: continue
            files = os.listdir(os.path.join(f'/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/', d))
            temp_d = d.replace("_after", "").replace("_before", "")
            try:
                files.remove(temp_d+".png.png")
                right = temp_d+".png.png"
            except:
                files.remove(temp_d+".png")
                right =temp_d+".png"
            files.remove(temp_d+"og.png")
            # assert len(files) == 1, files
            # file = files[0]
            pred = cv2.cvtColor(cv2.imread(os.path.join(f'/Volumes/Normal/gapjnc/0_50_extend_train/{split}_gts/{d}/', right)), cv2.COLOR_BGR2GRAY)
            if len(np.unique(pred)) == 1: 
                empty_gts.append(d)
                continue
            others.append(d)
        total = int(len(others) * 10/8)
        print(total)
        others += random.sample(empty_gts, total - len(others))

        os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/train_imgs_balanced/")
        os.makedirs(f"/Volumes/Normal/gapjnc/0_50_extend_train/train_gts_balanced/")
        for f in others:
            shutil.copytree(f"/Volumes/Normal/gapjnc/0_50_extend_train/train_imgs/{f}", f"/Volumes/Normal/gapjnc/0_50_extend_train/train_imgs_balanced/{f}")
            shutil.copytree(f"/Volumes/Normal/gapjnc/0_50_extend_train/train_gts/{f}", f"/Volumes/Normal/gapjnc/0_50_extend_train/train_gts_balanced/{f}")

    if True:
        images_dir = "/Volumes/Normal/gapjnc/sem_dauer_2_full_front"
        cell_id_dir = "/Volumes/Normal/gapjnc/sem_dauer_2_cell_ids"
        nr_mask_dir = "/Volumes/Normal/gapjnc/nr_in_out"

        img_template = "SEM_dauer_2_export_"
        nr_mask_template = "nr_in_out_"
        id_template="20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_"
        images = os.listdir(images_dir)

        unique_set = set()
        
        for img in tqdm(images):
            try:
                s = re.findall("s\d\d\d", img)[0][1:]
            except:
                print(img)
                continue
            image = cv2.imread(os.path.join(images_dir, img), -1)
            neurons = cv2.imread(os.path.join(cell_id_dir, img.replace(img_template, id_template)), -1)
            nr_mask = cv2.imread(os.path.join(nr_mask_dir, img.replace(img_template, nr_mask_template)), -1)

            nr_mask[nr_mask != 5497] = 0
            nr_mask[nr_mask == 5497] = 1

            neurons *= nr_mask

            cell_ids = np.unique(neurons)

            unique_set = unique_set.union(set(cell_ids.tolist()))

        print(len(unique_set))



        # create_fn_dataset("/Volumes/Normal/gapjnc/final_jnc_only_split/valid_imgs", "/Volumes/Normal/gapjnc/final_jnc_only_split/valid_gts", "/Volumes/Normal/gapjnc/final_jnc_only_split/valid_preds", "SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_", "SEM_dauer_2_image_export_", "/Volumes/Normal/gapjnc/final_jnc_only_split_rwt_valid/", npy=True)




    