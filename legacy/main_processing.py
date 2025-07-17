from preprocessing import *
from postprocessing import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing", action="store_true", help="For creation of datasets, train-val splits, with special options")
    parser.add_argument("--postprocessing", action="store_true", help="For assembling the tiled predictions into a full EM image")
    parser.add_argument("--results", action="store_true", help="performance metrics")

    parser.add_argument("--imgs_dir", default=None, type=str, help="Full path to directory where images are stored")# -- Shared
    parser.add_argument("--seg_dir", default="", type=str, help="Full path to where segmentation masks are stored")# -- Shared
    parser.add_argument("--add_dir", action="append", type=str, help="Full path to any additional training data")# -- Shared
    parser.add_argument("--add_dir_templates", action="append", type=str, help="Naming template for additional directories. e.g --add_dir_templates neuro_1 mito_pred")# -- Shared
    parser.add_argument("--seg_template", default="sem_dauer_2_gj_gt_", type=str, help="Naming template for segmentation masks") # -- Shared
    parser.add_argument("--img_template", default="SEM_dauer_2_em_", type=str, help="Naming template for EM images") # -- Shared
    parser.add_argument("--nr_mask_dir", default=None, type=str, help="Directory where relevancy area masks are stored")
    parser.add_argument("--nr_mask_template", default="nr_mask", type=str, help="Naming template for the relevancy area masks")

    # -- Pre-processing Flags --
    #extra unsupervised flags:
    parser.add_argument("--unsupervised_dataset", action="store_true", help="Unsupervised dataset based on EM cell IDs")
    parser.add_argument("--cell_id_template", default="cell_id", type=str, help="Naming template for the cell id masks")
    parser.add_argument("--cell_id_dir", default=None, type=str, help="Directory where the cell id masks are stored")

    parser.add_argument("--seg_ignore", action="append", type=int, default=[2, 15], help="Any indices in mask that should be ignored when training. These will be set to 0, rest to 255")
    parser.add_argument("--depth_pattern", type=str, default="sXXX", help="numbers where the depth is located should be contiguously annotated with an X. e.g sXXX")
    parser.add_argument("--window", action="append", type=int, default=[0, 1, 0, 0], help="window size for 3d dataset creation. Specify as tuple of one 1. 0 for context, 1 for target image. e.g. --window 0 1 0 0")

    parser.add_argument("--make_threeD", action="store_true", help="Create a 3d dataset from a flat dataset")
    parser.add_argument("--make_twoD", action="store_true", help="Create a 2d dataset from a full EM dataset")
    parser.add_argument("--flat_dataset_dir", default=None, type=str, help="Full path to the flat dataset directory")
    parser.add_argument("--output_dir", default=None, type=str, help="Full path to the output directory")
    parser.add_argument("--img_size", default=512, type=int, help="Size of the output image")
    parser.add_argument("--create_overlap", action="store_true", help="Create overlapping images for testing")
    parser.add_argument("--test", action="store_true", help="run in test mode, i.e, only care about generating datasets with imgs in them") # -- Shared

    parser.add_argument("--train_val_split", action="store_true", help="Split a dataset into training and validation")
    parser.add_argument("--train_dataset_dir", default=None, type=str, help="Full path to the source dataset directory")
    parser.add_argument("--filter_neurons", action="store_true", help="Ignore images without neurons")
    parser.add_argument("--filter_gj", action="store_true", help="Ignore images without GJs")
    parser.add_argument("--td", action="store_true", help="The dataset for postprocessing is a 3d dataset")

    # -- Post-processing Flags --

    parser.add_argument("--missing_dir", default=None, type=str, help="Directory where missing image data is stored. Generally used when GJs or neuron-wise filtering of dataset was done")
    parser.add_argument("--preds_dir", default=None, type=str, help="Full path to the predictions directory")
    parser.add_argument("--extend_dir", default=None, type=str, help="Full path to the directory where the extended predictions are stored")
    parser.add_argument("--filter_gjs", action="store_true", help="Filter out GJs from the predictions using cell ids")

    parser.add_argument("--Smin", type=int, default=101, help="starting S index for the image")
    parser.add_argument("--Smax", type=int, default=109, help="Ending S index for the image")
    parser.add_argument("--Xmin", default=0, type=int, help="Starting X index for the image")
    parser.add_argument("--Ymin", default=0, type=int, help="Starting Y index for the image")
    parser.add_argument("--Ymax", default=17, type=int, help="Ending Y index for the image") #nincl
    parser.add_argument("--Xmax", default=19, type=int, help="Ending X index for the image") #nincl
    parser.add_argument("--offset", default=256, type=int, help="generally half of the image size")
    parser.add_argument("--plot_legend", action="store_true", help="Print the legend for the assembled EM predictions")

    # -- Results Flags --
    parser.add_argument("--results_dir", default=None, type=str, help="Full path to the results directory")
    parser.add_argument("--no_assemble", action="store_true", help="Do not assemble the predictions before calculating the metrics")
    parser.add_argument("--breakdown", action="store_true", help="Breakdown the metrics by confidence type")

    args = parser.parse_args()

    #checking for argument consistency:
    if (args.preprocessing and args.postprocessing) or (args.preprocessing and args.results) or (args.postprocessing and args.results):
        raise ValueError("Cannot run post, pre, and/or results mode at the same time. Choose one!")
    if not (args.preprocessing or args.postprocessing or args.results):
        raise ValueError("Choose a mode to run: pre-processing, post-processing, or results")
    
    if args.preprocessing:
        if args.imgs_dir is None and args.train_val_split is None:
            raise ValueError("Please specify the image directory")
        if (args.make_twoD or args.make_threeD) and args.train_val_split:
            raise ValueError("Cannot make a 2d or 3d dataset and train val split at the same time. Run the 2d or 3d first, and then proceed with the split")
        if not args.test and args.seg_dir is None and args.make_threeD is None:
            raise ValueError("Please specify the segmentation directory or raise the test flag")
        if args.add_dir is not None and args.add_dir_templates is None:
            raise ValueError("Please specify the template for the additional directories")
        if args.add_dir is None and args.add_dir_templates is not None:
            raise ValueError("Please specify the additional directories")
        if args.make_threeD and args.flat_dataset_dir is None and args.make_twoD is None:
            raise ValueError("Please specify the flat dataset directory or opt in for making a 2d dataset too")
        if (args.make_twoD or args.make_threeD or args.train_val_split) and args.output_dir is None:
            raise ValueError("Please specify the output directory")
        if (args.train_val_split and args.train_dataset_dir is None):
            raise ValueError("Please specify the training dataset directory")
        if (args.train_val_split and args.filter_neurons and args.add_dir_templates is None):
            raise ValueError(f"Please specify the template for the additional subdirs present in {args.train_dataset_dir}, atleast for the neuron masks")
        if args.unsupervised_dataset:
            # assert args.cell_id_template is not None, "Please specify the cell id template"
            assert args.nr_mask_dir is not None, "Please specify the directory where the relevancy masks are stored"
            # assert args.cell_id_dir is not None, "Please specify the directory where the cell id masks are stored"
            assert args.nr_mask_template is not None, "Please specify the naming template for the relevancy masks"
        
        #report printing:
        print("Relevant Preprocessing Arguments:")
        print(f"Image Directory: {args.imgs_dir}")
        print(f"Image Template: {args.img_template}")
        if not args.test: 
            print(f"Segmentation Directory: {args.seg_dir}")
            print(f"Segmentation Template: {args.seg_template}")
        if args.add_dir is not None and not args.test: 
            print(f"Additional Directories: {args.add_dir}")
            print(f"Additional Directory Templates: {args.add_dir_templates}")
        print(f"Testing mode: {args.test}")
        if args.make_twoD:
            print(f"Output Directory for 2d Dataset: {args.output_dir}")
            print(f"Requested Output Image Size: {args.img_size}")
            print(f"Create Overlap: {args.create_overlap}")
            print(f"Segmentation Ignore: {args.seg_ignore}")
        if args.make_threeD:
            print(f"Flat Dataset Directory: {args.flat_dataset_dir}")
            print(f"Output Directory for 3d Dataset: {args.output_dir}")
            print(f"Depth Pattern: {args.depth_pattern}")
            print(f"Context Window: {args.window}")
        if args.train_val_split:
            print(f"Source Dataset Directory: {args.train_dataset_dir}")
            print(f"Output Directory for Training and Validation Split: {args.output_dir}")
            print(f"Filter Neurons: {args.filter_neurons}")
            print(f"Filter GJs: {args.filter_gj}")
        if args.unsupervised_dataset:
            print(f"Image Directory: {args.imgs_dir}")
            print(f"Image Template: {args.img_template}")
            print(f"Cell ID Directory: {args.cell_id_dir}")
            print(f"Cell ID Template: {args.cell_id_template}")
            print(f"Relevancy Mask Directory: {args.nr_mask_dir}")
            print(f"Relevancy Mask Template: {args.nr_mask_template}")
            print(f"Output Directory for Unsupervised Dataset: {args.output_dir}")
            print(f"Image Size for Unsupervised Dataset: {args.img_size}")
        print("NOTE: Unreported arguments were ignored.")
    
    if args.postprocessing:
        if args.imgs_dir is None:
            raise ValueError("Please specify the image directory")
        if args.seg_dir is None:
            raise ValueError("Please specify the segmentation directory")
        if args.preds_dir is None:
            raise ValueError("Please specify the predictions directory")
        if args.output_dir is None:
            raise ValueError("Please specify the output directory")
        
        print("Relevant Postprocessing Arguments:")
        print(f"Image Directory: {args.imgs_dir}")
        print(f"Segmentation Directory: {args.seg_dir}")
        print(f"Prediction Directory: {args.preds_dir}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Print legend: {args.plot_legend}")
        print(f"Missing Directory: {args.missing_dir}")
        print(f"Image Template: {args.img_template}")
        print(f"Segmentation Template: {args.seg_template}")
        print(f"Smin: {args.Smin} Smx: {args.Smax}")
        print(f"Xmin: {args.Xmin} Xmax: {args.Xmax}")
        print(f"Ymin: {args.Ymin} Ymax: {args.Ymax}")
        print(f"Offset: {args.offset}")
        print("Note: Unreported arguments were ignored.")
    
    if args.results:
        if args.preds_dir is None:
            raise ValueError("Please specify the predictions directory")
        if args.imgs_dir is None:
            raise ValueError("Please specify the image directory")
        if args.seg_dir is None:
            raise ValueError("Please specify the segmentation directory")
        
        print("Relevant Results Arguments:")
        print(f"Image Directory: {args.imgs_dir}")
        print(f"Segmentation Directory: {args.seg_dir}")
        print(f"Prediction Directory: {args.preds_dir}")
        print(f"Nr Mask Directory: {args.nr_mask_dir}")
        print(f"Nr Mask Template: {args.nr_mask_template}")
        print(f"Image Template: {args.img_template}")
        print(f"Segmentation Template: {args.seg_template}")
        print(f"Smin: {args.Smin} Smx: {args.Smax}")
        print(f"Xmin: {args.Xmin} Xmax: {args.Xmax}")
        print(f"Ymin: {args.Ymin} Ymax: {args.Ymax}")
        print(f"Offset: {args.offset}")
        print("Note: Unreported arguments were ignored.")
    
    # Uncomment the following lines to add a prompt for user confirmation before proceeding
    # cont = input("Do you want to continue? (y/n): ")
    # if cont.lower() != 'y':
    #     sys.exit(0)

    if args.preprocessing and args.unsupervised_dataset and args.cell_id_dir:
        create_unsupervised_dataset(args.imgs_dir, args.cell_id_dir, args.nr_mask_dir, args.img_template, args.cell_id_template, args.nr_mask_template, args.output_dir, args.img_size)
    if args.preprocessing and args.unsupervised_dataset:
        create_nerve_ring_split(args.imgs_dir, args.nr_mask_dir, args.output_dir, args.img_template, args.nr_mask_template, args.img_size, args.offset) 

    if args.preprocessing and args.make_twoD:
        f = None if args.test else lambda x: x.replace(args.img_template, args.seg_template)
        gs = None if args.add_dir is None else {i: lambda x: x.replace(args.img_template, args.add_dir_templates[j]) for j, i in enumerate(args.add_dir)}
        create_dataset_2d_from_full(args.imgs_dir, args.output_dir, seg_dir=args.seg_dir, img_size=args.img_size, image_to_seg_name_map= f, add_dir=args.add_dir, add_dir_maps=gs, seg_ignore=args.seg_ignore, create_overlap=args.create_overlap, test=args.test)
        args.flat_dataset_dir = args.output_dir
    
    if args.preprocessing and args.make_threeD:
        f = None if args.test else lambda x: x.replace(args.img_template, args.seg_template)
        gs = None if args.add_dir is None else {i: lambda x: x.replace(args.img_template, args.add_dir_templates[j]) for j, i in enumerate(args.add_dir)}
        output_dir = args.output_dir if not args.make_twoD else args.output_dir+"_3d"
        create_dataset_3d(args.flat_dataset_dir, output_dir, depth_pattern=r's\d\d\d', window=args.window, test=args.test, image_to_seg_name_map=f, add_dir_maps=gs, add_dir=args.add_dir)
    
    if args.preprocessing and args.train_val_split:
        f = None if args.test else lambda x: x.replace(args.img_template, args.seg_template)
        gs = None if args.add_dir is None else {i: lambda x: x.replace(args.img_template, args.add_dir_templates[i]) for i in args.add_dir}
        create_train_val_split(args.train_dataset_dir, args.output_dir, td=args.td, filter_neurons=args.filter_neurons, filter_gj=args.filter_gj, image_to_seg_name_map=f, add_dir_maps=gs, add_dir=args.add_dir)
    
    if args.postprocessing:
        f = None if args.test else lambda x: x.replace(args.img_template, args.seg_template)
        assemble_overlap(args.imgs_dir, args.seg_dir, args.preds_dir, args.output_dir, extend_dir=args.extend_dir,overlap=args.create_overlap, missing_dir=args.missing_dir, img_templ=args.img_template, seg_templ=args.seg_template, s_range=range(args.Smin, args.Smax), x_range=range(args.Xmin, args.Xmax), y_range=range(args.Ymin, args.Ymax), offset=args.offset, plot_legend=args.plot_legend, fn_mask_dir=args.nr_mask_dir, fn_filter=args.filter_gjs, fn_cell_dir=args.cell_id_dir)
    
    if args.results and not args.no_assemble:
        recalls, precisions, precisions_gen, accs, accs_gen = [], [], [], [], []


        stats = assemble_overlap(args.imgs_dir, args.seg_dir, args.preds_dir, args.output_dir, extend_dir=args.extend_dir, overlap=True, missing_dir=args.missing_dir, img_templ=args.img_template, seg_templ=args.seg_template, s_range=range(args.Smin, args.Smax), x_range=range(args.Xmin, args.Xmax), y_range=range(args.Ymin, args.Ymax), offset=args.offset, fn=assembled_stats, fn_mask_dir=args.nr_mask_dir, fn_filter=args.filter_gjs, fn_cell_dir=args.cell_id_dir)
        
        for s in stats:
            recalls.append(s[0])
            precisions.append(s[1])
            precisions_gen.append(s[2])
            accs.append(s[3])
            accs_gen.append(s[4])

        print("Recall: ", np.nanmean(recalls))
        print("Precision: ", np.nanmean(precisions))
        print("Precision Generous: ", np.nanmean(precisions_gen))
        print("Accuracy: ", np.nanmean(accs))
        print("Accuracy Generous: ", np.nanmean(accs_gen))
    elif args.results and args.no_assemble:
        preds_to_seg = lambda x: x.replace(args.image_template, args.seg_template)
        preds_to_mask = lambda x: x.replace(args.image_template, args.nr_mask_template)
        mask_acc_split(args.seg_dir, args.preds_dir, args.nr_mask_dir, args.td, args.breakdown, preds_to_seg, preds_to_mask)







    
