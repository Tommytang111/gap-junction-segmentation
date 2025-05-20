#!/bin/bash

python3 main_processing.py \
  --postprocessing \
  --imgs_dir "/mnt/e/Mishaal/sem_dauer_2/image_export" \
  --preds_dir "/home/tommytang111/results" \
  --output_dir "/home/tommytang111/assembled_results" \
  --img_template SEM_dauer_2_image_export_ \
  --seg_template SEM_dauer_2_image_export_ \
  --Smin 0 --Smax 2 \
  --Ymin 0 --Ymax 17 \
  --Xmin 0 --Xmax 19 \
  --offset 0

#"/mnt/e/Mishaal/sem_dauer_2/jnc_only_dataset_test/imgs"
