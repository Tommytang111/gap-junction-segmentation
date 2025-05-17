#!/bin/bash

python3 main_processing.py \
  --postprocessing \
  --imgs_dir "/mnt/e/Mishaal/sem_dauer_2/jnc_only_dataset_test/imgs" \
  --preds_dir "/home/tommytang111/results" \
  --output_dir "/home/tommytang111/assembled_results" \
  --missing_dir "/mnt/e/Mishaal/sem_dauer_2/image_export" \
  --img_template SEM_dauer_2_image_export_ \
  --seg_template SEM_dauer_2_image_export_ \
  --Smin 0 --Smax 51 \
  --Ymin 4 --Ymax 16 \
  --Xmin 5 --Xmax 10 \
  --offset 0
