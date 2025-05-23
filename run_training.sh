#!/bin/bash

python3 unet.py \
--dataset new \
--batch_size 8 \
--focal \
--epochs 100 \
--aug \
--model_name ../models/2d_gd_mem_run1_model5_epoch149.pk1