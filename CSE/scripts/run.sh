#! /bin/bash

#cd ..
python re-NSFW_slic_fullgrad.py \
--attr_map=grad_cam \
--seg_map=slic \
--output_class=1 \
--img_dir=/workspace/adv_robustness/CSE/labelme/nsfw/test_images