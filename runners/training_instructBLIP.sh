#!/bin/bash

# export TRANSFORMERS_VERBOSITY=info
export DS_SKIP_CUDA_CHECK=0

torchrun --nproc_per_node=8 --master_port=2345 ../data_scripts/train_instructBLIP.py \
    --config ../configs/training_args.yml


#     --evaluation_steps 3000 \
#     --lr_scheduler_num_cycles 0.5 \
#     --gradient_checkpointing \