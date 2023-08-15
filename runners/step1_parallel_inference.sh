#!/bin/bash

# make sure to do `accelerate config default` before running this to suppress warnings
accelerate launch --num_processes=8 ../data_scripts/step1_generate_instruct_parallel.py