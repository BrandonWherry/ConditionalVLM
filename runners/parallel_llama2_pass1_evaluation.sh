#!/bin/bash

# make sure to do `accelerate config default` before running this to suppress warnings
accelerate launch --num_processes=8 ../data_scripts/parallel_evaluation_pass1_llama2.py