"""Contains helper functions for running the BASS algorithm.

This module is based on having cloned & compiled https://github.com/BGU-CS-VIL/BASS
"""
import os
import subprocess
import numpy as np
from typing import List, Optional
import shutil


def run_bass_batch(image_dir_path: str, 
             output_dir_path: str,
             bass_path: Optional[str] = '/workspace/adv_robustness/region_explainability/BASS',
             n: Optional[int] = 15,
             im_size: Optional[int] = 0,  # 0 im_size means that no resize will happen
             i_std: Optional[float] = 0.018,
             alpha: Optional[float] = 0.5,
             beta: Optional[float] = 0.5,
             show_bass_output: Optional[bool] = False
            ) -> None:
    """Runs the BASS algorithm on a batch of images, and moves the output to the user-specificed output_dir_path.
    
    Args:
        image_dir_path (str): Where the to-be-segmented images are located
        output_dir_path (str): Where the segmentation output should go (csv mask files, and segmented images)
        bass_program_path (str): Where BASS dir is located (compiled build for BASS algorithm)
        n (int): the desired number of pixels on the side of a superpixel
        i_std (float): std dev for color Gaussians, should be 0.01 <= value <= 0.05. 
                       A smaller value leads to more irregular superpixels
        im_size (int): resizing input images (single number)
        beta (float): beta (Potts) 0 < value < 10
        alpha (float): alpha (Hasting ratio) 0.01 < value < 100
        
    Returns:
        None
    """
    results_dir = os.path.join(bass_path, 'result/')
    results_dir_list = [file_name for file_name in os.listdir(results_dir) if not file_name.startswith('.')]
    
    # removing previous output from results, if exists
    if len(results_dir_list) != 0:
        for file in results_dir_list:
            os.remove(os.path.join(results_dir, file))
    
    # running BASS program
    bass_build_path = os.path.join(bass_path, 'build')
    args = ['./Sp_demo_for_direc', '-d', image_dir_path,
            '-n', str(n),
            '--im_size', str(im_size), 
            '--i_std', str(i_std),
            '--alpha', str(alpha),
            '--beta', str(beta)]
    print('Running the following subprocess command:\n', *args)
    subprocess.run(args, cwd=bass_build_path, capture_output=not(show_bass_output))
            
    # grabbing all output from results dir, and sending it to output_dir_path
    results_dir_list = [file_name for file_name in os.listdir(results_dir) if not file_name.startswith('.')]
    
    # if output already exists with the same name, delete it, and then replace
    for file in results_dir_list:
        source = os.path.join(results_dir, file)
        destination = os.path.join(output_dir_path, file)
        if os.path.exists(destination):
            os.remove(destination)
        shutil.move(source, destination)
        

def csv_mask_to_numpy(csv_path: str) -> np.ndarray:
    """Converts a csv_mask_file into a numpy arrays.
    
    Args:
        csv_mask_dir_path (str): The path to csv mask directory.
            Note, this should be the same as 'output_dir_path' from run_bass() function
        
    Returns:
        A list of np array masks.
    """
    # - 1 so that the superpixel values start at 0
    np_array = np.loadtxt(csv_path, delimiter=",", dtype=int)
    np_array = np.repeat(np.repeat(np_array , 4, axis=0), 4, axis=1)
    return np_array


def csv_mask_to_numpy_SAM(csv_path: str) -> np.ndarray:
    """Converts a csv_mask_file into a numpy arrays.
    
    Args:
        csv_mask_dir_path (str): The path to csv mask directory.
            Note, this should be the same as 'output_dir_path' from run_bass() function
        
    Returns:
        A list of np array masks.
    """
    # - 1 so that the superpixel values start at 0
    np_array = np.loadtxt(csv_path, delimiter=",", dtype=int)
    return np_array
