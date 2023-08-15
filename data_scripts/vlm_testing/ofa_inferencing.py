"""This module requires it's own environment build! see './mPLUG_Owl' for details.
"""
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from tqdm import tqdm

# Assuming you have the complete transformation sequence for images
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

ckpt_dir = 'OFA-Sys/ofa-large'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
model = OFAModel.from_pretrained(ckpt_dir, use_cache=True).to('cuda')

# Load the JSON data
with open('vlm_testing_pass1_step1.json', 'r') as json_file:
    data = json.load(json_file)

# Loop through each item in the data
for item in tqdm(data, desc="Processing items"):
    image_path = '../' + item['image']
    img = Image.open(image_path)
    patch_img = patch_resize_transform(img).unsqueeze(0).to('cuda')
    
    ofa_responses = []
    
    for question in item['questions']:
        inputs = tokenizer([question], return_tensors="pt", truncation=True, padding=True).input_ids.to('cuda')
        
        gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
        
        response = tokenizer.decode(gen[0], skip_special_tokens=True)
        ofa_responses.append(response)
        print(response)
    
    item['ofa_answers'] = ofa_responses

# Save the modified data back to the JSON file
with open('vlm_testing_pass1_step1_OFA_filled.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Inference complete and JSON saved.")
