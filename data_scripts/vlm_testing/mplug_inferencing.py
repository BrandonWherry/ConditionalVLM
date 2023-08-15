"""This module requires it's own environment build! see './OFA' for details.
"""
import json
from PIL import Image
import torch
from tqdm import tqdm
from mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mPLUG_Owl.mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mPLUG_Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and processors
pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)

model.to(device)

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

# Load the JSON data
with open('vlm_testing_pass1_step1.json', 'r') as json_file:
    data = json.load(json_file)

# Loop through each item in the data
for item in tqdm(data, desc="Processing items"):
    image_path = item['image']
    print(f'Processing image: {image_path}')
    image = Image.open('../' + image_path)
    mplug_responses = []

    for i, question in enumerate(item['questions']):
        prompt = (
            "The following is a conversation between a curious human and AI assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f"Human: <image>\n"
            f"Human: {question}\n"
            "AI: "
        )

        inputs = processor(text=[prompt], images=[image], return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)

        # Extract the response and add to the mplug_responses list
        response = tokenizer.decode(res.tolist()[0], skip_special_tokens=True).split('AI: ')[-1]  # We only want the AI's answer
        mplug_responses.append(response)
        print(response)

    # Update the item with the mplug responses
    item['mplug_answers'] = mplug_responses

# Save the modified data back to the JSON file
with open('vlm_testing_pass1_step1_mplug.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Inference complete and JSON saved.")
