import multiprocessing
import torch
import warnings
import transformers
import random
import io
from pathlib import Path
from PIL import Image
from datasets import load_dataset
import datasets
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, Trainer
from dataclasses import dataclass, field
from typing import Optional
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine
import argparse
import yaml
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="deepspeed")


NUM_CORES = multiprocessing.cpu_count()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Salesforce/instructblip-vicuna-7b")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    images_path: str = field(default=None, metadata={
        "help": "Path to the images."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_strategy: str
    save_strategy: str
    save_total_limit: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    fp16: bool
    lr_scheduler_type: str
    deepspeed: str
    log_level: Optional[str] = 'info'
    log_level_replica: Optional[str] = 'critical'
    remove_unused_columns: bool = False
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def load_image(example):
    image_path = Path(example['image']).resolve()
    image = Image.open(image_path)
    if image.mode == 'P':  # P mode means the image has a palette
        image = image.convert('RGBA')
    image = image.convert('RGB').resize((224, 224))
    example['image'] = np.array(image)
    return example


class InstructBlipDataCollator:
    def __init__(self, processor: InstructBlipProcessor):
        self.processor = processor

    def __call__(self, examples):
        images = [Image.fromarray(np.array(ex['image']).astype(np.uint8)).convert('RGB')
                  for ex in examples]

        images = self.processor.image_processor(images, return_tensors='pt')

        question_inputs_qformer = self.processor.qformer_tokenizer(
            [ex['question'] for ex in examples], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        question_inputs_llm = self.processor.tokenizer(
            [ex['question'] for ex in examples], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        # answer_inputs = self.processor.tokenizer(
        #     [ex['answer'] for ex in examples], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        concatenated_inputs = self.processor.tokenizer(
            [ex['question'] + " " + ex['answer'] for ex in examples], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return {
            'qformer_input_ids': question_inputs_qformer['input_ids'],
            'input_ids': question_inputs_llm['input_ids'],
            'qformer_attention_mask': question_inputs_qformer['attention_mask'],
            'attention_mask': question_inputs_llm['attention_mask'],
            'labels': concatenated_inputs['input_ids'],
            'pixel_values': images['pixel_values']
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_args = ModelArguments(**config['model_args'])
    data_args = DataArguments(**config['data_args'])
    training_args = TrainingArguments(**config['training_args'])

    train_data_path = Path(data_args.train_data_path).absolute()
    # image_path_base = Path(data_args.images_path).absolute()

    train_dataset = load_dataset(
        'json', data_files=str(train_data_path), split='train').shuffle(seed=42)

    # train_test_split = train_dataset.train_test_split(test_size=0.99)

    # train_dataset = train_test_split['train']
    # test_dataset = train_test_split['test'] # Not needed here

    mapped_train_dataset = train_dataset.map(load_image, num_proc=4, writer_batch_size=500)

    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path
    )

    # Freezing Vision Model
    for param in model.vision_model.parameters():
        param.requires_grad = False

    for param in model.qformer.parameters():
        param.requires_grad = True

    # Freezing Language Model
    for param in model.language_model.parameters():
        param.requires_grad = False

    processor = InstructBlipProcessor.from_pretrained(
        model_args.model_name_or_path
    )

    data_collator = InstructBlipDataCollator(
        processor=processor
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mapped_train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()

    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

