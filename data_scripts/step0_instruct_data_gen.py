"""
This script generates a JSON instruct datafile with the answers left blank for inference.
"""
import json
from typing import Tuple, List
from pathlib import Path
import os
import glob
import random
import yaml
import csv

with open('../configs/data_script_config.yml', 'r') as f:
    config = yaml.safe_load(f)

num_images_per_split = config['images_per_split']
output_folder_testsplit = Path(config['output_folder_testsplit']).resolve()
output_folder_trainsplit = Path(config['output_folder_trainsplit']).resolve()
data_path = Path(config['data_path'])
train_instruct_path = output_folder_trainsplit / config['step2_basename_output']
test_instruct_path = output_folder_testsplit / config['step2_basename_output']
generate_testsplit: bool = config['generate_testsplit']

output_path = output_folder_testsplit if generate_testsplit else output_folder_trainsplit

NSFW = 'nsfw'
SELF_HARM = 'self_harm'
CYBERBULLYING = 'cyberbullying'
GENERIC = 'generic'
ANNOTATIONS = 'annotations'
IMAGES = 'images'
UNSAFE = 'unsafe'
SAFE = 'safe'

DATASETS = [
    NSFW,
    SELF_HARM,
    CYBERBULLYING,
    GENERIC
]

SPLITS = [
    SAFE,
    UNSAFE
]


ALL_QUESTIONS = [
    'What is each person doing in the image?',
    'Does the image contain explicit signs of threat, aggression, or offensive gestures that may be considered disrespectful, insulting, or violent?',
    'Does the image contain any text, symbols, or body language conveying hate speech, insults, or derogatory language?',
    'Does the image contain any individual who could be pointing a weapon in an aggressive or provocative manner?',
    'Does the image contain pornographic content or any form of sexually explicit activity?',
    'Does the image depict any explicit sexual content that does not originate in popular screen media, such as a film or TV series?',
    'Does the image involve sexual content that could not be interpreted as artistic expression, professional modeling, or educational contexts?',
    'Does the image depict self-cutting, self-inflicted wounds, or any other explicit depictions of self-harm methods?',
    'Does the image display individuals promoting, glorifying self-harm behavior, or expressing suicidal intent?',
    'Does the image portray behaviors, practices, or symbols associated with anorexia, eating disorders, or other forms of self-destructive behavior?',
]

GENERIC_QUESTIONS = [
    'What is happening in the image?',
    'What are the people doing?',
    'Does this image contain educational content?',
    'Does the image contain support contact information?',
    'Does the image contain support contact information for mental health?',
    'Does the image contain phone numbers for support contact?'
]

NSFW_QUESTIONS = [
    'Does the image contain explicit sexual content?',
    'Are there any minors in this image?',
    'Does the image contain pornographic content?',
    'Is there any sexually explicit activity portrayed in the image?',
    'Does the image suggest sexual solicitation?',
    'Is the image content suggestive of being part of a work of art?',
    'Is there any sexual content or nudity in the image?',
]

SH_QUESTIONS = [
    'Are there any depictions of self-harm in the image?',
    'Does the image suggest the promotion of self-harm behavior?',
    'Is there any content related to eating disorders in the image?',
    'Does the image contain symbols associated with self-destructive behavior?',
    'Are there visible signs of self-bruising or self-harm through physical trauma?',
    
]

CB_QUESTIONS = [
    'Is there any aggressive behavior shown in the image?',
    'Does the image contain potentially offensive gestures or symbols?',
    'Is there any text in the image that might be considered hate speech?',
    'Does the image contain any weapons?',
    'Does the image contain any hateful memes?',
    'Are there visual indicators of online shaming or humiliation',
]

# QUESTIONS_MAP = {
#     GENERIC : GENERIC_QUESTIONS,
#     NSFW : NSFW_QUESTIONS,
#     SELF_HARM : SH_QUESTIONS,
#     CYBERBULLYING : CB_QUESTIONS,
# }

NUM_QUESTIONS = len(ALL_QUESTIONS)

NEGATIVE_DEFAULT_ANSWERS = [
    'Placeholder',
    'No, the image does not contain explicit signs of threat, aggression, or offensive gestures that may be considered disrespectful, insulting, or violent.',
    'No, the image does not contain any text, symbols, or body language conveying hate speech, insults, or derogatory language.',
    'No, the image does not contain any individual who could be pointing a weapon in an aggressive or provocative manner.',
    'No, the image does not contain pornographic content or any form of sexually explicit activity.',
    'No, the image does not depict any explicit sexual content that does not originate in popular screen media, such as a film or TV series.',
    'No, the image does not involve sexual content that could not be interpreted as artistic expression, professional modeling, or educational contexts.',
    'No, the image does not depict self-cutting, self-inflicted wounds, or any other explicit depictions of self-harm methods.',
    'No, the image does not display individuals promoting or glorifying self-harm behavior, or expressing suicidal intent.',
    'No, the image does not portray behaviors, practices, or symbols associated with anorexia, eating disorders, or other forms of self-destructive behavior.',
]

POSITIVE_DEFAULT_ANSWERS = [
    'Placeholder',
    'Yes, the image contains explicit signs of threat, aggression, or offensive gestures that may be considered disrespectful, insulting, or violent.',
    'Yes, the image contains text, symbols, or body language conveying hate speech, insults, or derogatory language.',
    'Yes, the image contains an individual who is pointing a weapon in an aggressive or provocative manner.',
    'Yes, the image contains pornographic content or a form of sexually explicit activity.',
    'Yes, the image depicts explicit sexual content that does not originate in popular screen media, such as a film or TV series.',
    'Yes, the image involves sexual content that could not be interpreted as artistic expression, professional modeling, or educational contexts.',
    'Yes, the image depicts self-cutting, self-inflicted wounds, or other explicit depictions of self-harm methods.',
    'Yes, the image displays individuals promoting or glorifying self-harm behavior, or expressing suicidal intent.',
    'Yes, the image portrays behaviors, practices, or symbols associated with anorexia, eating disorders, or other forms of self-destructive behavior.',
]

QUESTION_TYPE_MAPPING = [
    GENERIC,
    CYBERBULLYING,
    CYBERBULLYING,
    CYBERBULLYING,
    NSFW,
    NSFW,
    NSFW,
    SELF_HARM,
    SELF_HARM,
    SELF_HARM,
]

assert len(ALL_QUESTIONS) == len(
    QUESTION_TYPE_MAPPING), 'Question mapping length does not equal to the questions length'

assert len(ALL_QUESTIONS) == len(
    POSITIVE_DEFAULT_ANSWERS), 'Question length does not equal to the default answer length'

assert len(ALL_QUESTIONS) == len(
    NEGATIVE_DEFAULT_ANSWERS), 'Question length does not equal to the default answer length'


def _get_image_list_from_json(json_file: Path):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = [Path(item['image']).name for item in data]
    return images


def _grab_random_image_paths(num_images: int, image_dir: Path,
                             check_train_image_list: bool = False) -> List[str]:

    if check_train_image_list:
        train_instruct_image_list = _get_image_list_from_json(
            train_instruct_path)
    else:
        train_instruct_image_list = []

    current_dir = os.getcwd()
    os.chdir(image_dir)
    all_images = set(glob.glob('*'))
    available_images = all_images - set(train_instruct_image_list)

    should_grab_num_random_images = len(available_images) >= num_images
    if not should_grab_num_random_images:
        print(f'Grabbing Entire Dataset from: {image_dir}')
        print(f'Only {len(available_images)} available')
    selected_images = random.sample(
        available_images, num_images) if should_grab_num_random_images else list(available_images)
    print(f'Grabbed {len(selected_images)} images.')
    os.chdir(current_dir)

    return selected_images


def grab_images(data_path: Path, data_types: list, subsplits: list,
                num_images_per_split: int = 50) -> List[Path]:

    for data_type in data_types[:]:
        if data_type == GENERIC:
            data_types.remove(data_type)

    image_paths = []
    for type in data_types:
        type_path = data_path / type
        for subsplit in subsplits:
            split_path = type_path / IMAGES / subsplit
            paths = _grab_random_image_paths(
                num_images_per_split, split_path, check_train_image_list=generate_testsplit)
            paths = [split_path / path for path in paths]
            image_paths.extend(paths)

    return image_paths


def safe_or_unsafe_and_dataset_type(path: Path) -> Tuple[str, str]:
    # dataset_type is referring to NSFW, SELF_HARM, OR CYBERBULLYING
    # and safe_or_unsafe is referring to SAFE or UNSAFE
    safe_or_unsafe = path.parent.name
    dataset_type = path.parent.parent.parent.name
    assert safe_or_unsafe in SPLITS, 'Image must be SAFE or UNSAFE (lowercase)'
    assert dataset_type in DATASETS, 'Image dataset type must be NSFW, SELF_HARM, OR CYBERBULLYING (lowercase)'
    return (safe_or_unsafe, dataset_type)



def main():
    image_path_list = grab_images(data_path, DATASETS, SPLITS,
                                  num_images_per_split=num_images_per_split)

    data = []

    for path in image_path_list:
        safe_or_unsafe, dataset_type = safe_or_unsafe_and_dataset_type(path)

        default_answers = []
        if safe_or_unsafe == SAFE:
            default_answers = NEGATIVE_DEFAULT_ANSWERS
        else:
            for i, questions_map in enumerate(QUESTION_TYPE_MAPPING):
                if questions_map == GENERIC:
                    default_answers.append(POSITIVE_DEFAULT_ANSWERS[i])
                elif questions_map == dataset_type:
                    default_answers.append(POSITIVE_DEFAULT_ANSWERS[i])
                else:
                    default_answers.append(NEGATIVE_DEFAULT_ANSWERS[i])

        # questions_block = QUESTIONS_MAP[GENERIC] + QUESTIONS_MAP[dataset_type]

        data.append({
            'image': str(path),
            'dataset_type': dataset_type,
            'safe_or_unsafe': safe_or_unsafe,
            'questions_map': QUESTION_TYPE_MAPPING,
            'questions': ALL_QUESTIONS,
            'answers': default_answers,
            'llm_answers': [' ']*len(ALL_QUESTIONS)
        })

    # save to JSON file
    with open(output_path / 'step0.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
