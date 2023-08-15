import json
from pathlib import Path
from typing import List, Dict
import os
import re
import yaml

with open('../configs/data_script_config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_name_or_path: str = config['model_name_or_path']
data_path: Path = Path(config['data_path']).resolve()
step1_basename_output: Path = Path(config['step1_basename_output'])

output_folder_testsplit = Path(config['output_folder_testsplit']).resolve()
output_folder_trainsplit = Path(config['output_folder_trainsplit']).resolve()

train_instruct_path = output_folder_trainsplit / \
    config['step2_basename_output']
test_instruct_path = output_folder_testsplit / config['step2_basename_output']
generate_testsplit: bool = config['generate_testsplit']

output_path = output_folder_testsplit if generate_testsplit else output_folder_trainsplit

step0_path = Path(output_folder_testsplit / config['step0_basename_output'])

OVERALL = 'overall'
NSFW = 'nsfw'
SELF_HARM = 'self_harm'
CYBERBULLYING = 'cyberbullying'
GENERIC = 'generic'
SAFE = 'safe'
UNSAFE = 'unsafe'
DATASETS = [NSFW, SELF_HARM, CYBERBULLYING]
SPLITS = [SAFE, UNSAFE]
SAFE_OR_UNSAFE = 'safe_or_unsafe'
DATASET_TYPE = 'dataset_type'
QUESTIONS_MAP = 'questions_map'
LLM_ANSWERS = 'llm_answers'
ANSWERS = 'answers'
YES = 'yes'
NO = 'no'
QUESTIONS = 'questions'
IMAGE = 'image'


def get_first_word(s):
    words = s.split()
    first_word = words[0]
    first_word = first_word.strip(',')
    first_word = first_word.lower()
    return first_word


def save_dataset_as_json(instruct_data: dict, json_output_path: Path) -> None:
    with open(json_output_path, 'w') as f:
        json.dump(instruct_data, f, indent=4)


def print_options(human_response: str, llm_response: str, image_description: str, question: str, image: str) -> None:
    print(f'\n\nImage: {image}')
    print('Image Description:')
    print(image_description)
    print('\nQuestion:')
    print(question)
    print('\n1. Default answer:')
    print(human_response, '\n')
    print('2. LLM answer:')
    print(llm_response, '\n\n')
    print('Options:')
    print('1. Pick Default Response')
    print('2. Pick LLM Response')
    print('3. Enter New Response')
    print('4. Save and Exit')


def parse_response(num: int, data: List[Dict], i: int, j: int) -> bool:
    """Returns a bool on whether or not to break out of the loop and save"""
    assert num in list(range(1, 5)), 'Invalid Choice, must be between 1 and 4'

    llm_response = data[j][LLM_ANSWERS][i]
    response = data[j][ANSWERS][i]

    if num == 1:
        data[j][LLM_ANSWERS][i] = response
    elif num == 2:
        data[j][ANSWERS][i] = llm_response
    elif num == 3:
        new_response = input('New Response: ')
        data[j][LLM_ANSWERS][i] = new_response
        data[j][ANSWERS][i] = new_response
    elif num == 4:
        return True

    return False


def main():
    # Open the JSON file
    with open(step0_path) as file:
        data = json.load(file)

    save_and_exit = False

    for j, entry in enumerate(data):
        if entry[DATASET_TYPE] != CYBERBULLYING or entry[SAFE_OR_UNSAFE] != UNSAFE:
            continue

        for i, question_map in enumerate(entry[QUESTIONS_MAP]):
            if i != 3:
                continue

            image = entry[IMAGE]
            question = entry[QUESTIONS][i]
            image_description = entry[ANSWERS][0]
            llm_response = entry[LLM_ANSWERS][i]
            response = entry[ANSWERS][i]

            llm_sentiment = get_first_word(llm_response)
            human_sentiment = get_first_word(response)

            if llm_sentiment == human_sentiment:
                entry[ANSWERS][i] = llm_response
                continue

            weapon_keywords = ['gun', 'knife', 'weapon', 'guns', 'knives', 'weapons', 'rifle', 'rifles']

            has_weapon_keyword = False

            for word in weapon_keywords:
                if re.search(r'\b' + word + r'\b', image_description, re.IGNORECASE):
                    has_weapon_keyword = True

            if has_weapon_keyword:
                entry[LLM_ANSWERS][i] = response
                continue

            print_options(response, llm_response,
                          image_description, question, image)

            selection = int(input('Enter an Option: '))

            save_and_exit = parse_response(selection, data, i, j)

            if save_and_exit:
                break

            os.system('clear')

        if save_and_exit:
            break

    save_dataset_as_json(data, output_path)


if __name__ == '__main__':
    main()
