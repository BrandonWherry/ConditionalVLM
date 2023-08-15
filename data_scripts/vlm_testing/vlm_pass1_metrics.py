import json

# Load the JSON data
with open('vlm_testing_pass1_step2.json', 'r') as json_file:
    data = json.load(json_file)

# Model names
models = ["instructblip", "ofa", "mplug"]

# Initialize results
results = {
    model: {
        'nsfw': {'total': 0, 'correct': 0},
        'cyberbullying': {'total': 0, 'correct': 0},
        'self_harm': {'total': 0, 'correct': 0}
    }
    for model in models
}

# Populate results based on the data
for item in data:
    true_bucket = item['dataset_type']
    for model in models:
        predicted_bucket = item[f'{model}_bucket']
        results[model][true_bucket]['total'] += 1
        if predicted_bucket == true_bucket:
            results[model][true_bucket]['correct'] += 1

# Compute and print accuracies
for model, dataset_results in results.items():
    print(f"\nModel: {model}")
    print("=================================")
    for dataset, counts in dataset_results.items():
        if counts['total'] > 0:
            accuracy = counts['correct'] / counts['total'] * 100
        else:
            accuracy = 0
        print(f"{dataset}: {accuracy:.2f}% ({counts['correct']} out of {counts['total']})")

print("\nAccuracy report complete.")