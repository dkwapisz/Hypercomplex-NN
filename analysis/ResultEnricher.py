import os
import json

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def update_json_with_f1_score(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    precision = data['evaluation_result']['Precision']
    recall = data['evaluation_result']['Recall']
    f1_score = calculate_f1_score(precision, recall)

    data['evaluation_result']['F1-Score'] = f1_score

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_all_training_jsons(phases, runs):
    for phase in phases:
        for run in runs:
            phase_run_path = os.path.join(f'phase{phase}', f'run{run}')
            for root, _, files in os.walk(phase_run_path):
                if 'training.json' in files:
                    file_path = os.path.join(root, 'training.json')
                    update_json_with_f1_score(file_path)

phases = range(16, 17)
runs = range(1, 4)

process_all_training_jsons(phases, runs)