import json
import os
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
import torch
from tqdm import tqdm

from utils.metrics import *

from config import DEVICE, RESULTS_DIR, GOOGLE_API_KEYs, EMB_MODEL_PATH, GPT2_MODEL_PATH, CLASSIFIER_PATHS


# ─── Helper Function ──────────────────────────────────────────────────────────


def get_json_data(results_dir: str) -> list:
    """Extracts the information from the filenames and returns the dataframes."""

    def extract_info_from_filename(filename: str):
        parts = filename.split('_')
        if len(parts) < 5:
            raise ValueError(f"Filename '{filename}' does not have enough parts to extract the information.")
        
        llm_name = '_'.join(parts[:3])
        task = parts[3]
        return llm_name, task

    filenames = os.listdir(results_dir)
    filenames = [f for f in filenames if f.endswith('.json')]
    filenames = sorted(filenames)

    data = []  # saves the list of (llm_name, task_name, middle_results)
    for filename in filenames:
        llm_name, task_name = extract_info_from_filename(filename)
        middle_results = pd.read_json(os.path.join(results_dir, filename), lines=True).dropna()
        data.append((llm_name, task_name, middle_results))

    return data

# ─── Main Loop ────────────────────────────────────────────────────────────────

stats = []
# saves the list of dictionaries with the scores
# each dictionary has the following keys: llm, generator, task, metric, avg score

cached_stats_paths = []
# saves the paths to the cached results
# these are for the case that the code crashes and we want to continue from where we left off

for llm_name, task_name, middle_results in tqdm(get_json_data(RESULTS_DIR), desc='Main loop'):

    # ─── 1. Get Some Meta Data For The Current Llm And Task ───────────────

    classifier_info = CLASSIFIER_PATHS.get(task_name, {})
    classifier_path = classifier_info.get('model_path', None)
    classifier_base_model_path = classifier_info.get('base_model_path', None)

    generator_names = [col for col in middle_results.columns if col not in ['prompt', 'ground_truth']]
    reference_texts = middle_results['ground_truth'].tolist()

    # ─── 2. Loop Over All Kinds Of Generators And Metrics ─────────────────

    results = {generator: defaultdict(list) for generator in generator_names}
    for generator in tqdm(generator_names, desc="Generator"):
        print()  # Add a new line for better readability

        # ─── 2.1. Get All Generated Texts ─────────────────────────────

        generated_texts = middle_results[generator].tolist()

        # ─── 2.2. Calculate Batch Scores For All Metrics ──────────────
        # Feel free to add or comment out any of the following metrics
        
        if task_name in ['PosToNeg', 'NegToPos']:
            results[generator]['Success'] = calculate_classifier_scores(generated_texts, classifier_path, classifier_base_model_path, task=task_name, device=DEVICE)

        results[generator]['CosScore'] = calculate_cos_scores(generated_texts, reference_texts, EMB_MODEL_PATH, device=DEVICE)
        results[generator]['Perplexity'] = calculate_ppl_scores(generated_texts, GPT2_MODEL_PATH, device=DEVICE)

        if task_name in ['toxicRandom', 'toxicTop']:
            results[generator]['Toxicity'] = calculate_toxicity_scores(generated_texts, GOOGLE_API_KEYs)


        # ─── 2.3. Calculate Average Score ─────────────────────────────

        for metric in results[generator].keys():
            results[generator][metric] = pd.DataFrame(results[generator][metric]).mean(skipna=True).item()

    # ─── 3. Save The Results ──────────────────────────────────────────────
            
    metric_names = results[generator].keys()
    
    for generator, metric in product(generator_names, metric_names):
        stats.append({
            'llm': llm_name,
            'generator': generator,
            'task': task_name,
            'metric': metric,
            'score': results[generator][metric]
        })

    # ─── 4. Cache The Results To ./.cache ─────────────────────────────────

    path = f'./.cache/{llm_name}_{task_name}_stats_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f)
    cached_stats_paths.append(path)

# ─── Save Final Stats To A Csv File ───────────────────────────────────────────

stats_df = pd.DataFrame(stats).pivot_table(index=['llm', 'generator'], columns=['task', 'metric'], values='score')
stats_df.to_csv(f'./stats/stats_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
