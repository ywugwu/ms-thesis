# main.py
import argparse
import os
import json
from collections import defaultdict
from scipy.stats import spearmanr
from utils import (
    load_config,
    load_model,
    get_dataset,
    CaptionGenerator,
    get_text_embeddings,
    get_image_embeddings,
    compute_zero_shot_accuracy,
    compute_zero_shot_accuracy_on_pseudo_images,
    compare_and_correlate_accuracies,
    knn_classification,
    plot_accuracy_correlation,
    plot_comparison_accuracies,
    plot_knn_vs_actual_accuracies,
    plot_tsne,
    get_image_loader,
    PseudoImageDataset,
    combine_all_data,
    plot_consistency_scores,  # Ensure this is defined in utils.py
    CLIPTextConsistencyScorer,
    plot_consistency_scores,
)

import numpy as np
from torch.utils.data import DataLoader
import torch  # Needed for device check
from tqdm import tqdm
import zipfile

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Extract experiment parameters
    experiment_params = config['experiment_params']
    batch_size = experiment_params['batch_size']
    knn_batch_size = experiment_params['knn_batch_size']
    num_captions = experiment_params['num_captions']
    tsne_perplexity = experiment_params['tsne_perplexity']
    tsne_n_iter = experiment_params['tsne_n_iter']
    top_k_classes = experiment_params['top_k_classes']
    pseudo_data_flag = experiment_params['pseudo_data']
    save_results_flag = experiment_params['save_results']
    results_dir = experiment_params['results_dir']
    pseudo_images_folder = experiment_params.get('pseudo_images_folder', 'pseudo_images')
    consistency_scorer_config = config.get('consistency_scorer', {})
    consistency_scorer_enabled = consistency_scorer_config.get('enable', False)
    gpt_model = config['gpt_model']['name']
    # if cuda is available, use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Loop over models and datasets
    for model_cfg in config['models']:
        for dataset_cfg in config['datasets']:
            dataset_json_paths = []
            model_name = model_cfg['name']

            dataset_name = dataset_cfg['name']
            plotfile_path = os.path.join('figures', f"{model_name.replace('/', '_')}_{dataset_name}")
            os.makedirs(plotfile_path, exist_ok=True)
            print(f"\nProcessing dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)
            n_classes = dataset.classes
            n_object = len(n_classes)
            dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(n_classes)}

            # Initialize CaptionGenerator
            capGenerator = CaptionGenerator(dataset_name=dataset_name, class_names=n_classes, num_captions=num_captions, model = gpt_model, prompt_template=dataset.prompt_template)
            # print("Generating alternative captions...")
            for class_name in n_classes:
                capGenerator.get_alternative_captions(class_name)
                cache_path = capGenerator._generate_cache_path(class_name)
                dataset_json_paths.append(cache_path)
            
            # Create a zip file containing all dataset JSON files
            zip_path = f'cache/{dataset_name}.zip'
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for json_path in dataset_json_paths:
                    if os.path.exists(json_path):
                        zipf.write(json_path, os.path.basename(json_path))
            print(f"Created zip file at {zip_path}")


if __name__ == '__main__':
    main()
