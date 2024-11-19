# mean_zero_shot.py

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
)

import numpy as np
from torch.utils.data import DataLoader
import torch  # Needed for device check
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Zero-Shot Accuracy Using Mean Embeddings")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file.')
    return parser.parse_args()


def compute_mean_embeddings(descriptive_text_embeddings, descriptive_text_labels, n_classes):
    """
    Compute the mean embedding for each class.

    Args:
        descriptive_text_embeddings (List[np.ndarray]): List of descriptive text embeddings.
        descriptive_text_labels (List[str]): Corresponding list of labels for each embedding.
        n_classes (List[str]): List of class names.

    Returns:
        np.ndarray: Array of mean embeddings for each class with shape (C, D).
    """
    class_to_embeddings = defaultdict(list)
    for emb, label in zip(descriptive_text_embeddings, descriptive_text_labels):
        class_to_embeddings[label].append(emb)
    
    mean_embeddings = []
    for class_name in n_classes:
        embeddings = class_to_embeddings[class_name]
        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_embeddings.append(mean_emb)
        else:
            # Handle classes with no embeddings
            mean_embeddings.append(np.zeros(descriptive_text_embeddings[0].shape))
    
    mean_embeddings = np.stack(mean_embeddings)  # Shape: (C, D)
    return mean_embeddings


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

    # Initialize a dictionary to store performance gains
    performance_gains = {}

    # Loop over models and datasets
    for model_cfg in config['models']:
        for dataset_cfg in config['datasets']:
            model_name = model_cfg['name']
            dataset_name = dataset_cfg['name']
            print(f"\nProcessing Model: {model_name} | Dataset: {dataset_name}")

            # Initialize the results storage for this model-dataset pair
            performance_gains[f"{model_name}_{dataset_name}"] = {}


            print(f"Loading model: {model_name}")
            model_info = load_model(model_name)

            plotfile_path = os.path.join('figures', f"{model_name.replace('/', '_')}_{dataset_name}")
            os.makedirs(plotfile_path, exist_ok=True)
            print(f"\nProcessing dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)
            n_classes = dataset.classes
            n_object = len(n_classes)
            dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(n_classes)}

            # Generate prompts and captions
            standard_prompts = [f"a photo of a {class_name}" for class_name in n_classes]
            standard_labels = n_classes

            # Initialize CaptionGenerator
            capGenerator = CaptionGenerator(dataset_name=dataset_name, class_names=n_classes, num_captions=num_captions, model=gpt_model, prompt_template=dataset.prompt_template)
            alter_caption_list = []
            labels = []
            print("Generating alternative captions...")
            print("Prompt template: ", dataset.prompt_template)
            for class_name in n_classes:
                alter_captions = capGenerator.get_alternative_captions(class_name)
                alter_caption_list.extend(alter_captions)
                labels.extend([class_name] * len(alter_captions))

            # Compute descriptive text embeddings (CxN_c x D)
            print("Computing descriptive text embeddings...")
            descriptive_text_embeddings = get_text_embeddings(alter_caption_list, model_info, device)
            descriptive_text_labels = labels

            # Compute mean embeddings per class
            print("Computing mean embeddings per class...")
            mean_text_embeddings = compute_mean_embeddings(descriptive_text_embeddings, descriptive_text_labels, n_classes)
            mean_text_labels = standard_labels  # Same labels as standard prompts

            # Compute standard text embeddings (CxD)
            print("Computing standard text embeddings...")
            text_embeddings = get_text_embeddings(standard_prompts, model_info, device)
            text_labels = standard_labels

            # Ensure embeddings are NumPy arrays
            if isinstance(text_embeddings, torch.Tensor):
                text_embeddings = text_embeddings.cpu().numpy()
            elif isinstance(text_embeddings, list):
                text_embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in text_embeddings]
            
            # check the shape of mean_text_embeddings and text_embeddings are same
            assert mean_text_embeddings.shape[1] == text_embeddings.shape[1], "Mean and standard text embeddings should have the same dimension."
            assert mean_text_embeddings.shape[0] == text_embeddings.shape[0], "Mean and standard text embeddings should have the same number of classes."
            # avg mean text and standard text embeddings
            mean_text_embeddings = (mean_text_embeddings + text_embeddings) / 2

            # Compute image embeddings
            print("Computing image embeddings...")
            class_indices = [dataset.class_to_idx[class_name] for class_name in n_classes]
            image_loader = get_image_loader(dataset, class_indices, batch_size)
            image_embeddings, image_labels = get_image_embeddings(image_loader, model_info, device)
            image_labels = [dataset.classes[label] for label in image_labels]

            
            print("Computing zero-shot accuracy using standard text embeddings...")
            real_per_class_accuracy_std, real_overall_accuracy_std, real_class_correct_std, real_class_total_std = compute_zero_shot_accuracy(
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                text_embeddings=text_embeddings,
                text_labels=text_labels,
                batch_size=knn_batch_size,
            )

            # Compute zero-shot accuracy using mean text embeddings
            print("Computing zero-shot accuracy using mean text embeddings...")
            real_per_class_accuracy_mean, real_overall_accuracy_mean, real_class_correct_mean, real_class_total_mean = compute_zero_shot_accuracy(
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                text_embeddings=mean_text_embeddings,
                text_labels=mean_text_labels,
                batch_size=knn_batch_size,
            )

            # Calculate performance gain
            print("Calculating performance gain...")
            per_class_gain = {}
            for class_name in n_classes:
                acc_std = real_per_class_accuracy_std.get(class_name, 0)
                acc_mean = real_per_class_accuracy_mean.get(class_name, 0)
                gain = acc_mean - acc_std
                per_class_gain[class_name] = gain
                print(f"{class_name}: Standard Acc = {acc_std:.2%}, Mean Acc = {acc_mean:.2%}, Gain = {gain:.2%}")

            overall_gain = real_overall_accuracy_mean - real_overall_accuracy_std
            print(f"Overall Accuracy - Standard: {real_overall_accuracy_std:.2%}, Mean: {real_overall_accuracy_mean:.2%}, Gain: {overall_gain:.2%}")

            # Store the gains in the performance_gains dictionary
            performance_gains[f"{model_name}_{dataset_name}"] = {
                'overall_gain': overall_gain
            }

            # Cleanup to free memory
            import gc
            del mean_text_embeddings, mean_text_labels
            del descriptive_text_embeddings, descriptive_text_labels
            del text_embeddings, text_labels
            del image_embeddings, image_labels
            del real_per_class_accuracy_std, real_overall_accuracy_std, real_class_correct_std, real_class_total_std
            del real_per_class_accuracy_mean, real_overall_accuracy_mean, real_class_correct_mean, real_class_total_mean
            del per_class_gain
            del model_info
            gc.collect()

    # After all models and datasets are processed, save the aggregated performance gains
    if save_results_flag:
        aggregated_results_filepath = os.path.join(results_dir, "performance_gains_mean_zero_shot.json")
        with open(aggregated_results_filepath, 'w') as f:
            json.dump(performance_gains, f, indent=4)
        print(f"\nAggregated Performance Gains saved to {aggregated_results_filepath}")


if __name__ == '__main__':
    main()
