# ensemble_zero_shot.py

import argparse
import os
import json
from collections import defaultdict, Counter
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
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Zero-Shot Accuracy Using Ensemble of Descriptive Captions")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file.')
    return parser.parse_args()


def compute_zero_shot_accuracy_ensemble(image_embeddings, image_labels, text_embeddings, text_labels, k, batch_size=1024):
    """
    Compute zero-shot accuracy using an ensemble approach based on top k descriptive text embeddings.

    Args:
        image_embeddings (np.ndarray): Array of image embeddings with shape (N_images, D).
        image_labels (List[str]): List of true labels for each image.
        text_embeddings (np.ndarray): Array of text embeddings with shape (N_text, D).
        text_labels (List[str]): List of labels corresponding to text_embeddings.
        k (int): Number of top closest descriptive text embeddings to consider for ensemble voting.
        batch_size (int, optional): Batch size for processing. Defaults to 1024.

    Returns:
        per_class_accuracy (Dict[str, float]): Per-class accuracies.
        overall_accuracy (float): Overall accuracy.
        class_correct (Dict[str, int]): Correct predictions per class.
        class_total (Dict[str, int]): Total samples per class.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_features = torch.from_numpy(image_embeddings).to(device)
    text_features = torch.from_numpy(text_embeddings).to(device)

    # Normalize embeddings
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    num_images = image_features.size(0)
    predicted_labels = []

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for start in tqdm(range(0, num_images, batch_size), desc="Computing Zero-Shot Predictions (Ensemble)"):
        end = min(start + batch_size, num_images)
        batch_image = image_features[start:end]  # Shape: [batch_size, D]

        # Compute cosine similarity
        similarity = torch.matmul(batch_image, text_features.T)  # Shape: [batch_size, N_text]

        # Get top k indices
        topk_similarities, topk_indices = similarity.topk(k, dim=1, largest=True, sorted=True)  # Each row: top k indices

        # Get predicted labels for top k
        batch_predicted_labels = []
        for indices in topk_indices:
            labels = [text_labels[idx.item()] for idx in indices]
            # Majority voting
            label_counts = Counter(labels)
            most_common_label, _ = label_counts.most_common(1)[0]
            batch_predicted_labels.append(most_common_label)

        predicted_labels.extend(batch_predicted_labels)

        # Update class_correct and class_total
        for true_label, pred_label in zip(image_labels[start:end], batch_predicted_labels):
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1

    # Compute per-class accuracies
    per_class_accuracy = {}
    for class_name in set(image_labels):
        if class_total[class_name] > 0:
            accuracy = class_correct[class_name] / class_total[class_name]
            per_class_accuracy[class_name] = accuracy
        else:
            per_class_accuracy[class_name] = None  # No samples for this class

    # Compute overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return per_class_accuracy, overall_accuracy, class_correct, class_total


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
            # processor = model_info['processor']  # Uncomment if needed

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
            capGenerator = CaptionGenerator(dataset_name=dataset_name, class_names=n_classes, num_captions=num_captions, model=gpt_model)
            alter_caption_list = []
            labels = []
            print("Generating alternative captions...")
            for class_name in n_classes:
                alter_captions = capGenerator.get_alternative_captions(class_name)
                alter_caption_list.extend(alter_captions)
                labels.extend([class_name] * len(alter_captions))

            # Compute descriptive text embeddings (CxN_c x D)
            print("Computing descriptive text embeddings...")
            descriptive_text_embeddings = get_text_embeddings(alter_caption_list, model_info, device)
            descriptive_text_labels = labels

            # Compute image embeddings
            print("Computing image embeddings...")
            class_indices = [dataset.class_to_idx[class_name] for class_name in n_classes]
            image_loader = get_image_loader(dataset, class_indices, batch_size)
            image_embeddings, image_labels = get_image_embeddings(image_loader, model_info, device)
            image_labels = [dataset.classes[label] for label in image_labels]

            
            # Compute zero-shot accuracy using ensemble of descriptive captions
            print("Computing zero-shot accuracy using ensemble of descriptive captions...")
            # Set k as the number of captions per class
            k = num_captions
            per_class_accuracy_ensemble, overall_accuracy_ensemble, class_correct_ensemble, class_total_ensemble = compute_zero_shot_accuracy_ensemble(
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                text_embeddings=descriptive_text_embeddings,
                text_labels=descriptive_text_labels,
                k=1,
                batch_size=knn_batch_size,
            )

            # Optionally, you can also compute standard zero-shot accuracy for comparison
            print("Computing zero-shot accuracy using standard text embeddings for comparison...")
            real_per_class_accuracy_std, real_overall_accuracy_std, real_class_correct_std, real_class_total_std = compute_zero_shot_accuracy(
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                text_embeddings=(standard_prompts_embeddings := get_text_embeddings(standard_prompts, model_info, device)),
                text_labels=standard_labels,
                batch_size=knn_batch_size,
            )

            # Calculate performance gain
            print("Calculating performance gain...")
            per_class_gain = {}
            for class_name in n_classes:
                acc_std = real_per_class_accuracy_std.get(class_name, 0)
                acc_ensemble = per_class_accuracy_ensemble.get(class_name, 0)
                gain = acc_ensemble - acc_std
                per_class_gain[class_name] = gain
                print(f"{class_name}: Standard Acc = {acc_std:.2%}, Ensemble Acc = {acc_ensemble:.2%}, Gain = {gain:.2%}")

            overall_gain = overall_accuracy_ensemble - real_overall_accuracy_std
            print(f"Overall Accuracy - Standard: {real_overall_accuracy_std:.2%}, Ensemble: {overall_accuracy_ensemble:.2%}, Gain: {overall_gain:.2%}")

            # Store the gains in the performance_gains dictionary
            performance_gains[f"{model_name}_{dataset_name}"] = {
                'overall_gain': overall_gain,
                # 'overall_accuracy_std': real_overall_accuracy_std,
                # 'overall_accuracy_ensemble': overall_accuracy_ensemble,
                # 'per_class_gain': per_class_gain,
            }

            # Cleanup to free memory
            import gc
            del descriptive_text_embeddings, descriptive_text_labels
            del standard_prompts_embeddings, standard_prompts
            del image_embeddings, image_labels
            del real_per_class_accuracy_std, real_overall_accuracy_std, real_class_correct_std, real_class_total_std
            del per_class_accuracy_ensemble, overall_accuracy_ensemble, class_correct_ensemble, class_total_ensemble
            del per_class_gain
            del model_info
            gc.collect()

    # After all models and datasets are processed, save the aggregated performance gains
    if save_results_flag:
        aggregated_results_filepath = os.path.join(results_dir, "performance_gains_ensemble_zero_shot.json")
        with open(aggregated_results_filepath, 'w') as f:
            json.dump(performance_gains, f, indent=4)
        print(f"\nAggregated Performance Gains saved to {aggregated_results_filepath}")


if __name__ == '__main__':
    main()
