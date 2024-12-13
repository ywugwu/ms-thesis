# qualitative_analysis.py

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import (
    load_config,
    load_model,
    get_dataset,
    CaptionGenerator,
    get_text_embeddings,
    get_image_embeddings,
    compute_zero_shot_accuracy,
    knn_classification,
    combine_all_data,
    get_image_loader,
    CLIPTextConsistencyScorer,  # Import the consistency scorer
)
import torch
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
import json

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
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
    gpt_model = config['gpt_model']['name']

    # Output directory for qualitative analysis
    qualitative_dir = os.path.join(results_dir, 'qualitative_analysis')
    os.makedirs(qualitative_dir, exist_ok=True)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_cfg in config['models']:
        for dataset_cfg in config['datasets']:
            model_name = model_cfg['name']
            dataset_name = dataset_cfg['name']
            print(f"Processing model {model_name} on dataset {dataset_name}")

            # Load the model
            print(f"Loading model: {model_name}")
            model_info = load_model(model_name)

            # Initialize the consistency scorer
            print("Initializing CLIPTextConsistencyScorer...")
            consistency_scorer = CLIPTextConsistencyScorer(device=device)

            # Load the dataset
            print(f"Loading dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)
            n_classes = dataset.classes
            dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(n_classes)}

            # Generate prompts and captions
            standard_prompts = [f"a photo of a {class_name}" for class_name in n_classes]
            standard_labels = n_classes

            # Initialize CaptionGenerator
            capGenerator = CaptionGenerator(dataset_name=dataset_name, class_names=n_classes, num_captions=num_captions, model=gpt_model, prompt_template=dataset.prompt_template)
            alter_caption_list = []
            labels = []
            print("Generating alternative captions...")
            for class_name in n_classes:
                alter_captions = capGenerator.get_alternative_captions(class_name)
                alter_caption_list.extend(alter_captions)
                labels.extend([class_name] * len(alter_captions))

            # Compute descriptive text embeddings
            print("Computing descriptive text embeddings...")
            descriptive_text_embeddings = get_text_embeddings(alter_caption_list, model_info, device)
            descriptive_text_labels = labels

            # Compute standard text embeddings
            print("Computing standard text embeddings...")
            text_embeddings = get_text_embeddings(standard_prompts, model_info, device)
            text_labels = standard_labels

            # Compute image embeddings
            print("Computing image embeddings...")
            class_indices = [dataset.class_to_idx[class_name] for class_name in n_classes]
            image_loader = get_image_loader(dataset, class_indices, batch_size)
            image_embeddings, image_labels = get_image_embeddings(image_loader, model_info, device)
            image_labels = [dataset.classes[label] for label in image_labels]

            # Combine all data modalities
            descriptive_text_data = [
                {'embedding': emb, 'label': label, 'modality': 'text'}
                for emb, label in zip(descriptive_text_embeddings, descriptive_text_labels)
            ]
            standard_text_data = [
                {'embedding': emb, 'label': label, 'modality': 'standard'}
                for emb, label in zip(text_embeddings, text_labels)
            ]
            image_data = [
                {'embedding': emb, 'label': label, 'modality': 'image'}
                for emb, label in zip(image_embeddings, image_labels)
            ]
            combined_embeddings, combined_labels, combined_modalities = combine_all_data(
                descriptive_text_data, standard_text_data, image_data,
            )

            # Organize descriptive embeddings per class for consistency scorer
            descriptive_text_embeddings_per_class = []
            class_to_descriptive_embeddings = defaultdict(list)
            print("Organizing descriptive embeddings per class...")
            for emb, label in tqdm(zip(descriptive_text_embeddings, descriptive_text_labels)):
                class_to_descriptive_embeddings[label].append(emb)
            for class_name in n_classes:
                embeddings = class_to_descriptive_embeddings[class_name]
                if embeddings:
                    descriptive_text_embeddings_per_class.append(np.stack(embeddings))
                else:
                    # Append an empty array if no descriptive embeddings exist for the class
                    descriptive_text_embeddings_per_class.append(np.empty((0, text_embeddings.shape[1])))

            # Compute consistency scores
            print("Computing consistency scores...")
            consistency_scores = consistency_scorer.compute(
                standard_embeddings=text_embeddings,  # Shape: (C, D)
                descriptive_text_embeddings=descriptive_text_embeddings_per_class  # List of C tensors, each (N_c, D)
            )

            # Compute zero-shot accuracy on real images
            print("Computing zero-shot accuracy on real images...")
            real_per_class_accuracy, real_overall_accuracy, real_class_correct, real_class_total = compute_zero_shot_accuracy(
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                text_embeddings=text_embeddings,
                text_labels=text_labels,
                batch_size=knn_batch_size,
            )

            # Display results
            print(f"\nPer-Class Zero-Shot Accuracy on Real Images:")
            for class_name in n_classes:
                accuracy = real_per_class_accuracy.get(class_name)
                if accuracy is not None:
                    print(f"{class_name}: {accuracy:.2%} ({real_class_correct[class_name]}/{real_class_total[class_name]})")
                else:
                    print(f"{class_name}: No samples")
            print(f"\nOverall Zero-Shot Accuracy on Real Images: {real_overall_accuracy:.2%}")

            # KNN Classification on descriptive_text_embeddings
            print("Performing KNN classification...")
            y_test, prob_correct_class, label_encoder = knn_classification(descriptive_text_embeddings, descriptive_text_labels)
            labels_array = np.array(descriptive_text_labels)
            y = label_encoder.transform(labels_array)

            # Initialize dictionaries to accumulate probabilities per class
            class_prob_sums = defaultdict(float)
            class_counts = defaultdict(int)

            # Accumulate probabilities
            for i in range(len(y_test)):
                true_class_idx = y_test[i]
                true_class_label = label_encoder.inverse_transform([true_class_idx])[0]
                class_prob_sums[true_class_label] += prob_correct_class[i]
                class_counts[true_class_label] += 1

            # Compute per-class average probabilities
            knn_per_class_probabilities = {}
            for class_label in class_prob_sums:
                knn_per_class_probabilities[class_label] = class_prob_sums[class_label] / class_counts[class_label]

            # Convert accuracies and probabilities to arrays
            class_names = np.array(n_classes)
            real_accuracies = np.array([real_per_class_accuracy.get(cls, 0.0) for cls in n_classes])
            knn_probabilities = np.array([knn_per_class_probabilities.get(cls, 0.0) for cls in n_classes])

            # Create 2D plot to visualize KNN probabilities vs actual accuracies
            plt.figure(figsize=(8,6))
            plt.scatter(knn_probabilities, real_accuracies)
            for i, cls in enumerate(n_classes):
                plt.text(knn_probabilities[i], real_accuracies[i], cls, fontsize=8)
            plt.xlabel('KNN Probabilities')
            plt.ylabel('Zero-Shot Accuracy')
            plt.title(f'{model_name} on {dataset_name}')
            plt.grid(True)
            # Save plot
            plot_path = os.path.join(qualitative_dir, f"{model_name.replace('/', '_')}_{dataset_name}_scatter.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Scatter plot saved to {plot_path}")

            # Define thresholds for KNN probabilities
            knn_threshold = np.median(knn_probabilities)
            # Use fixed thresholds for accuracies
            high_acc_threshold = 0.8
            low_acc_threshold = 0.2

            # Identify classes in each case for KNN probabilities
            # Case 1: High KNN probabilities and high real zero-shot accuracy
            case1_indices_knn = np.where((knn_probabilities >= knn_threshold) & (real_accuracies >= high_acc_threshold))[0]
            # Case 2: High KNN probabilities and low real zero-shot accuracy
            case2_indices_knn = np.where((knn_probabilities >= knn_threshold) & (real_accuracies < low_acc_threshold))[0]
            # Case 3: Low KNN probabilities and high real zero-shot accuracy
            case3_indices_knn = np.where((knn_probabilities < knn_threshold) & (real_accuracies >= high_acc_threshold))[0]
            # Case 4: Low KNN probabilities and low real zero-shot accuracy
            case4_indices_knn = np.where((knn_probabilities < knn_threshold) & (real_accuracies < low_acc_threshold))[0]

            # Function to select the class based on KNN probabilities
            def select_class_by_knn(indices, knn_probs, real_accs, select_knn_max, select_acc_max):
                if len(indices) == 0:
                    return None
                # Select indices with extreme KNN probability
                if select_knn_max:
                    extreme_knn_value = np.max(knn_probs[indices])
                else:
                    extreme_knn_value = np.min(knn_probs[indices])
                extreme_knn_indices = indices[knn_probs[indices] == extreme_knn_value]
                # Among these, select based on zero-shot accuracy
                if len(extreme_knn_indices) > 1:
                    if select_acc_max:
                        extreme_acc_value = np.max(real_accs[extreme_knn_indices])
                    else:
                        extreme_acc_value = np.min(real_accs[extreme_knn_indices])
                    selected_indices = extreme_knn_indices[real_accs[extreme_knn_indices] == extreme_acc_value]
                    selected_idx = selected_indices[0]  # Take the first one if multiple
                else:
                    selected_idx = extreme_knn_indices[0]
                return selected_idx

            # Select classes for each KNN case
            # Case 1: High KNN probabilities and high real zero-shot accuracy
            case1_class_idx_knn = select_class_by_knn(
                case1_indices_knn, knn_probabilities, real_accuracies, select_knn_max=True, select_acc_max=True)

            # Case 2: High KNN probabilities and low real zero-shot accuracy
            case2_class_idx_knn = select_class_by_knn(
                case2_indices_knn, knn_probabilities, real_accuracies, select_knn_max=True, select_acc_max=False)

            # Case 3: Low KNN probabilities and high real zero-shot accuracy
            case3_class_idx_knn = select_class_by_knn(
                case3_indices_knn, knn_probabilities, real_accuracies, select_knn_max=False, select_acc_max=True)

            # Case 4: Low KNN probabilities and low real zero-shot accuracy
            case4_class_idx_knn = select_class_by_knn(
                case4_indices_knn, knn_probabilities, real_accuracies, select_knn_max=False, select_acc_max=False)

            # For each selected class, retrieve images and save them
            selected_cases_knn = [
                ('High KNN, High Acc', case1_class_idx_knn),
                ('High KNN, Low Acc', case2_class_idx_knn),
                ('Low KNN, High Acc', case3_class_idx_knn),
                ('Low KNN, Low Acc', case4_class_idx_knn),
            ]

            metadata_knn = []

            for case_name, class_idx in selected_cases_knn:
                if class_idx is None:
                    print(f"No class found for case: {case_name}")
                    continue
                class_name = n_classes[class_idx]
                knn_prob = knn_probabilities[class_idx]
                real_acc = real_accuracies[class_idx]

                print(f"Case: {case_name}")
                print(f"Class: {class_name}, KNN Prob: {knn_prob:.4f}, Zero-Shot Acc: {real_acc:.4f}")

                # Retrieve sample images
                class_images = get_images_for_class(dataset, class_name, num_images=5)

                # Save images and collect metadata
                images_dir = os.path.join(qualitative_dir, f"{model_name.replace('/', '_')}_{dataset_name}_{case_name.replace(' ', '_')}")
                os.makedirs(images_dir, exist_ok=True)
                image_filenames = []
                for i, img in enumerate(class_images):
                    img_save_path = os.path.join(images_dir, f"{class_name}_{i}.png")
                    save_image(img, img_save_path)
                    image_filenames.append(img_save_path)
                print(f"Images for class {class_name} saved to {images_dir}")

                # Append metadata
                metadata_knn.append({
                    'case': case_name,
                    'class_name': class_name,
                    'knn_probability': float(knn_prob),
                    'zero_shot_accuracy': float(real_acc),
                    'images': image_filenames,
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                })

            # Save KNN metadata to JSON file
            metadata_filepath_knn = os.path.join(qualitative_dir, f"{model_name.replace('/', '_')}_{dataset_name}_metadata_knn.json")
            with open(metadata_filepath_knn, 'w') as f:
                json.dump(metadata_knn, f, indent=4)
            print(f"KNN metadata saved to {metadata_filepath_knn}")

            # Now perform the same process for each consistency score
            for score_key in consistency_scores:
                print(f"\nProcessing consistency score: {score_key}")
                scores = np.array(consistency_scores[score_key])  # Scores per class
                # Ensure that scores are aligned with class names
                if scores.shape[0] != len(n_classes):
                    print(f"Error: Number of scores ({scores.shape[0]}) does not match number of classes ({len(n_classes)})")
                    continue

                # Define thresholds for consistency scores
                score_threshold = np.median(scores)

                # Identify classes in each case for the current consistency score
                # Case 1: High consistency score and high real zero-shot accuracy
                case1_indices_score = np.where((scores >= score_threshold) & (real_accuracies >= high_acc_threshold))[0]
                # Case 2: High consistency score and low real zero-shot accuracy
                case2_indices_score = np.where((scores >= score_threshold) & (real_accuracies < low_acc_threshold))[0]
                # Case 3: Low consistency score and high real zero-shot accuracy
                case3_indices_score = np.where((scores < score_threshold) & (real_accuracies >= high_acc_threshold))[0]
                # Case 4: Low consistency score and low real zero-shot accuracy
                case4_indices_score = np.where((scores < score_threshold) & (real_accuracies < low_acc_threshold))[0]

                # Function to select the class based on consistency scores
                def select_class_by_score(indices, scores, real_accs, select_score_max, select_acc_max):
                    if len(indices) == 0:
                        return None
                    # Select indices with extreme consistency score
                    if select_score_max:
                        extreme_score_value = np.max(scores[indices])
                    else:
                        extreme_score_value = np.min(scores[indices])
                    extreme_score_indices = indices[scores[indices] == extreme_score_value]
                    # Among these, select based on zero-shot accuracy
                    if len(extreme_score_indices) > 1:
                        if select_acc_max:
                            extreme_acc_value = np.max(real_accs[extreme_score_indices])
                        else:
                            extreme_acc_value = np.min(real_accs[extreme_score_indices])
                        selected_indices = extreme_score_indices[real_accs[extreme_score_indices] == extreme_acc_value]
                        selected_idx = selected_indices[0]  # Take the first one if multiple
                    else:
                        selected_idx = extreme_score_indices[0]
                    return selected_idx

                # Select classes for each case
                # Case 1: High score, high accuracy
                case1_class_idx_score = select_class_by_score(
                    case1_indices_score, scores, real_accuracies, select_score_max=True, select_acc_max=True)

                # Case 2: High score, low accuracy
                case2_class_idx_score = select_class_by_score(
                    case2_indices_score, scores, real_accuracies, select_score_max=True, select_acc_max=False)

                # Case 3: Low score, high accuracy
                case3_class_idx_score = select_class_by_score(
                    case3_indices_score, scores, real_accuracies, select_score_max=False, select_acc_max=True)

                # Case 4: Low score, low accuracy
                case4_class_idx_score = select_class_by_score(
                    case4_indices_score, scores, real_accuracies, select_score_max=False, select_acc_max=False)

                # For each selected class, retrieve images and save them
                selected_cases_score = [
                    ('High Consistency, High Acc', case1_class_idx_score),
                    ('High Consistency, Low Acc', case2_class_idx_score),
                    ('Low Consistency, High Acc', case3_class_idx_score),
                    ('Low Consistency, Low Acc', case4_class_idx_score),
                ]

                metadata_score = []

                for case_name, class_idx in selected_cases_score:
                    if class_idx is None:
                        print(f"No class found for case: {case_name}")
                        continue
                    class_name = n_classes[class_idx]
                    score_value = scores[class_idx]
                    real_acc = real_accuracies[class_idx]

                    print(f"Case: {case_name}")
                    print(f"Class: {class_name}, Consistency Score: {score_value:.4f}, Zero-Shot Acc: {real_acc:.4f}")

                    # Retrieve sample images
                    class_images = get_images_for_class(dataset, class_name, num_images=5)

                    # Save images and collect metadata
                    images_dir = os.path.join(qualitative_dir, f"{model_name.replace('/', '_')}_{dataset_name}_{score_key}_{case_name.replace(' ', '_')}")
                    os.makedirs(images_dir, exist_ok=True)
                    image_filenames = []
                    for i, img in enumerate(class_images):
                        img_save_path = os.path.join(images_dir, f"{class_name}_{i}.png")
                        save_image(img, img_save_path)
                        image_filenames.append(img_save_path)
                    print(f"Images for class {class_name} saved to {images_dir}")

                    # Append metadata
                    metadata_score.append({
                        'case': case_name,
                        'class_name': class_name,
                        'consistency_score': float(score_value),
                        'zero_shot_accuracy': float(real_acc),
                        'consistency_score_type': score_key,
                        'images': image_filenames,
                        'model_name': model_name,
                        'dataset_name': dataset_name,
                    })

                # Save consistency score metadata to JSON file
                metadata_filepath_score = os.path.join(qualitative_dir, f"{model_name.replace('/', '_')}_{dataset_name}_metadata_{score_key}.json")
                with open(metadata_filepath_score, 'w') as f:
                    json.dump(metadata_score, f, indent=4)
                print(f"Metadata for {score_key} saved to {metadata_filepath_score}")

def get_images_for_class(dataset, class_name, num_images=5):
    # Get indices of images for the class
    class_idx = dataset.class_to_idx[class_name]
    if hasattr(dataset, 'samples'):
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    elif hasattr(dataset, 'targets'):
        indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
    else:
        raise ValueError("Dataset does not have 'samples' or 'targets' attribute.")

    # Select images
    selected_indices = indices[:num_images]
    images = []
    for idx in selected_indices:
        if hasattr(dataset, 'samples'):
            img_path, _ = dataset.samples[idx]
            img = dataset.loader(img_path)
        else:
            img, _ = dataset[idx]
        img = transforms.ToTensor()(img)
        images.append(img)
    return images

if __name__ == '__main__':
    main()
