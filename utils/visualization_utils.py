# utils/visualization_utils.py

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os

def plot_accuracy_correlation(actual_accuracies, predicted_accuracies, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_accuracies, actual_accuracies, color='blue', alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='y=x')
    plt.xlabel('Predicted Accuracy')
    plt.ylabel('Actual Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute Spearman correlation
    spearman_corr, p_value = spearmanr(actual_accuracies, predicted_accuracies)
    print(f"\nSpearman Correlation Coefficient: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.4e}")

    err = np.abs(np.array(actual_accuracies) - np.array(predicted_accuracies))
    print(f"Mean Absolute Error: {np.mean(err):.4f}")

def plot_comparison_accuracies(actual_accuracies, pseudo_accuracies, filtered_labels, title,save_path=None):
    """
    Plot predicted vs actual accuracies with swapped axes.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(pseudo_accuracies, actual_accuracies, color='blue', alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='y=x')
    plt.xlabel('Zero-Shot Accuracy (Pseudo Images)')
    plt.ylabel('Actual Zero-Shot Accuracy (Real Images)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'comparison_accuracies.pdf'))
    else:
        plt.show()
    

    # Compute Spearman correlation
    spearman_corr, p_value = spearmanr(actual_accuracies, pseudo_accuracies)
    print(f"\nSpearman Correlation Coefficient: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.4e}")

    err = np.abs(np.array(actual_accuracies) - np.array(pseudo_accuracies))
    print(f"Mean Absolute Error: {np.mean(err):.4f}")

def plot_knn_vs_actual_accuracies(knn_per_class_probabilities, real_per_class_accuracy, n_classes, title, save_path=None):
    """
    Plot KNN average probabilities vs actual accuracies.
    """
    # Filter labels that have both actual accuracies and KNN average probabilities
    filtered_labels_knn = [
        label for label in n_classes
        if real_per_class_accuracy.get(label) is not None and knn_per_class_probabilities.get(label) is not None
    ]
    
    # Extract accuracies and average probabilities
    actual_accuracies_knn = [real_per_class_accuracy[label] for label in filtered_labels_knn]
    knn_avg_probabilities = [knn_per_class_probabilities[label] for label in filtered_labels_knn]
    
    # Plot KNN average probabilities vs actual accuracies
    plt.figure(figsize=(10, 10))
    plt.scatter(knn_avg_probabilities, actual_accuracies_knn, color='green', alpha=0.6, label='KNN Predictions')
    plt.plot([0, 1], [0, 1], 'r--', label='y = x')  # Reference line
    
    plt.xlabel('KNN Predicted Accuracy')
    plt.ylabel('Actual Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'knn_vs_actual_accuracies.pdf'))
    else:
        plt.show()
    
    # Compute Spearman correlation
    spearman_corr_knn, p_value_knn = spearmanr(knn_avg_probabilities, actual_accuracies_knn)
    print(f"\nSpearman Correlation Coefficient (KNN Avg Probabilities vs Actual): {spearman_corr_knn:.4f}")
    print(f"P-value: {p_value_knn:.4e}")
    
    # Compute absolute differences between KNN predicted accuracies and actual accuracies
    absolute_errors = np.abs(np.array(knn_avg_probabilities) - np.array(actual_accuracies_knn))
    
    # Compute the average error
    average_error = np.mean(absolute_errors)
    
    print(f"Average Error of KNN Prediction vs Actual Accuracies: {average_error:.4f}")

def plot_tsne(combined_embeddings, combined_labels, combined_modalities, 
              per_class_accuracy, n_classes, top_k=12, selected_modalities=None,
              tsne_perplexity=30, tsne_n_iter=1000, dataset_name='Dataset', save_path=None):
    if selected_modalities is None:
        selected_modalities = ['text', 'standard', 'image', 'pseudo_image']

    from adjustText import adjust_text

    # Select the first k classes
    top_k_classes = n_classes[:top_k]
    print(f"Visualizing the first {top_k} classes: {top_k_classes}")

    # Create a mask using NumPy for proper boolean indexing
    mask = np.array([
        (mod in selected_modalities) and (label in top_k_classes)
        for label, mod in zip(combined_labels, combined_modalities)
    ])

    # Apply the mask to filter embeddings, labels, and modalities
    filtered_embeddings = combined_embeddings[mask]
    filtered_labels = np.array(combined_labels)[mask]
    filtered_modalities = np.array(combined_modalities)[mask]

    if filtered_embeddings.size == 0:
        print("No data points match the selected modalities and classes.")
        return

    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=tsne_n_iter)
    tsne_results = tsne.fit_transform(filtered_embeddings)

    # Assign colors to classes
    unique_classes = sorted(set(filtered_labels))
    cmap = plt.get_cmap('tab20')  # Up to 20 distinct colors
    colors = cmap.colors
    class_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_classes)}

    # Define markers for modalities
    markers = {
        'text': 'x',
        'standard': 's',
        'image': 'o',
        'pseudo_image': '^'
    }

    plt.figure(figsize=(18, 18))

    # Group data by class and modality for efficient plotting
    for class_name in unique_classes:
        for modality in selected_modalities:
            # Create a boolean mask for the current class and modality
            group_mask = (filtered_labels == class_name) & (filtered_modalities == modality)
            group_embeddings = tsne_results[group_mask]

            if group_embeddings.size == 0:
                continue  # Skip if no data points in this group

            plt.scatter(
                group_embeddings[:, 0],
                group_embeddings[:, 1],
                color=class_to_color[class_name],
                marker=markers.get(modality, 'o'),  # Default to 'o' if modality not found
                edgecolors='w',
                linewidth=0.5,
                label=f"{class_name} - {modality}"  # Unique label for legend
            )

    # Create legend entries for modalities
    modality_handles = [
        Line2D([], [], color='black', marker=markers[mod], linestyle='None',
               markersize=10, label=mod.capitalize().replace('_', ' '))
        for mod in selected_modalities
    ]

    # Create legend entries for classes (colors)
    class_patches = [
        mpatches.Patch(color=class_to_color[label], label=label) 
        for label in unique_classes
    ]

    # Combine modality handles and class patches
    all_handles = modality_handles + class_patches

    # Place the legend outside the plot
    plt.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc='upper left', 
               title='Legend', fontsize=10, title_fontsize=12)

    plt.title(f't-SNE Visualization on {dataset_name}', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'tsne_visualization.pdf'))
    else:
        plt.show()
