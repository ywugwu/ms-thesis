# utils/evaluation_utils.py

import torch
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import spearmanr

def compute_zero_shot_accuracy(image_embeddings, image_labels, text_embeddings, text_labels, batch_size=1024):
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

    for start in tqdm(range(0, num_images, batch_size), desc="Computing Zero-Shot Predictions"):
        end = min(start + batch_size, num_images)
        batch_image = image_features[start:end]  # Shape: [batch_size, embedding_dim]

        # Compute cosine similarity
        similarity = torch.matmul(batch_image, text_features.T)  # Shape: [batch_size, num_text_embeddings]

        # Get indices of most similar text embeddings
        _, max_indices = similarity.max(dim=1)  # Shape: [batch_size]

        # Map indices to labels
        batch_predicted_labels = [text_labels[idx] for idx in max_indices.cpu().numpy()]
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

def compute_zero_shot_accuracy_on_pseudo_images(pseudo_data, text_embeddings, text_labels, batch_size=1024):
    """
    Compute zero-shot accuracy on pseudo images.
    """
    pseudo_embeddings_np = np.array([item['embedding'] for item in pseudo_data])
    pseudo_labels_list = [item['label'] for item in pseudo_data]

    pseudo_per_class_accuracy, pseudo_overall_accuracy, pseudo_class_correct, pseudo_class_total = compute_zero_shot_accuracy(
        image_embeddings=pseudo_embeddings_np,
        image_labels=pseudo_labels_list,
        text_embeddings=text_embeddings,
        text_labels=text_labels,
        batch_size=batch_size
    )

    return pseudo_per_class_accuracy, pseudo_overall_accuracy, pseudo_class_correct, pseudo_class_total

def compare_and_correlate_accuracies(real_per_class_accuracy, pseudo_per_class_accuracy, n_classes):
    """
    Compare and correlate accuracies between pseudo images and real images.
    """
    # Filter labels with available accuracies
    filtered_labels = [
        label for label in n_classes 
        if real_per_class_accuracy.get(label) is not None and pseudo_per_class_accuracy.get(label) is not None
    ]
    
    # Extract accuracies
    actual_accuracies = [real_per_class_accuracy[label] for label in filtered_labels]
    pseudo_accuracies = [pseudo_per_class_accuracy[label] for label in filtered_labels]
    
    return actual_accuracies, pseudo_accuracies, filtered_labels

def knn_classification(text_embeddings, text_labels, n_neighbors=10):
    X = text_embeddings
    labels_array = np.array(text_labels)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_array)

    # Stratified split: train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
    train_index, test_index = next(sss.split(X, y))

    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn_classifier.fit(X_train, y_train)

    # Predict probabilities on test data
    y_proba = knn_classifier.predict_proba(X_test)

    # For each test sample, get the probability assigned to the correct class
    prob_correct_class = y_proba[np.arange(len(y_test)), y_test]

    return y_test, prob_correct_class, label_encoder
