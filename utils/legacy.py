# legacy.py

import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class CLIPTextConsistencyScorer:
    def __init__(self, device: str = None) -> None:
        """
        Initialize the CLIPTextConsistencyScorer.

        Args:
            device (str, optional): The device to use ('cuda' or 'cpu'). 
                                    Defaults to 'cpu' since NumPy operations are device-agnostic.
        """
        # Device is not needed for NumPy operations, but kept for consistency
        self.device = device if device else "cpu"

    def compute(
        self,
        standard_embeddings: np.ndarray,
        descriptive_text_embeddings: List[np.ndarray]
    ) -> Dict[str, List[float]]:
        """
        Compute consistency scores based on precomputed text embeddings.

        Args:
            standard_embeddings (np.ndarray): Array of standard text embeddings with shape (C, D).
            descriptive_text_embeddings (List[np.ndarray]): 
                List of arrays, each containing descriptive text embeddings for a class with shape (N_c, D).

        Returns:
            Dict[str, List[float]]: 
                Dictionary containing text-only consistency scores and classification margin scores per class.
        """
        text_only_consistency_scores = []
        classification_margin_scores = []

        num_classes, dim = standard_embeddings.shape

        # Normalize standard embeddings
        standard_embeddings_norm = standard_embeddings / np.linalg.norm(standard_embeddings, axis=1, keepdims=True)  # shape (C, D)

        # Compute pairwise cosine similarities between standard embeddings
        standard_cosine_sim = np.dot(standard_embeddings_norm, standard_embeddings_norm.T)  # shape (C, C)
        # Set diagonal to -inf to exclude self-similarity in max computations
        np.fill_diagonal(standard_cosine_sim, -np.inf)

        # Compute mean descriptive embeddings per class and normalize them
        per_class_mean_descriptive_embeddings = []
        for embeddings in descriptive_text_embeddings:
            if embeddings.shape[0] > 0:
                mean_embedding = np.mean(embeddings, axis=0)
            else:
                mean_embedding = np.zeros((dim,))
            per_class_mean_descriptive_embeddings.append(mean_embedding)
        per_class_mean_descriptive_embeddings = np.array(per_class_mean_descriptive_embeddings)  # shape (C, D)
        per_class_mean_descriptive_embeddings_norm = per_class_mean_descriptive_embeddings / np.linalg.norm(per_class_mean_descriptive_embeddings, axis=1, keepdims=True)

        # Compute standard_embeddings_diff for T_i - T_j
        standard_embeddings_diff = standard_embeddings_norm[:, np.newaxis, :] - standard_embeddings_norm[np.newaxis, :, :]  # shape (C, C, D)
        # Precompute norms of standard_embeddings_diff
        norms_standard_embeddings_diff = np.linalg.norm(standard_embeddings_diff, axis=2)  # shape (C, C)

        # Iterate over each class
        for c in tqdm(range(num_classes), desc="Computing consistency scores per class"):
            N_c = descriptive_text_embeddings[c].shape[0]
            if N_c == 0:
                # Handle classes with no descriptive embeddings
                text_only_consistency_scores.append(0.0)
                classification_margin_scores.append(0.0)
                continue

            # Normalize descriptive embeddings for class c
            descriptive_embeddings_c = descriptive_text_embeddings[c]  # shape (N_c, D)
            descriptive_embeddings_c_norm = descriptive_embeddings_c / np.linalg.norm(descriptive_embeddings_c, axis=1, keepdims=True)  # shape (N_c, D)

            # Mask to exclude class c
            mask = np.ones(num_classes, dtype=bool)
            mask[c] = False

            # Compute s_i^E components
            # Max interclass similarity
            max_interclass_sim = np.max(standard_cosine_sim[c, mask])  # scalar

            # Min intraclass similarity between T_c and T_i^d
            T_c_norm = standard_embeddings_norm[c, :]  # shape (D,)
            cos_sims_intraclass = np.dot(descriptive_embeddings_c_norm, T_c_norm)  # shape (N_c,)
            min_intraclass_sim = np.min(cos_sims_intraclass)  # scalar

            s_i_E = -max_interclass_sim + min_intraclass_sim  # scalar

            # Compute s_k^T and s'_k^T for each descriptive embedding in class c
            s_k_T_list = []
            s_k_T_prime_list = []
            for k in range(N_c):
                T_k_d_norm = descriptive_embeddings_c_norm[k, :]  # shape (D,)

                # Compute T_k^d - \overline{T}_j^d for all j != c
                diff_T_d = T_k_d_norm - per_class_mean_descriptive_embeddings_norm[mask, :]  # shape (C-1, D)
                norms_diff_T_d = np.linalg.norm(diff_T_d, axis=1)  # shape (C-1,)

                # Retrieve T_c - T_j for all j != c
                diff_T_std = standard_embeddings_diff[c, mask, :]  # shape (C-1, D)
                norms_diff_T_std = norms_standard_embeddings_diff[c, mask]  # shape (C-1,)

                # Compute cosine similarities for s_k^T
                dot_products_T = np.sum(diff_T_d * diff_T_std, axis=1)  # shape (C-1,)
                denom_T = norms_diff_T_d * norms_diff_T_std  # shape (C-1,)
                # Handle zero denominators
                denom_T = np.where(denom_T == 0, 1e-8, denom_T)
                cos_sims_T = dot_products_T / denom_T  # shape (C-1,)

                s_k_T = np.min(cos_sims_T)  # scalar
                s_k_T_list.append(s_k_T)

                # Compute cosine similarities for s'_k^T
                # Cosine between T_k^d and T_c - T_j
                dot_products_T_prime = np.dot(diff_T_std, T_k_d_norm)  # shape (C-1,)
                # Since T_k_d_norm is normalized, denom is norms_diff_T_std
                denom_T_prime = norms_diff_T_std  # shape (C-1,)
                denom_T_prime = np.where(denom_T_prime == 0, 1e-8, denom_T_prime)
                cos_sims_T_prime = dot_products_T_prime / denom_T_prime  # shape (C-1,)

                s_k_T_prime = np.min(cos_sims_T_prime)  # scalar
                s_k_T_prime_list.append(s_k_T_prime)

            # Compute final scores for class c
            S_i = (1 / N_c) * np.sum(s_k_T_list) + s_i_E
            S_i_prime = (1 / N_c) * np.sum(s_k_T_prime_list) + s_i_E

            text_only_consistency_scores.append(S_i)
            classification_margin_scores.append(S_i_prime)

        return {
            "text_only_consistency_scores": text_only_consistency_scores,
            "classification_margin_scores": classification_margin_scores,
        }
