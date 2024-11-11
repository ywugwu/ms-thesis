# utils/data_utils.py

import os
import glob
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import numpy as np

def get_image_loader(dataset, class_indices, batch_size, shuffle=False):
    if hasattr(dataset, 'targets'):
        selected_indices = [i for i, label in enumerate(dataset.targets) if label in class_indices]
    else:
        selected_indices = [i for i, (_, label) in enumerate(dataset.imgs) if label in class_indices]
    selected_dataset = Subset(dataset, selected_indices)

    transform = None
    selected_dataset.dataset.transform = transform

    loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1])  # Custom collate function
    )
    return loader

class PseudoImageDataset(Dataset):
    def __init__(self, image_folder, class_names, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = set(class_names)  # For faster lookup

        for image_path in glob.glob(os.path.join(image_folder, '*.png')):
            filename = os.path.basename(image_path)
            class_name = filename.split('-')[0].replace('_', ' ')
            if class_name in self.class_names:
                self.image_paths.append(image_path)
                self.labels.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def combine_all_data(text_data, standard_data, image_data, pseudo_data=None):
    """
    Combine all data modalities into a single dataset.
    """
    combined_data = text_data + standard_data + image_data
    if pseudo_data is not None:
        combined_data += pseudo_data

    combined_embeddings = np.array([item['embedding'] for item in combined_data])
    combined_labels = [item['label'] for item in combined_data]
    combined_modalities = [item['modality'] for item in combined_data]

    return combined_embeddings, combined_labels, combined_modalities
