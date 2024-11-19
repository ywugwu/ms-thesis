import os
import requests
import hashlib
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from datasets import load_dataset
import cv2

def load_class_names(classnames_url, data_root, dataset_name):
    """
    Downloads and loads class names from the provided URL.
    """
    classnames_path = os.path.join(data_root, dataset_name, "classnames.txt")
    if not os.path.exists(classnames_path):
        os.makedirs(os.path.dirname(classnames_path), exist_ok=True)
        response = requests.get(classnames_url)
        response.raise_for_status()
        with open(classnames_path, "w") as f:
            f.write(response.text)
    with open(classnames_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_prompt_templates(prompt_url, data_root, dataset_name):
    """
    Downloads and loads prompt templates from the provided URL.
    """
    prompt_path = os.path.join(data_root, dataset_name, "zeroshot_classification_templates.txt")
    if not os.path.exists(prompt_path):
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        response = requests.get(prompt_url)
        response.raise_for_status()
        with open(prompt_path, "w") as f:
            f.write(response.text)
    with open(prompt_path, "r") as f:
        prompt_templates = [line.strip() for line in f.readlines()]
    return prompt_templates

class ClipBenchmarkDataset(Dataset):
    """
    Custom PyTorch Dataset for CLIP Benchmark datasets.
    Loads images directly from the 'webp', 'jpg', or 'png' keys provided by the dataset.
    Includes prompt templates for zeroshot classification.
    """
    def __init__(self, split, dataset_name, data_root='data'):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load the dataset from Hugging Face
        self.hf_dataset = load_dataset(f"clip-benchmark/{dataset_name}", split=split)

        # Load class names manually
        classnames_url = f"https://huggingface.co/datasets/clip-benchmark/{dataset_name}/resolve/main/classnames.txt"
        self.class_names = load_class_names(classnames_url, data_root, dataset_name)

        # Load prompt templates
        prompt_url = f"https://huggingface.co/datasets/clip-benchmark/{dataset_name}/resolve/main/zeroshot_classification_templates.txt"
        self.prompt_template = load_prompt_templates(prompt_url, data_root, dataset_name)

        self.targets = self.hf_dataset['cls']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        class_id = sample['cls']

        if 'webp' in sample:
            image = sample['webp']
        elif 'jpg' in sample:
            image = sample['jpg']
        elif 'png' in sample:
            image = sample['png']
        else:
            raise ValueError("No image found in the sample.")

        try:
            image = image.resize((224, 224))
        except:
            # If image is not a PIL Image, try using OpenCV
            image = cv2.resize(image, (224, 224))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Verify that 'image' is a PIL Image
        if not isinstance(image, Image.Image):
            print(f"Expected PIL.Image.Image for sample index {idx}, but got {type(image)}.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            # Optionally, you can perform any additional processing here
            pass

        if self.transform:
            image = self.transform(image)

        return image, class_id  # Removed prompts from the returned tuple

class CustomImageFolder(torchvision.datasets.ImageFolder):
    """
    Custom ImageFolder that includes prompt templates.
    """
    def __init__(self, root, prompt_template, transform=None, class_format=lambda c: c):
        super().__init__(root, transform=transform)
        self.prompt_template = [prompt_template]  # Store as a list for consistency
        self.class_format = class_format
        # Apply class_format to class names
        self.classes = [self.class_format(c) for c in self.classes]

    def __getitem__(self, index):
        # Get image and label from the original ImageFolder
        image, label = super().__getitem__(index)

        return image, label  # Removed prompts from the returned tuple

def get_dataset(dataset_name, data_root='data', split='test'):
    """
    Returns a PyTorch Dataset based on the dataset_name.
    Supports both existing datasets and CLIP Benchmark datasets.
    Includes prompt templates for zeroshot classification.
    """
    if dataset_name == "CUB_200_2011":
        dataset_path = os.path.join(data_root, "CUB_200_2011", "images")
        class_format = lambda c: c[4:].replace('_', ' ')
        prompt_template = "a photo of a {c}, a type of bird."
        dataset = CustomImageFolder(
            root=dataset_path,
            prompt_template=prompt_template,
            transform=transforms.ToTensor(),
            class_format=class_format
        )
    elif dataset_name == "Flower102":
        dataset_path = os.path.join(data_root, "Flower102")
        class_format = lambda c: c
        prompt_template = "a photo of a {c}, a type of flower."
        dataset = CustomImageFolder(
            root=dataset_path,
            prompt_template=prompt_template,
            transform=transforms.ToTensor(),
            class_format=class_format
        )
    elif dataset_name == "Stanford_dogs":
        dataset_path = os.path.join(data_root, "Stanford_dogs", "Images")
        class_format = lambda c: ' '.join(c.split('-')[1:]).replace('_', ' ')
        prompt_template = "a photo of a {c}, a type of dog."
        dataset = CustomImageFolder(
            root=dataset_path,
            prompt_template=prompt_template,
            transform=transforms.ToTensor(),
            class_format=class_format
        )
    elif dataset_name == "NWPU-RESISC45":
        dataset_path = os.path.join(data_root, "NWPU-RESISC45")
        class_format = lambda c: c.replace('_', ' ')
        prompt_template = "a satellite image containing {c}."
        dataset = CustomImageFolder(
            root=dataset_path,
            prompt_template=prompt_template,
            transform=transforms.ToTensor(),
            class_format=class_format
        )
    elif dataset_name.startswith("wds_") or dataset_name.startswith("wds_vtab-"):
        # Handle CLIP Benchmark datasets
        clip_dataset_name = dataset_name
        split_name = split  # 'train' or 'test'
        dataset = ClipBenchmarkDataset(split=split_name, dataset_name=clip_dataset_name, data_root=data_root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name.startswith("wds_") or dataset_name.startswith("wds_vtab-"):
        # For CLIP Benchmark datasets, set classes attribute from class_names
        class_format = lambda c: c.replace('_', ' ')  # Adjust if needed per dataset
        dataset.classes = [class_format(c) for c in dataset.class_names]
    else:
        # For CustomImageFolder datasets, classes are already formatted
        pass
    
    return dataset

if __name__ == "__main__":
    # CLIP Benchmark Datasets
    clip_datasets = [
        "wds_cars",
        "wds_fgvc_aircraft",
        "wds_food101",
        "wds_imagenetv2",
        "wds_objectnet",
        "wds_sun397",
        "wds_vtab-cifar100",
        "wds_vtab-flowers",
        "wds_vtab-pets",
        "wds_vtab-resisc45"
    ]
    
    for clip_dataset_name in clip_datasets:
        try:
            dataset = get_dataset(clip_dataset_name, "data", split='train')
            print(f"{clip_dataset_name} classes:", dataset.classes)
            print(f"{clip_dataset_name} prompt templates:", dataset.prompt_template)
            # Try loading one image            
            image, class_id = dataset[0]
            print(f"Loaded image and class_id {class_id} from {clip_dataset_name}")
        except Exception as e:
            print(f"Error loading {clip_dataset_name}: {e}")

    # Existing Datasets
    existing_datasets = [
        ("CUB_200_2011", "a photo of a {c}, a type of bird."),
        ("Flower102", "a photo of a {c}, a type of flower."),
        ("Stanford_dogs", "a photo of a {c}, a type of dog."),
        ("NWPU-RESISC45", "a satellite image containing {c}.")
    ]
    
    for dataset_name, expected_prompt in existing_datasets:
        try:
            dataset = get_dataset(dataset_name, "data")
            print(f"{dataset_name} classes:", dataset.classes)
            print(f"{dataset_name} prompt templates:", dataset.prompt_template)
            image, class_id = dataset[0]
            print(f"Loaded image and class_id {class_id} from {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
