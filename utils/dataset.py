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

class ClipBenchmarkDataset(Dataset):
    """
    Custom PyTorch Dataset for CLIP Benchmark datasets.
    Loads images directly from the 'webp' key provided by the dataset.
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
            image = cv2.resize(image, (224, 224))
        # Verify that 'image' is a PIL Image
        if not isinstance(image, Image.Image):
            print(f"Expected PIL.Image.Image for sample index {idx}, but got {type(image)}.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            # Optionally, you can perform any additional processing here
            pass

        if self.transform:
            image = self.transform(image)

        return image, class_id


def get_dataset(dataset_name, data_root='data', split='test',):
    """
    Returns a PyTorch Dataset based on the dataset_name.
    Supports both existing datasets and CLIP Benchmark datasets.
    """
    # if transform is None:
    #     # Define default transformations if none are provided
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ])
    
    if dataset_name == "CUB_200_2011":
        dataset_path = os.path.join(data_root, "CUB_200_2011", "images")
        class_format = lambda c: c[4:].replace('_', ' ')
        dataset = torchvision.datasets.ImageFolder(dataset_path, )
    elif dataset_name == "Flower102":
        dataset_path = os.path.join(data_root, "Flower102")
        class_format = lambda c: c
        dataset = torchvision.datasets.ImageFolder(dataset_path, )
    elif dataset_name == "Stanford_dogs":
        dataset_path = os.path.join(data_root, "Stanford_dogs", "Images")
        class_format = lambda c: ' '.join(c.split('-')[1:]).replace('_', ' ')
        dataset = torchvision.datasets.ImageFolder(dataset_path, )
    elif dataset_name == "NWPU-RESISC45":
        dataset_path = os.path.join(data_root, "NWPU-RESISC45")
        class_format = lambda c: c.replace('_', ' ')
        dataset = torchvision.datasets.ImageFolder(dataset_path, )
    elif dataset_name.startswith("wds_") or dataset_name.startswith("wds_vtab-"):
        # Handle CLIP Benchmark datasets
        # Example dataset names: wds_fgvc_aircraft, wds_cars, etc.
        clip_dataset_name = dataset_name
        split_name = split  # 'train' or 'test'
        class_format = lambda c: c.replace('_', ' ')  # Adjust if needed per dataset
        
        dataset = ClipBenchmarkDataset(split=split_name, dataset_name=clip_dataset_name, data_root=data_root,)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name.startswith("wds_") or dataset_name.startswith("wds_vtab-"):
        # For CLIP Benchmark datasets, set classes attribute from class_names
        dataset.classes = [class_format(c) for c in dataset.class_names]
    else:
        # For ImageFolder datasets, format the classes
        dataset.classes = [class_format(c) for c in dataset.classes]
    
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
            # try load one image            
        except Exception as e:
            print(f"Error loading {clip_dataset_name}: {e}")
        image, class_id = dataset[0]

    # Existing Datasets
    try:
        dataset = get_dataset("CUB_200_2011", "data",)
        print("CUB_200_2011 classes:", dataset.classes)
    except Exception as e:
        print(f"Error loading CUB_200_2011: {e}")
    try:
        dataset = get_dataset("Flower102", "data", )
        print("Flower102 classes:", dataset.classes)
    except Exception as e:
        print(f"Error loading Flower102: {e}")
    
    try:
        dataset = get_dataset("Stanford_dogs", "data", )
        print("Stanford_dogs classes:", dataset.classes)
    except Exception as e:
        print(f"Error loading Stanford_dogs: {e}")
    
    try:
        dataset = get_dataset("NWPU-RESISC45", "data", )
        print("NWPU-RESISC45 classes:", dataset.classes)
    except Exception as e:
        print(f"Error loading NWPU-RESISC45: {e}")
    