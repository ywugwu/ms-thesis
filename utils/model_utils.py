# utils/model_utils.py

import torch
import torch.nn.functional as F
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
    FlavaModel,          # Added for FLAVA
    FlavaProcessor,     # Added for FLAVA
    SiglipProcessor,
    SiglipModel,
    FlavaFeatureExtractor,
    BertTokenizer,
)
from tqdm import tqdm
import numpy as np
import math
from transformers import SiglipProcessor, SiglipModel

def load_model(model_name, device='cuda'):
    """
    Load the specified model along with its processor.

    Supports CLIP, SigLip, and FLAVA models based on the model name.

    Args:
        model_name (str): The name or path of the model to load.
        device (str, optional): Device to load the model onto ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        dict: A dictionary containing the model, processor, and model_type.
    """
    model_name_lower = model_name.lower()
    if 'clip' in model_name_lower:
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name).to(device)
            model_type = 'clip'
        except Exception as e:
            raise ValueError(f"Failed to load CLIP model or processor: {e}")
    elif 'siglip' in model_name_lower:
        try:
            processor = SiglipProcessor.from_pretrained(model_name)
            model = SiglipModel.from_pretrained(model_name).to(device)
            model_type = 'siglip'
        except Exception as e:
            raise ValueError(f"Failed to load SigLip model or processor: {e}")
    elif 'flava' in model_name_lower:
        try:
            processor = FlavaProcessor.from_pretrained(model_name)
            model = FlavaModel.from_pretrained(model_name).to(device)
            image_processor = FlavaFeatureExtractor.from_pretrained(model_name)
            text_processor = BertTokenizer.from_pretrained("facebook/flava-full")
            model_type = 'flava'
            model.eval()
            return {'model': model, 'processor': processor, 'model_type': model_type, 'image_processor': image_processor, 'text_processor': text_processor}
        except Exception as e:
            raise ValueError(f"Failed to load FLAVA model or processor: {e}")
    else:
        raise ValueError(f"Unsupported model type in model_name: {model_name}")

    model.eval()  # Ensure the model is in evaluation mode
    return {'model': model, 'processor': processor, 'model_type': model_type}


def get_text_embeddings(captions, model_info, device, batch_size=64,):
    """
    Compute text embeddings for a list of captions using batch processing to manage GPU memory.

    Args:
        captions (List[str]): List of caption strings.
        processor: Processor compatible with the model.
        model: Pre-trained model with a get_text_features or equivalent method.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        batch_size (int, optional): Number of captions to process per batch. Default is 64.
        model_type (str, optional): Type of the model ('clip', 'siglip', 'flava'). Default is 'clip'.

    Returns:
        numpy.ndarray: Array of normalized embeddings with shape (len(captions), embedding_dim).
    """
    
    
    embeddings_list = []
    num_captions = len(captions)
    num_batches = math.ceil(num_captions / batch_size)

    processor = model_info['processor']
    model = model_info['model']
    model_type = model_info['model_type']
    model.to(device)
    model.eval()  # Ensure the model is in evaluation mode

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_captions)
            batch_captions = captions[start_idx:end_idx]

            # Use the processor to tokenize the batch of captions
            
            # important: we pass `padding=max_length` since the model was trained with this for SigLip
            if model_type == 'siglip':
                inputs = processor(
                    text=batch_captions,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
            elif model_type == 'flava':
                inputs = model_info['text_processor'](batch_captions, return_tensors='pt', padding='max_length', max_length=77, truncation=True).to(device)
            else:
                inputs = processor(
                    text=batch_captions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

            # Get text features from the model based on model_type
            if model_type in ['clip', 'siglip']:
                if hasattr(model, 'get_text_features'):
                    batch_embeddings = model.get_text_features(**inputs).detach()
                elif hasattr(model, 'encode_text'):
                    # For models that use a different method to get text embeddings
                    batch_embeddings = model.encode_text(**inputs).detach()
                else:
                    raise AttributeError("The model does not have a method to get text features.")
            elif model_type == 'flava':
                text_embedding = model.get_text_features(**inputs)
                batch_embeddings = text_embedding[:, 0]
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Normalize the embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)

            # Move embeddings to CPU and convert to numpy
            embeddings_list.append(batch_embeddings.cpu().numpy())

            # Optional: Print progress
            print(f"Processed batch {i + 1}/{num_batches}")

    # Concatenate all batch embeddings into a single numpy array
    embeddings = np.vstack(embeddings_list)
    return embeddings


def get_image_embeddings(dataloader, model_info, device,):
    """
    Compute image embeddings for a set of images using a dataloader.

    Args:
        dataloader: PyTorch DataLoader providing batches of images and labels.
        processor: Processor compatible with the model.
        model: Pre-trained model with a get_image_features or equivalent method.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        model_type (str, optional): Type of the model ('clip', 'siglip', 'flava'). Default is 'clip'.

    Returns:
        Tuple[numpy.ndarray, List]: A tuple containing the array of normalized embeddings and the corresponding labels.
    """
    
    processor = model_info['processor']
    model = model_info['model']
    model_type = model_info['model_type']
    
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataloader, desc="Processing Images"):
            # Use the processor to preprocess the images
            # Ensure all images are in RGB
            # if batch_images are not torch tensors, do the following:
            if not isinstance(batch_images[0], torch.Tensor):
                batch_images = [image.convert("RGB") for image in batch_images]
            
            if model_type != 'flava':
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(device)
            else:
                inputs = model_info['image_processor'](
                    images=batch_images,
                    return_tensors="pt"
                ).to(device)
            # Get image features from the model based on model_type
            if model_type in ['clip', 'siglip',]:
                if hasattr(model, 'get_image_features'):
                    image_features = model.get_image_features(**inputs).detach()
                elif hasattr(model, 'encode_image'):
                    # For models that use a different method to get image embeddings
                    image_features = model.encode_image(**inputs).detach()
                else:
                    raise AttributeError("The model does not have a method to get image features.")
            elif model_type == 'flava':
                # For FLAVA, use the forward method to get image embeddings
                image_embedding = model.get_image_features(**inputs).detach()
                image_features = image_embedding [:, 0]
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Normalize the embeddings
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Move embeddings to CPU and convert to numpy
            embeddings.append(image_features.cpu().numpy())
            labels.extend(batch_labels)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, labels
