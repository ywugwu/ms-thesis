from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class Model:
    def __init__(self, model='clip-vit-base-patch16'):
        self.model = CLIPModel.from_pretrained(f"openai/{model}")
        self.processor = CLIPProcessor.from_pretrained(f"openai/{model}")
    
    def encode_text(self, text):
        """
        Args:
            text (_type_): a list of strings
        """
        text_tensors = self.processor(text, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**text_tensors)
        return text_features
        