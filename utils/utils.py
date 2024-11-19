import os
import json
import re
from pydantic import BaseModel
from typing import List
from openai import OpenAI  # Ensure this aligns with your OpenAI wrapper

import matplotlib.pyplot as plt
import os
import numpy as np

# Define the Pydantic models for the API responses
class CaptionResponse(BaseModel):
    captions: List[str]

class TraitsResponse(BaseModel):
    traits: List[str]

# Define the CaptionGenerator class
class CaptionGenerator:
    def __init__(self, dataset_name: str, class_names: List[str], model: str = "gpt-4o-mini-2024-07-18", num_captions: int = 32, cache_dir: str = "cache",prompt_template: list = ['a photo of a {c}']):
        """
        Initializes the CaptionGenerator with the specified OpenAI model, number of captions, and cache directory.
        
        Args:
            dataset_name (str): The name of the dataset.
            class_names (List[str]): The list of class names in the dataset.
            model (str): The name of the OpenAI model to use.
            num_captions (int): The number of captions to generate.
            cache_dir (str): The directory where cached responses are stored.
        """
        self.client = OpenAI()
        self.model = model
        self.num_captions = num_captions
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.global_traits = None  # Will be set after generating traits
        self.prompt_template = prompt_template
        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Generate or load global traits
        self.global_traits = self.get_global_traits()
        
        self.meta_prompt = (
            f"You are an AI assistant that generates creative and diverse image captions "
            f"suitable for use with image generation models like DALL-E. Given a subject, "
            f"provide {self.num_captions} distinct, diverse and descriptive captions, "
            f"considering the following global taxonomical traits when generating captions: {self.global_traits}."
        )
        print("Using OpenAI model:", model)
        print(f"Configured to generate {self.num_captions} captions.")
        print(f"Meta Prompt: {self.meta_prompt}")

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitizes a string to be safe for use as a filename.
        """
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Remove invalid characters
        name = re.sub(r'[\\/:"*?<>|]+', '', name)
        # Optionally, truncate the name to a reasonable length
        return name[:100]  # Adjust the length as needed

    def _generate_cache_path(self, subject: str) -> str:
        """
        Constructs the file path for a given subject based on the model, dataset, and number of captions.
        """
        # Sanitize each part to ensure valid filenames
        sanitized_model = self._sanitize_filename(self.model)
        sanitized_dataset = self._sanitize_filename(self.dataset_name)
        sanitized_subject = self._sanitize_filename(subject)
        sanitized_num = str(self.num_captions)
        
        # Create a subdirectory structure: cache/model/dataset/num_captions/
        model_dir = os.path.join(self.cache_dir, sanitized_model)
        dataset_dir = os.path.join(model_dir, sanitized_dataset)
        num_dir = os.path.join(dataset_dir, sanitized_num)
        os.makedirs(num_dir, exist_ok=True)
        
        # Define the cache file path
        cache_filename = f"{sanitized_subject}.json"
        return os.path.join(num_dir, cache_filename)


    def _generate_global_traits_cache_path(self) -> str:
        """
        Constructs the file path for caching the global traits based on the dataset name.
        """
        sanitized_model = self._sanitize_filename(self.model)
        sanitized_dataset = self._sanitize_filename(self.dataset_name)
        
        # Create a subdirectory for the model
        model_dir = os.path.join(self.cache_dir, sanitized_model)
        os.makedirs(model_dir, exist_ok=True)

        # Define the cache file path
        cache_filename = f"{sanitized_dataset}_global_traits.json"
        return os.path.join(model_dir, cache_filename)

    def _load_from_cache(self, cache_path: str) -> dict:
        """
        Attempts to load data from the cache.
        """
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded data from cache: {cache_path}")
                return data
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")
        return None

    def _save_to_cache(self, cache_path: str, data: dict):
        """
        Saves data to the cache.
        """
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Saved data to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache to {cache_path}: {e}")

    def get_global_traits(self) -> List[str]:
        """
        Generates global taxonomical traits for the dataset.
        """
        cache_path = self._generate_global_traits_cache_path()
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data.get('traits', [])

        # Define the prompt for generating global traits
        class_names_str = ', '.join(self.class_names)
        user_content = (
            f"The dataset '{self.dataset_name}' contains the following classes: {class_names_str}.\n"
            "Identify and list the key global taxonomical traits and characteristics "
            "that are most visible in your imagination. "
            "Provide a concise list of traits."
        )

        # Construct the messages list
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant that understands taxonomical traits and classifications."},
            {"role": "user", "content": user_content},
        ]

        # Call the OpenAI API with the specified model and response format
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=TraitsResponse,
            )
            # Extract the parsed response
            traits_response = completion.choices[0].message.parsed
            # Save to cache
            self._save_to_cache(cache_path, {'traits': traits_response.traits})
            return traits_response.traits
        except Exception as e:
            print(f"An error occurred while generating global traits: {e}")
            return []

    def get_alternative_captions(self, subject: str) -> List[str]:
        """
        Generates alternative captions for a given subject prompt, utilizing cache when possible.
        """
        cache_path = self._generate_cache_path(subject)
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data.get('captions', [])

        # Define the meta-prompt with the subject and global traits
        # traits_str = ', '.join(self.global_traits)
        user_content = (
            f"Please generate {self.num_captions} diverse and creative alternative captions for the subject '{subject}'. "
            f"Each caption should be compatible with the CLIP model and adhere to the original prompt template provided: '{self.prompt_template}'. "
            f"Ensure that the captions maintain the structure and format of the template, appropriately replacing any placeholders, while introducing variety in wording and expression."
        )

        # Construct the messages list
        messages = [
            {"role": "system", "content": self.meta_prompt},
            {"role": "user", "content": user_content},
        ]

        # Call the OpenAI API with the specified model and response format
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=CaptionResponse,
            )
            # Extract the parsed response
            caption_response = completion.choices[0].message.parsed
            # Save to cache
            self._save_to_cache(cache_path, {'captions': caption_response.captions})
            return caption_response.captions
        except Exception as e:
            print(f"An error occurred while generating captions: {e}")
            return []

def plot_consistency_scores(consistency_results, class_names, dataset_name, save_path):
    """
    Plots the consistency scores per class.

    Args:
        consistency_results (dict): Dictionary with score types as keys and lists of scores as values.
        class_names (list): List of class names corresponding to the scores.
        dataset_name (str): Name of the dataset for labeling the plot.
        save_path (str): Directory path to save the plot.
    """
    # Extract scores
    type_1_scores = consistency_results['type_1_text_consistency_score']
    type_2_scores = consistency_results['type_2_text_consistency_score']
    type_3_scores = consistency_results['type_3_text_consistency_score']

    # Ensure the length of scores matches the number of classes
    assert len(type_1_scores) == len(class_names), "Mismatch between number of classes and Type 1 scores."
    assert len(type_2_scores) == len(class_names), "Mismatch between number of classes and Type 2 scores."
    assert len(type_3_scores) == len(class_names), "Mismatch between number of classes and Type 3 scores."

    x = np.arange(len(class_names))
    width = 0.25  # Width of the bars

    plt.figure(figsize=(15, 8))
    plt.bar(x - width, type_1_scores, width, label='Type 1')
    plt.bar(x, type_2_scores, width, label='Type 2')
    plt.bar(x + width, type_3_scores, width, label='Type 3')

    plt.xlabel('Classes')
    plt.ylabel('Consistency Scores')
    plt.title(f'Consistency Scores per Class for {dataset_name}')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()

    plt.tight_layout()
    plot_filename = f"{dataset_name}_consistency_scores.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()
    print(f"Consistency scores plot saved to {os.path.join(save_path, plot_filename)}")

# Example usage
if __name__ == "__main__":
    # Initialize the caption generator with dataset information and number of captions
    dataset_name = "Stanford_dogs"
    class_names = ["Labrador Retriever", "German Shepherd", "Poodle", "Beagle", "Rottweiler"]

    caption_generator = CaptionGenerator(dataset_name=dataset_name, class_names=class_names, num_captions=12)
    # Define the subject prompt
    subject = "Golden Retriever"
    
    # Get alternative captions
    captions = caption_generator.get_alternative_captions(subject)
    
    # Print the captions
    for idx, caption in enumerate(captions, start=1):
        print(f"Caption {idx}: {caption}")
