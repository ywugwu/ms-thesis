# utils/config.py
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    # Add validation logic if necessary
    return config
