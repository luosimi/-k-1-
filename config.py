import os
import json

# Default configuration
DEFAULT_CONFIG = {
    "gallery_path": "E:\\MyPhotos",
    "model_name": "ViT-H-14",
    "pretrained": "laion2b_s32b_b79k",
    "device": "cuda", # will be checked in code
    "index_path": "./data/index.csv"
}

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# Global config instance
config = load_config()
