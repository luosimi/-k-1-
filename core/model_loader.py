import open_clip
import torch
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelWrapper, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_name='ViT-H-14', pretrained='laion2b_s32b_b79k', device=None):
        if self.initialized:
            return
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Explicitly log device capability for user transparency
        if self.device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            logger.info(f"✅ GPU Detected: {props.name} (VRAM: {props.total_memory / 1024**3:.2f} GB)")
        else:
            logger.warning("⚠️ No GPU detected. Running on CPU (this will be slower).")
            
        logger.info(f"Loading model {model_name} on {self.device}...")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.model.eval()
            self.initialized = True
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def get_device_info(self):
        """
        Return current device information.
        """
        info = {
            "device": self.device,
            "name": "CPU"
        }
        if self.device == 'cuda':
            try:
                info["name"] = torch.cuda.get_device_name(0)
                info["vram"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            except:
                info["name"] = "Unknown GPU"
        return info

    def encode(self, image_input) -> np.ndarray:
        """
        Encode an image (PIL Image or Tensor) into a normalized numpy vector.
        """
        # Ensure input is a batch
        if isinstance(image_input, Image.Image):
            image = self.preprocess(image_input).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            image = image_input.to(self.device)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        else:
            raise ValueError("Unsupported image input type")

        # Use generic amp.autocast which supports both cuda and cpu (if backend available)
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        with torch.no_grad(), torch.amp.autocast(device_type=device_type):
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def encode_batch(self, image_tensor) -> np.ndarray:
        """
        Encode a batch of images (Tensor) into normalized numpy vectors.
        image_tensor shape: (B, C, H, W)
        """
        image_tensor = image_tensor.to(self.device)
        
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        with torch.no_grad(), torch.amp.autocast(device_type=device_type):
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()
