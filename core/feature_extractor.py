from PIL import Image
from core.model_loader import ModelWrapper
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.model = ModelWrapper()

    def extract_from_path(self, image_path):
        """
        Open image from path and return feature vector.
        Returns None if image cannot be opened.
        """
        try:
            with Image.open(image_path) as img:
                # Ensure image is RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.model.encode(img)
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None

    def extract_from_image(self, image: Image.Image):
        """
        Extract feature vector from PIL Image object.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.model.encode(image)
