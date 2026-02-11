import os
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

def get_mtime(filepath: str) -> float:
    """
    Get file modification time.
    """
    return os.path.getmtime(filepath)

def is_valid_image_ext(filepath: str) -> bool:
    """
    Check if file has a valid image extension.
    """
    ext = os.path.splitext(filepath)[1].lower()
    return ext in IMAGE_EXTENSIONS

def is_valid_image_file(filepath: str) -> bool:
    """
    Check if file is a valid image by attempting to open it.
    """
    if not is_valid_image_ext(filepath):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def list_images(root_dir: str):
    """
    Generator that yields all valid image paths in a directory.
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if is_valid_image_ext(filepath):
                yield filepath
