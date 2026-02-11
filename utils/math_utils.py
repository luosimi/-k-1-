import base64
import numpy as np

def serialize_embedding(embedding: np.ndarray) -> str:
    """
    Serialize a float32 numpy array to a base64 string.
    """
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return base64.b64encode(embedding.tobytes()).decode('utf-8')

def deserialize_embedding(base64_str: str) -> np.ndarray:
    """
    Deserialize a base64 string to a float32 numpy array.
    """
    return np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)
