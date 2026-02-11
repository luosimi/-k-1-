import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from utils.math_utils import deserialize_embedding

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        self.features = None
        self.paths = None
        self.filenames = None

    def load_index(self, csv_path: str):
        logger.info(f"Loading index from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning("Index is empty.")
                self.features = np.empty((0, 1024), dtype=np.float32)
                self.filenames = []
                return

            # Decode embeddings
            # Apply deserialize to the column. This might be slow for huge CSVs, but okay for prototype.
            embeddings = df['embedding'].apply(deserialize_embedding)
            
            # Stack into a matrix
            self.features = np.vstack(embeddings.values)
            self.filenames = df['filename'].tolist()
            
            # Normalize features (just in case)
            norms = np.linalg.norm(self.features, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-10
            self.features = self.features / norms
            
            logger.info(f"Index loaded. {len(self.filenames)} images.")
        except FileNotFoundError:
            logger.warning("Index file not found.")
            self.features = np.empty((0, 1024), dtype=np.float32)
            self.filenames = []
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.features = np.empty((0, 1024), dtype=np.float32)
            self.filenames = []

    def search(self, query_vec: np.ndarray, top_k=50) -> List[Dict]:
        if self.features is None or len(self.features) == 0:
            return []

        # Ensure query_vec is normalized
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        
        # Dot product
        scores = self.features @ query_vec
        
        # Get top k
        # If top_k is larger than n, clip it
        k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'path': self.filenames[idx],
                'score': float(scores[idx])
            })
            
        return results
