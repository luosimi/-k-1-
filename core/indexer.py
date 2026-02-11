import os
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from core.feature_extractor import FeatureExtractor
from core.model_loader import ModelWrapper
from utils.file_utils import get_mtime, list_images
from utils.math_utils import serialize_embedding

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, root_dir, rel_paths, preprocess_fn):
        self.root_dir = root_dir
        self.rel_paths = rel_paths
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel_path = self.rel_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        try:
            with Image.open(full_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Return tuple: (processed_tensor, rel_path, valid)
                return self.preprocess(img), rel_path, True
        except Exception as e:
            # Return dummy tensor and False flag
            # We need to return a tensor of correct shape to stack
            # OpenCLIP preprocess usually returns (3, 224, 224)
            return torch.zeros((3, 224, 224)), rel_path, False

def collate_fn(batch):
    # batch is list of (tensor, rel_path, valid)
    # We filter out invalid ones? No, default_collate can't handle variable sizes if we drop.
    # So we keep them and filter after batching.
    tensors = []
    paths = []
    valids = []
    
    for t, p, v in batch:
        tensors.append(t)
        paths.append(p)
        valids.append(v)
        
    return torch.stack(tensors), paths, valids

class Indexer:
    def __init__(self, root_dir: str, csv_path: str):
        self.root_dir = root_dir
        self.csv_path = csv_path
        # We access ModelWrapper directly for batch processing
        self.model_wrapper = ModelWrapper()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def _load_existing_index(self):
        if not os.path.exists(self.csv_path):
            return {}
        try:
            df = pd.read_csv(self.csv_path)
            # Return dict: {filename: mtime}
            return dict(zip(df['filename'], df['mtime']))
        except Exception as e:
            logger.error(f"Failed to load index csv: {e}")
            return {}

    def sync(self, progress_callback=None):
        logger.info(f"Starting sync for {self.root_dir}...")
        
        # 1. Scan current files
        current_files = {}
        for filepath in list_images(self.root_dir):
            try:
                rel_path = os.path.relpath(filepath, self.root_dir)
                current_files[rel_path] = get_mtime(filepath)
            except Exception as e:
                logger.warning(f"Skipping file {filepath}: {e}")

        # 2. Load existing index
        indexed_files = self._load_existing_index()

        # 3. Calculate diff
        to_add = [] 
        to_update = [] 
        to_remove = [] 

        for path, mtime in current_files.items():
            if path not in indexed_files:
                to_add.append(path)
            elif indexed_files[path] != mtime:
                to_update.append(path)
        
        for path in indexed_files:
            if path not in current_files:
                to_remove.append(path)

        logger.info(f"Sync status: Add {len(to_add)}, Update {len(to_update)}, Remove {len(to_remove)}")

        # 4. Handle removals and clean up DF
        df = pd.DataFrame(columns=['filename', 'mtime', 'embedding'])
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                # Remove deleted AND updated files (we will re-add updated ones)
                files_to_drop = set(to_remove + to_update)
                if files_to_drop:
                    df = df[~df['filename'].isin(files_to_drop)]
            except Exception as e:
                logger.error(f"Error reading CSV, creating new: {e}")
                df = pd.DataFrame(columns=['filename', 'mtime', 'embedding'])
        
        # 5. Process new/updated files with DataLoader
        process_list = to_add + to_update
        if not process_list:
            if to_remove: # If only removals, save and exit
                 df.to_csv(self.csv_path, index=False)
            logger.info("Nothing to process.")
            return

        batch_size = 32
        # On Windows, num_workers > 0 can be problematic in some envs, 
        # but 4 is usually safe if entry point is guarded. 
        # Since this runs in a thread, let's use 0 to be 100% safe against spawn issues for now,
        # OR 2 if we are confident.
        # Given user asked about "multi-process", let's try to enable it.
        # But for stability in this specific context (Flask Thread + Windows), 0 is safer.
        # However, BATCHING on GPU is the main speedup.
        num_workers = 0 
        
        dataset = ImageDataset(self.root_dir, process_list, self.model_wrapper.preprocess)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        new_records = []
        total = len(process_list)
        processed_count = 0

        for batch_tensors, batch_paths, batch_valids in tqdm(dataloader, desc="Processing batches"):
            # Filter valid images
            valid_indices = [i for i, v in enumerate(batch_valids) if v]
            if not valid_indices:
                processed_count += len(batch_paths)
                if progress_callback:
                    progress_callback(processed_count, total)
                continue

            # Select valid tensors
            valid_tensors = batch_tensors[valid_indices]
            
            # Batch Inference
            try:
                embeddings = self.model_wrapper.encode_batch(valid_tensors)
                
                # Map back to paths
                for i, idx in enumerate(valid_indices):
                    rel_path = batch_paths[idx]
                    emb = embeddings[i]
                    
                    new_records.append({
                        'filename': rel_path,
                        'mtime': current_files[rel_path],
                        'embedding': serialize_embedding(emb)
                    })
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
            
            processed_count += len(batch_paths)
            if progress_callback:
                progress_callback(processed_count, total)
            
            # Periodic Save
            if len(new_records) >= 100:
                temp_df = pd.DataFrame(new_records)
                df = pd.concat([df, temp_df], ignore_index=True)
                df.to_csv(self.csv_path, index=False)
                new_records = []

        # Final Save
        if new_records:
            temp_df = pd.DataFrame(new_records)
            df = pd.concat([df, temp_df], ignore_index=True)
            df.to_csv(self.csv_path, index=False)

        logger.info("Sync complete.")
