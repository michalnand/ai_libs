import os
import numpy
import threading
from concurrent.futures import ThreadPoolExecutor
from .images_loader import *
from .dataset_collator import *

class DatasetImages:
    def __init__(self, root_path, num_workers=4):
        dirs = os.listdir(root_path)       
        print(dirs)

        self.images_datasets = []
        # Minor fix: used os.path.join for safer cross-platform path building
        for d in dirs:
            path = os.path.join(root_path, d)
            self.images_datasets.append(ImagesLoader(path))

        self.dataset = DatasetCollator(self.images_datasets)
        
        # 1. Encapsulate the thread pool inside the class
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 2. Setup thread-local storage for independent Random Generators
        self.thread_local = threading.local()

    def _get_thread_rng(self):
        """Ensures each thread gets its own isolated NumPy random generator."""
        if not hasattr(self.thread_local, 'rng'):
            # default_rng() pulls fresh entropy from the OS, guaranteeing unique sequences
            self.thread_local.rng = numpy.random.default_rng()
        return self.thread_local.rng    

    def get(self, idx):
        return self.dataset[idx]

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __len__(self):
        return len(self.dataset)

    def _fetch_random_item(self, _):
        """Internal worker function executed by the thread pool."""
        rng = self._get_thread_rng()
        # rng.integers is the modern, faster equivalent of numpy.random.randint
        idx = rng.integers(0, len(self.dataset))
        return self.get(idx)

    def get_batch(self, batch_size):
        # 3. Map the fetching function across the persistent thread pool.
        # We pass range(batch_size) simply to trigger the function N times.
        result = list(self.executor.map(self._fetch_random_item, range(batch_size)))
        return result

    def __del__(self):
        # 4. Clean up threads automatically when the Dataset object is garbage collected
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)