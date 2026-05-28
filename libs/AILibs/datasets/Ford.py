import os
import numpy
from scipy.io import arff

class FordDataset():
    def __init__(self, data_dir, split="TRAIN"):
        """
        Args:
            data_dir (str): Path to the directory containing the Ford files.
            split (str): "TRAIN" or "TEST" depending on the split you want to load.
        """
        self.data_dir = data_dir
        self.split = split.upper()
        
        # Define the file path for the single ARFF file
        file_path = os.path.join(data_dir, f"FordA_{self.split}.arff")
        
        # Load and parse the ARFF file
        print(f"Loading FordA {self.split} split...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find the dataset file at {file_path}")
            
        data, meta = arff.loadarff(file_path)
        
        # Convert structured numpy arrays to standard float arrays
        # Ford packs the sequence into columns, with the last column being the class label
        
        # 1. Extract features (all columns except the last one)
        # scipy loads these as structured arrays, so we convert them to regular 2D float arrays
        raw_features = numpy.array([list(row)[:-1] for row in data], dtype=numpy.float32)
        
        # Expand dimensions to match (num_samples, seq_length, num_features) structure
        # Ford is univariate, so num_features = 1
        self.features = numpy.expand_dims(raw_features, axis=-1)
        
        # 2. Extract labels from the last column
        raw_labels = numpy.array([row[-1] for row in data])
        raw_labels = raw_labels.astype(numpy.float32).astype(numpy.int64)
        
            
        self.labels = (raw_labels > 0).astype(numpy.int64)

        self.input_shape = self.features.shape[1:]  # (seq_length, 1)
        self.num_classes = len(numpy.unique(self.labels))
        
        print(f"Loaded {self.features.shape[0]} samples.")
        print(f"Feature shape per sample: {self.input_shape} (seq_length, num_features)")
        print(f"Number of unique classes found: {self.num_classes}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            x (np.ndarray): Shape (seq_length, 1)
            y (int): Class label (0 to 4)
        """
        x = self.features[idx]
        y = self.labels[idx]
        return x, y