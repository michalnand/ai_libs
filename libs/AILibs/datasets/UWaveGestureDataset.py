import os
import numpy
from scipy.io import arff

class UWaveGestureDataset():
    def __init__(self, data_dir, split="TRAIN"):
        """
        Args:
            data_dir (str): Path to the directory containing the .arff files.
            split (str): "TRAIN" or "TEST" depending on the split you want to load.
        """
        self.data_dir = data_dir
        self.split = split.upper()
        
        # Define the file paths for the 3 dimensions
        dim1_path = os.path.join(data_dir, f"UWaveGestureLibraryDimension1_{self.split}.arff")
        dim2_path = os.path.join(data_dir, f"UWaveGestureLibraryDimension2_{self.split}.arff")
        dim3_path = os.path.join(data_dir, f"UWaveGestureLibraryDimension3_{self.split}.arff")
        
        # Load and parse the ARFF files
        print(f"Loading {self.split} dimensions...")
        data_d1, meta_d1 = arff.loadarff(dim1_path)
        data_d2, meta_d2 = arff.loadarff(dim2_path)
        data_d3, meta_d3 = arff.loadarff(dim3_path)
        
        # Convert structured numpy arrays to standard float arrays
        # ARFF files from UWave pack the sequence into columns, with the last column being the class label
        
        # 1. Extract features (all columns except the last one)
        # scipy loads these as structured arrays, so we convert them to regular 2D float arrays
        d1_features = numpy.array([list(row)[:-1] for row in data_d1], dtype=numpy.float32)
        d2_features = numpy.array([list(row)[:-1] for row in data_d2], dtype=numpy.float32)
        d3_features = numpy.array([list(row)[:-1] for row in data_d3], dtype=numpy.float32)
        
        # Stack dimensions along a new axis to get shape: (num_samples, seq_length, 3)
        self.features = numpy.stack([d1_features, d2_features, d3_features], axis=-1)
        
        # 2. Extract labels from the last column of the first dimension 
        # (Labels are identical across all dimension files)
        
        raw_labels = numpy.array([row[-1] for row in data_d1])

        
        
        # ARFF often stores labels as byte strings (e.g., b'1', b'2'). 
        # We decode them and convert to 0-indexed integers (0 to 7) for PyTorch compatibility.
        if isinstance(raw_labels[0], bytes):
            raw_labels = numpy.array([lbl.decode('utf-8') for lbl in raw_labels])
            
        # Convert string labels to integers and map to 0-7 range
        #self.labels = raw_labels.astype(numpy.int64) - 1 

        self.labels = raw_labels.astype(numpy.float32).astype(numpy.int64) - 1


        self.input_shape = self.features.shape[1:]  # (seq_length, num_features)
        self.num_classes = len(numpy.unique(self.labels))
        
        print(f"Loaded {self.features.shape[0]} samples.")
        print(f"Feature shape per sample: {self.input_shape} (seq_length, num_features)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            x (np.ndarray): Shape (seq_length, 3)
            y (int): Class label (0 to 7)
        """
        x = self.features[idx]
        y = self.labels[idx]
        return x, y