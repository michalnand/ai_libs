import numpy
import AILibs

class RUGDDataset:

    '''
        positive_indices = []
        positive_indices.append([64, 64, 64])   # asphalt
        positive_indices.append([255, 128, 0])  # gravel
    '''
    def __init__(self, root_path, part_name, positive_indices, size = None):
        self.positive_indices = positive_indices


        self.images   = AILibs.ImagesLoader(root_path + "/" + "RUGD_frames-with-annotations/"  + part_name, size = size)
        self.seg_mask = AILibs.ImagesLoader(root_path + "/" + "RUGD_annotations/" + part_name, size = size, keep_uint8=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        x = self.images[idx]
        
        y = self.seg_mask[idx]
        y = self._create_binary_mask(y, self.positive_indices)

        return x, y
    

    def _create_binary_mask(self, y, positive_indices):
        # 1. Transpose y from (3, H, W) to (H, W, 3) to make pixel-wise comparison easier
        y_transposed = numpy.transpose(y, (1, 2, 0))
        
        # 2. Initialize a blank 2D boolean mask of shape (height, width)
        combined_mask = numpy.zeros(y_transposed.shape[:2], dtype=bool)
        
        # 3. Iterate through each target color and find matching pixels
        for color in positive_indices:
            # numpy.all(..., axis=-1) checks if R, G, and B all match simultaneously
            color_match = numpy.all(y_transposed == color, axis=-1)
            # Combine with previous matches using logical OR
            combined_mask = numpy.logical_or(combined_mask, color_match)
            
        # 4. Convert the boolean mask (True/False) to float 1/0
        return combined_mask.astype(int)
    
    def _get_unique_colors(self, y):
        # 1. Transpose y from (3, H, W) to (H, W, 3)
        y_transposed = numpy.transpose(y, (1, 2, 0))
        
        # 2. Flatten the height and width into a single dimension of pixels
        # This reshapes the array to (H * W, 3)
        pixels = y_transposed.reshape(-1, 3)
        
        # 3. Find unique rows (unique RGB combinations)
        unique_colors = numpy.unique(pixels, axis=0)
        
        # 4. Convert the NumPy array to a standard Python list of lists
        return unique_colors.tolist()
    