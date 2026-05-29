import numpy
import AILibs

class SegmentationDetectionDataset:

    def __init__(self, root_path, images_name, mask_name, mask_filter, positive_indices, size = None):
        self.positive_indices = positive_indices
        self.images   = AILibs.ImagesLoader(root_path + "/" + images_name + "/", size = size)
        self.seg_mask = AILibs.ImagesLoader(root_path + "/" + mask_name + "/", size = size, name_filter =mask_filter)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        positive_indices = self.positive_indices[0]
        x = self.images[idx]
        y = numpy.array(self.seg_mask[idx]*255, dtype=int)
        y = (y == positive_indices).astype(dtype=int)[0, :, :]

        return x, y
 