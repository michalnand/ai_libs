import os
from .images_loader    import *
from .dataset_collator import *

class DatasetImages:
    def __init__(self, root_path):
        dirs = os.listdir(root_path)
        print(dirs)

        self.images_datasets = []
        for dir in dirs:
            path         = root_path + "/" + dir + "/"
            self.images_datasets.append(ImagesLoader(path))

        self.dataset = DatasetCollator(self.images_datasets)

        
    def get(self, idx):
        return self.dataset[idx]

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __len__(self):
        return len(self.dataset)

    def get_batch(self, batch_size):
        result = []
        for n in range(batch_size):
            idx = numpy.random.randint(0, len(self.dataset))
            x   = self.get(idx)
            result.append(x)

        return result