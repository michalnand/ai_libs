import numpy


class DatasetCollator:

    """
    DatasetCollator is a utility class that combines multiple datasets into a single dataset. 
    It allows you to treat multiple datasets as one cohesive unit, making it easier to manage and access data from different sources.
    """

    def __init__(self, datasets):
        self.datasets = datasets

        self.length = 0

        self.lengths = []
        for n in range(len(self.datasets)):
            self.lengths.append(len(self.datasets[n]))
        
        self.length = numpy.sum(self.lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        offset = idx
        for dataset_id, dataset_len in enumerate(self.lengths):
            if offset < dataset_len:
                return self.datasets[dataset_id][offset]
            offset -= dataset_len
        
        return None