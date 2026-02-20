import  numpy 
import csv

class TabularSplit:

    def __init__(self, dataset, x_indices, y_indices):
        self.dataset = dataset
        self.x_indices = x_indices
        self.y_indices = y_indices

       
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tmp = self.dataset[idx]
        return  tmp[self.x_indices], tmp[self.y_indices]