import  numpy 

class WindowDataset:

    """
        A dataset wrapper that creates sliding windows over the input dataset.
        This is useful for time series data where we want to create sequences of a fixed length.
    """

    def __init__(self, dataset, window_size):

        x, y = dataset[0]

        size = len(dataset) - window_size

        num_x_features = x.shape[-1]
        num_y_features = y.shape[-1]


        self.x_result = numpy.zeros((size, window_size, num_x_features), dtype=numpy.float32)
        self.y_result = numpy.zeros((size, window_size, num_y_features), dtype=numpy.float32)

        for n in range(size):
            for t in range(window_size):
                x, y = dataset[n + t]
                self.x_result[n, t] = x
                self.y_result[n, t] = y
        
      

    def __len__(self):
        return self.x_result.shape[0]

    def __getitem__(self, idx):
        return self.x_result[idx], self.y_result[idx]