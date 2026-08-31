import AILibs
import numpy
    


class ForestAnomalyDetectionConfig:

    def __init__(self):

        file_name = "/Users/michal/datasets/creditcard/creditcard.csv"
        self.dataset   = AILibs.CSVDataset(file_name)
    
        print("Dataset shape:", self.dataset.x.shape)
        print("Total samples:", self.dataset.x.shape[0])

            
        self.batch_size                 = 16384
        self.num_trees                  = 256
        self.min_leaf_size              = 4 
        self.feature_subsample_ratio    = 0.5 
        self.projection_dim             = -1
    
        self.threshold                  = None
        self.result_path                = "./results/forest_anomaly_detection/"

        self.dataset_train, self.dataset_test = self._split()
        

    def _split(self):
        # train / test split
    
        test_ratio = 0.2
        indices    = numpy.arange(self.dataset.x.shape[0])
        numpy.random.shuffle(indices)
    
        split_idx     = int((1 - test_ratio) * self.dataset.x.shape[0])
        train_indices = indices[:split_idx]
        test_indices  = indices[split_idx:]
    
        # Features (all columns except the last)
        x_train = self.dataset.x[train_indices, :-1]
        x_test  = self.dataset.x[test_indices,  :-1]
    
        # Labels: binarise (> 0.5 → 1.0 = fraud)
        y_train = (self.dataset.x[train_indices, -1] > 0.5) * 1.0
        y_test  = (self.dataset.x[test_indices,  -1] > 0.5) * 1.0
    
        print(f"Train : {x_train.shape},  fraud rate = {y_train.mean():.4f}")
        print(f"Test  : {x_test.shape},  fraud rate = {y_test.mean():.4f}")

        return (x_train), (x_test, y_test)

        
if __name__ == "__main__":

    config = ForestAnomalyDetectionConfig()

    pipeline = AILibs.ForestAnomalyDetection(config)
 
    pipeline.run()      