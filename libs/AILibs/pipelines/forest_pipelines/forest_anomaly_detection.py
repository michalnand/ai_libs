import AILibs
import numpy
import os

from .forest_utils import *

class ForestAnomalyDetection:
    

    def __init__(self, config): 

        self.config = config

        self.dataset_train = config.dataset_train
        self.dataset_test  = config.dataset_test

        self.batch_size                 = config.batch_size
        self.num_trees                  = config.num_trees
        self.min_leaf_size              = config.min_leaf_size
        self.feature_subsample_ratio    = config.feature_subsample_ratio
        self.projection_dim             = -1

        self.threshold                  = config.threshold

        
        self.forest = AILibs.LargeScaleIsolationForest(batch_size=self.batch_size, num_trees=self.num_trees, min_leaf_size=self.min_leaf_size, feature_subsample_ratio=self.feature_subsample_ratio, projection_dim=self.projection_dim)


        self.result_path = config.result_path


    def run(self):
        self._training()
        m, y_gt, y_pred = self._testing()

        self._save(m, y_gt, y_pred) 

    def _training(self):
        self.forest.fit(self.dataset_train, verbose=True)  
 
    def _testing(self):
        y_pred = []

        x, y_gt = self.dataset_test

        for n in range(len(x)):
            y_hat = self.forest.score(x[n])

            y_pred.append(y_hat)
            
        y_gt   = numpy.array(y_gt) 
        y_pred = numpy.array(y_pred)
        
        y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())

        if self.threshold is None:
            self.threshold = AILibs.metrics.tune_threshold(y_gt, y_pred, metric="f1")
            print("autotune threshold to ", self.threshold)

        metrics = AILibs.metrics.anomaly_evaluation(y_gt, y_pred, th=self.threshold)
        return metrics, y_gt, y_pred


    def predict(self, x):
        return self.forest.score(x)


    def _save(self, metrics, y_gt, y_pred):

        # 1. Create result directory if it doesn't exist
        os.makedirs(self.result_path, exist_ok=True)
        
        # 2. Save the model
        model_file = os.path.join(self.result_path, "forest_model.json")
        self.forest.save(model_file)



        # 3, save plots and results
        forest_save_detector_prediction_distribution(self.result_path, y_gt, y_pred, threshold = self.threshold)
        forest_save_detector_cm(self.result_path, metrics)
        forest_save_anomaly_detection_metrics(self.result_path, self.config, metrics)


