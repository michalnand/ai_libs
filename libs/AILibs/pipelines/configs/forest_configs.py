
class ForestDetectionConfig:

    def __init__(self):

        self.batch_size                 = 16384
        self.num_trees                  = 128
        self.learning_rate              = 0.25
        self.max_depth                  = 12
        self.min_leaf_size              = 8
        self.feature_subsample_ratio    = 1.0

        self.threshold                  = 0.5
        self.result_path                = "./result/"

        self.dataset_train = None
        self.dataset_test = None
