import AILibs

class ConfigA:

    def __init__(self):

        #self.dataset_root_path = "/Users/michal/datasets/textures/"
        self.dataset_root_path = "/home/michal/datasets/unlabeled_images/"
        self.dataset = AILibs.DatasetImages(self.dataset_root_path)
        
        self.num_steps  = 100000
        self.batch_size = 128

        self.width      = 256   
        self.height     = 256 
        
        self.num_points = 32
        self.model      = AILibs.TinyCNNModel(in_ch = 3, num_features = 128)

        self.learning_rate = 0.001
        self.w_sim         = 1.0
        self.w_ssl         = 1.0        
        
        self.result_path = "results/images_features/"


class ConfigB:

    def __init__(self):

        #self.dataset_root_path = "/Users/michal/datasets/textures/"
        self.dataset_root_path = "/home/michal/datasets/unlabeled_images/"
        self.dataset = AILibs.DatasetImages(self.dataset_root_path)
        
        self.num_steps  = 100000
        self.batch_size = 128

        self.width      = 256   
        self.height     = 256 
        
        self.num_points = 32
        self.model      = AILibs.TinyCNNModel(in_ch = 3, num_features = 128)

        self.learning_rate = 0.001
        self.w_sim         = 1.0
        self.w_ssl         = 10.0        
        
        self.result_path = "results/images_features_b/"

if __name__ == "__main__":
    
    #config = ConfigA()
    config = ConfigB()

    pipeline = AILibs.ImageFeaturesPipeline(config)

    pipeline.run_training()
