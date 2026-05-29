from .rugd_dataset import *
from .segmentation_augmentation      import *
from .deeplab_mobile_net      import *


class DeepLabConfig:


    def __init__(self):

        dataset_root_path = "/home/michal/datasets/rugd/"
        self.results_path = "./results/deeplab_mobile_net/"
        self.device       = "cuda"

        self.image_size = (640, 480)

        self.training_dataset = self._create_daatset(dataset_root_path)
        self.testing_datset   = None


        self.learning_rate = 0.001
        self.batch_size    = 16

        self.num_epoch     = 20
        self.num_steps     = (self.num_epoch*len(self.training_dataset))//self.batch_size


        self.model = DeepLabMobileNet(3, 1)
        self.model.to(self.device)  

        self.augmentations = SegmentationAugmentation()


        print("total training images ", len(self.training_dataset))
        print("learning_rate         ", self.learning_rate)
        print("batch_size            ", self.batch_size)
        print("num_epoch             ", self.num_epoch)
        print("num_steps             ", self.num_steps)
        print("\n\n")
        print(self.model)
        print("\n\n")
        
        
    def _create_dataset(self, dataset_root_path):
        positive_indices = []
        positive_indices.append([64, 64, 64])   # asphalt
        positive_indices.append([11, 101, 101]) # concrete
        positive_indices.append([0, 128, 255])  # gravel
        
            
        # load multiple separeted sources
        datasets_parts = []
        datasets_parts.append([dataset_root_path, "creek/", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "park-1", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "park-2", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "park-8", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-3", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-4", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-5", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-6", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-7", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-9", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-10", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-11", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-12", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-13", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-14", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "trail-15", positive_indices, self.image_size])
        datasets_parts.append([dataset_root_path, "village", positive_indices, self.image_size])



        datasets_all = []
        for n in range(len(datasets_parts)):
            cfg = datasets_parts[n]
            dataset = RUGDDataset(cfg[0], cfg[1], cfg[2], cfg[3])
            datasets_all.append(dataset)

        # agregate into one dataset
        main_dataset = AILibs.DatasetCollator(datasets_all)
        return main_dataset