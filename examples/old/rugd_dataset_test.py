import image_libs

import cv2

import numpy

if __name__ == "__main__":

 
    images_loader      = image_libs.ImagesLoader("/Users/michal/datasets/rugd/train/img/")
    annotations_loader = image_libs.RUGDAnnLoader("/Users/michal/datasets/rugd/train/ann/")

    dataset = image_libs.SegmentationDataset(images_loader, annotations_loader, image_size = (256, 512))

    idx = 0

    for idx in range(len(images_loader)):

        img, ann = dataset.get_rgb(idx)
        
        result = numpy.concatenate([img, ann], axis=2)

        result = numpy.transpose(result, (1, 2, 0))  
        
        cv2.imshow("image", result)
        cv2.waitKey(50)