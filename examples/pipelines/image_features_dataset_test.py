import AILibs
import numpy

import cv2    

import torch

if __name__ == "__main__":

    width  = 300
    height = 200

    tile_size = 4

    batch_size = int(tile_size**2)

    dataset_root_path = "/Users/michal/datasets/pictures/"
    dataset = AILibs.DatasetImages(dataset_root_path)

    
    x = dataset.get_batch(batch_size)

    x = AILibs.crop_augmentation(x, 32, 0.5)  
    
    # all images to fixed size  
    x = AILibs.resize_augmentation(x, width, height)

    x = numpy.array(x, dtype=numpy.float32)

    
    x = torch.from_numpy(x)
    x = AILibs.photometric_augmentations(x, p = 0.2)
    
    # geometric augmentations
    M, M_inv = AILibs.generate_affine_matrices(batch_size)

    x = AILibs.affine_augmentation(x, M_inv)
    

    x = x.detach().cpu().numpy()
    


    result_im = numpy.zeros((tile_size*height, tile_size*width, 3), dtype=numpy.float32)

    for n in range(batch_size):
        ty = height*(n//tile_size)
        tx = width*(n%tile_size)

        x_tmp = numpy.moveaxis(x[n], 0, 2)

        result_im[ty:ty+height, tx:tx+width, :] = x_tmp

    
    cv2.imshow("image", result_im)
    cv2.waitKey(0)

    '''
    
    x = AILibs.torch.from_numpy(x)

    

    # simple colors augmentation, contrast, noise
    x0 = AILibs.photometric_augmentations(x)
    x1 = AILibs.photometric_augmentations(x)

    # geometric augmentations
    M, M_inv = AILibs.generate_affine_matrices(batch_size)
    x1 = AILibs.affine_augmentation(x1, M_inv.to(self.device))
    '''