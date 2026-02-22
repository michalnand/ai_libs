import cv2
import numpy
import os

class ImagesLoader:

    """
        Runtime images loader. Loads images from a given directory and its subdirectories, 
        resizes them to a specified size (if provided), 
        normalizes pixel values to the range [0, 1], 
        and returns them as numpy arrays in CHW format (channels, height, width). 
        The loader supports common image formats such as JPG and PNG.
    """

    def __init__(self, root_path, size = None):
        self.images_path = self._find_images(root_path)
        self.images_path.sort()

        self.size = size

        print("images list")
        for p in self.images_path:
            print(p)
        print()
        print("images count ", len(self.images_path))
        print()

    def _find_images(self, root_path):
        image_extensions = {'.jpg', '.JPG', '.png', '.PNG'}
        image_paths = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext in image_extensions:
                    full_path = os.path.join(dirpath, filename)
                    image_paths.append(full_path)

        return image_paths


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = self._load_image(self.images_path[idx])

        return img

    def _load_image(self, file_name):
        img = cv2.imread(file_name)

        if self.size is not None:
            img = cv2.resize(img, self.size, interpolation= cv2.INTER_LINEAR)
        
        img = numpy.array(img/255.0, dtype=numpy.float32)
        img = numpy.transpose(img, (2, 0, 1))

        return numpy.array(img)
