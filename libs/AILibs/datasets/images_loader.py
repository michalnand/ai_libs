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

    def __init__(self, root_path, size = None, name_filter = None, keep_uint8 = False):
        self.images_path = self._find_images(root_path, name_filter)
        self.images_path.sort()

        self.size = size

        self.keep_uint8 = keep_uint8

        
        print("images list")
        for p in self.images_path:
            print(p)
        print()
        
        print("images count ", len(self.images_path))
        

    def _find_images(self, root_path, name_filter):
        image_extensions = {'.jpg', '.JPG', '.png', '.PNG'}
        image_paths = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                
                if name_filter is not None:
                    if name_filter not in str(filename):
                        continue
                
                tmp = os.path.basename(filename)
                if tmp.startswith("."):
                    continue

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

        h, w = img.shape[:2]    

        if img is None or h < 32 or w < 32:
            raise Exception("Image size too small, got " + str(h) + "x" + str(w) + " file " + str(file_name))

        if self.size is not None:
            img = cv2.resize(img, self.size, interpolation= cv2.INTER_NEAREST)
        
        if self.keep_uint8:
            img = numpy.array(img, dtype=numpy.uint8)
        else:
            img = numpy.array(img/255.0, dtype=numpy.float32)


        img = numpy.transpose(img, (2, 0, 1))   

        return numpy.array(img)
    

    def _is_valid(self, file_name):
        try:
            # File must exist and be at least 4 KB
            if not os.path.isfile(file_name):
                return False

            if os.path.getsize(file_name) < 4096:
                return False
            
            return True 

            '''
            img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

            if img is None:
                return False

            if len(img.shape) < 2:
                return False

            h, w = img.shape[:2]

            return h > 32 and w > 32
            '''

        except Exception:
            return False
