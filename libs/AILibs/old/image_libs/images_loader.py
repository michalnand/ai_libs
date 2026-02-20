import cv2
import numpy
import os

class ImagesLoader:

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


class ImagesAug:

    def __init__(self, images_loader, size = (512, 512)):

        self.images_loader = images_loader
        self.size = size

    def set_size(self, size):
        self.size = size

    def __len__(self):
        return len(self.images_loader)

    def __getitem__(self, idx):
        img = self.images_loader[idx]

        # random crop
        if numpy.random.rand() > 0.5:
            img = self._random_crop(img)

        # resize
        img = self._resize(img)

        # random brightness
        if numpy.random.rand() > 0.5:
            img = self._random_brightness(img)
        
        # random color inversion
        if numpy.random.rand() > 0.5:
            img = self._random_inv(img)
        
        # random noise
        if numpy.random.rand() > 0.5:
            img = self._random_noise(img)

        # random flip
        if numpy.random.rand() > 0.5:
            img = numpy.flip(img, axis=1)

        # random flip
        if numpy.random.rand() > 0.5:
            img = numpy.flip(img, axis=2)

        # random channels shuffle
        if numpy.random.rand() > 0.5:
            img = self._random_ch_shuffle(img)

        # random channels inversion
        if numpy.random.rand() > 0.5:
            img = self._random_ch_inversion(img)


        img = numpy.clip(img, 0.0, 1.0)
        img = numpy.array(img, dtype=numpy.float32)
        return img


    def _random_crop(self, img):
        height = img.shape[1]
        width  = img.shape[2]

        crop_height = int(height*0.8)
        crop_width  = int(width*0.8)

        top = numpy.random.randint(0, height - crop_height + 1)
        left = numpy.random.randint(0, width - crop_width + 1)

        cropped_img = img[:, top:top + crop_height, left:left + crop_width]

        return cropped_img


    def _resize(self, img):

        img = numpy.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation= cv2.INTER_LINEAR)
        img = numpy.transpose(img, (2, 0, 1))

        return img

  
    def _random_brightness(self, img):
        br = numpy.random.uniform(0.1, 2.0)
        return br*img


    def _random_inv(self, img):
        return 1.0 - img


    def _random_noise(self, img):
        noise = numpy.random.rand(img.shape[0], img.shape[1], img.shape[2])
        level = numpy.random.rand()

        img = img + level*noise

        return img


    def _random_ch_shuffle(self, img):
        permuted_indices = numpy.random.permutation(3)
        return img[permuted_indices, :, :]


    def _random_ch_inversion(self, img):
        result = numpy.array(img)

        for n in range(3):
            if numpy.random.rand() > 0.5:
                result[n, :, :] = 1.0 - result[n, :, :]

        return result
