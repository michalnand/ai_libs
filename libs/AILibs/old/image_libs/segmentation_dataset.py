import numpy
import cv2

class SegmentationDataset:
    def __init__(self, images_loader, annotations_loader, image_size = (512, 512), apply_augs = True):

        self.images_loader      = images_loader
        self.annotations_loader = annotations_loader
        self.image_size         = image_size
        self.apply_augs         = apply_augs

        self.input_shape        = (3, self.image_size[0], self.image_size[1])
        self.num_classes        = annotations_loader.num_classes


    def __len__(self):
        return len(self.images_loader)

    def __getitem__(self, idx):
        return self.get(idx)

    def get_rgb(self, idx):
        img, ann = self.get(idx)
        ann_rgb  = self.annotations_loader.as_rgb(ann)
        return img, ann_rgb

    def get(self, idx):
        img = self.images_loader[idx]

        if self.apply_augs:
            img = self._aug_image(img)

        ann = self.annotations_loader[idx]
        

        if self.apply_augs:
           img, ann = self._aug_joint(img, ann)

        # final resize
        img = self._resize_im(img)
        img = numpy.clip(img, 0.0, 1.0)

        ann = self._resize_ann(ann)
        ann = numpy.array(ann, dtype=int)

        return img, ann 


   

    # colors augmentation
    def _aug_image(self, x):
    
        # random brightness
        if numpy.random.rand() > 0.5:
            x = self._random_brightness(x)
        
        # random color inversion
        if numpy.random.rand() > 0.5:
            x = self._random_inv(x)
        
        # random noise
        if numpy.random.rand() > 0.5:
            x = self._random_noise(x)

        # random channels shuffle
        if numpy.random.rand() > 0.5:
            x = self._random_ch_shuffle(x)

        # random channels inversion
        if numpy.random.rand() > 0.5:
            x = self._random_ch_inversion(x)


        # random blur
        if numpy.random.rand() > 0.5:
            kx = 2*numpy.random.randint(0, 10) + 1
            ky = 2*numpy.random.randint(0, 10) + 1

            x = numpy.transpose(x, (1, 2, 0)) 
            x = cv2.GaussianBlur(x,(kx, ky), 3)
            x = numpy.transpose(x, (2, 0, 1))
        

        # apply random convolution 
        if numpy.random.rand() > 0.5:
            ks = 7
            kernel = numpy.random.randn(ks, ks).astype(numpy.float32)

            kernel /= (numpy.max(numpy.abs(kernel)) + 1e-7)

            x = numpy.transpose(x, (1, 2, 0)) 
            x = cv2.filter2D(x, ddepth=-1, kernel=kernel)
            x = numpy.transpose(x, (2, 0, 1))

        return x

    def _aug_joint(self, img, ann):
        
        # random flip
        if numpy.random.rand() > 0.5:
            img = numpy.flip(img, axis=1)
            ann = numpy.flip(ann, axis=0)   

        # random flip
        if numpy.random.rand() > 0.5:
            img = numpy.flip(img, axis=2)
            ann = numpy.flip(ann, axis=1)
    
        # random crop
        if numpy.random.rand() > 0.5:
            img, ann = self._random_crop(img, ann)
        
        # random rotation
        if numpy.random.rand() > 0.5:
            img, ann = self._random_rotation(img, ann)
        
        return img, ann


    def _resize(self, img):
        pass

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

    def _random_crop(self, img, ann):
        height = img.shape[1]
        width  = img.shape[2]

        crop_height = int(height*0.5)   
        crop_width  = int(width*0.5)    

        top  = numpy.random.randint(0, height - crop_height + 1)
        left = numpy.random.randint(0, width - crop_width + 1)

        cropped_img = img[:, top:top + crop_height, left:left + crop_width]
        cropped_ann = ann[top:top + crop_height, left:left + crop_width]

        return cropped_img, cropped_ann

    def _resize_im(self, img):

        img = numpy.transpose(img, (1, 2, 0))   
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation= cv2.INTER_LINEAR)
        img = numpy.transpose(img, (2, 0, 1))

        return img

    def _resize_ann(self, ann):

        ann = cv2.resize(1.0*ann, (self.image_size[1], self.image_size[0]), interpolation= cv2.INTER_LINEAR)
        ann = numpy.array(ann, dtype=int)
        return ann

    def _random_rotation(self, img, ann):
        angle = numpy.random.uniform(-25.0, 25.0)
        scale = 1.0

        w = self.image_size[1]
        h = self.image_size[0]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        img = numpy.transpose(img, (1, 2, 0))
        img = cv2.warpAffine(img, M, (w, h))
        img = numpy.transpose(img, (2, 0, 1))

        ann = cv2.warpAffine(1.0*ann, M, (w, h))
        ann = numpy.array(ann, dtype=int)

        return img, ann
