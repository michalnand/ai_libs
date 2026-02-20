from .homography_mat import *
from .homography import *
from .random_images import *


class TransformationsDataset:

    def __init__(self, images_loader = None):
        self.images_loader = images_loader



    def get(self, image_size = (512, 512), num_z_points = 32, p_aug = 0.5):
        if self.images_loader is not None:
            self.images_loader.set_size(image_size)

        # sample random background and foreground
        background = self._random_bg_img(image_size)
        foreground = self._random_fg_img(image_size)

        # generate random pattern with keypoints
        xa_mask, kpa = self.random_patterns(image_size)

        # blurred mask
        if numpy.random.rand() > 0.5:
            xa_mask[0] = cv2.GaussianBlur(xa_mask[0], (21, 21), 3)
        
        kpa = self._clip_keypoints_range(kpa, image_size)

        # mix background with foreground
        xa = (1.0 - xa_mask)*background + xa_mask*foreground

        # fill keypoints into image
        xa_kp = self._create_kp_mask(image_size, kpa)


        # generate random homography
        h = self._random_homography(image_size)

        xb  = apply_h_image(h, xa)

        kpb      = apply_h_points(h, 1.0*kpa)
        xb_mask  = apply_h_image(h, xa_mask)

        kpb = self._clip_keypoints_range(kpb, image_size)

        xb_kp = self._create_kp_mask(image_size, kpb)

        # mask thresholding
        xa_mask = self._binarise_mask(xa_mask)
        xb_mask = self._binarise_mask(xb_mask)


        # random points sample
        x = numpy.random.randint(0, image_size[1], (num_z_points, ))
        y = numpy.random.randint(0, image_size[0], (num_z_points, ))

        pa = numpy.array(numpy.vstack([x, y]).T)
        pb = apply_h_points(h, 1.0*pa)
        pb = self._clip_keypoints_range(pb, image_size)


        return xa, xa_mask, xa_kp, kpa, xb, xb_mask, xb_kp, kpb,  pa, pb




    def random_patterns(self, image_size, gray_levels = True):
        
        pattern_id = numpy.random.randint(0, 5)

        if pattern_id == 0:
            mask, key_points = generate_random_lines(image_size, gray_levels=gray_levels)
        elif pattern_id == 1:
            mask, key_points = generate_random_polygon(image_size, gray_levels=gray_levels)
        elif pattern_id == 2:
            mask, key_points = generate_curvy_flake(image_size, gray_levels=gray_levels)
        elif pattern_id == 3:
            mask, key_points = generate_random_checkerboard(image_size, gray_levels=gray_levels)
        elif pattern_id == 4:
            mask, key_points = generate_random_stars(image_size, gray_levels=gray_levels)

        mask = numpy.expand_dims(mask, 0)

        return mask, key_points


        
    def _get_random_noise_image(self, size):
        result = numpy.random.rand(size[0], size[1], 3)

        # random blur noise
        if numpy.random.rand() > 0.5:
            blend_size = 2*numpy.random.randint(0, 7) + 1
            result = cv2.GaussianBlur(result, (blend_size, blend_size), 1)

        result = numpy.transpose(result, (2, 0, 1))

        # random magnitude scaling
        if numpy.random.rand() > 0.5:
            mag = numpy.random.uniform(0.1, 1.0)
            result = result*mag

        result = numpy.array(result, dtype=numpy.float32)

        return result

    def _random_blur(self, x):    
        kernel_size = 2*numpy.random.randint(1, 10) + 1
        result = cv2.GaussianBlur(x, (kernel_size, kernel_size), 1)

        return result
 


    def _random_fg_img(self, size):
        idx = numpy.random.randint(0, 2)
        
        if idx == 0:
            result = numpy.ones((3, size[0], size[1]), dtype=numpy.float32)
        else:
            img_idx = numpy.random.randint(0, len(self.images_loader))
            result  = self.images_loader[img_idx]

        return result

    def _random_bg_img(self, size):

        idx = numpy.random.randint(0, 2)
        
        idx = 1
        if idx == 0:
            result = numpy.zeros((3, size[0], size[1]), dtype=numpy.float32)
        else:
            img_idx = numpy.random.randint(0, len(self.images_loader))
            result  = self.images_loader[img_idx]

        return result



    def _filter_kp(self, kp_orig, kp_h, h, w, boundary = 1):
        kp_orig_result = []
        kp_h_result    = []

        for n in range(len(kp_orig)):

            xa = kp_orig[n][0]
            ya = kp_orig[n][1]

            xb = kp_h[n][0]
            yb = kp_h[n][1]

            #if xa > boundary and xa < (w-boundary) and ya > boundary and ya < (h-boundary):
            if xb > boundary and xb < (w-boundary) and yb > boundary and yb < (h-boundary):
                kp_orig_result.append(kp_orig[n])   
                kp_h_result.append(kp_h[n])

        kp_orig_result = numpy.array(kp_orig_result)
        kp_h_result = numpy.array(kp_h_result)

        return kp_orig_result, kp_h_result

    def _clip_keypoints_range(self, kp, image_size):
        kp_result = numpy.array(kp)

        kp_result[:, 0] = numpy.clip(kp_result[:, 0], 0, image_size[1]-1)
        kp_result[:, 1] = numpy.clip(kp_result[:, 1], 0, image_size[0]-1)

        return numpy.array(kp_result, dtype=int)

    def _binarise_mask(self, mask, th = 0.5):
        return numpy.array((mask > th), dtype=numpy.float32)

    def _create_kp_mask(self, image_size, kp, blurred = True):

        result = numpy.zeros((1, image_size[0], image_size[1]), dtype=numpy.float32)
        n_points = len(kp) 
        result[0, kp[range(n_points), 1], kp[range(n_points), 0]] = 1.0

        
        if blurred:
            result[0] = cv2.GaussianBlur(result[0], (7, 7), 3)

            max_blur_value = numpy.percentile(result[result > 0], 90) 
            result = result/(max_blur_value + 10e-6)
        


        return result

    def _random_homography(self, image_size):
        size = min(image_size)

        translation_range   = [0, 0]
        rotation_range      = [0, 0]
        scale_range         = [1.0, 1.0]
        shear_range         = [0, 0]

        if numpy.random.rand() > 0.5:
            translation_range   = [-size/10, size/10]
        
        if numpy.random.rand() > 0.5:
            rotation_range      = [-numpy.pi/4, numpy.pi/4]
        
        if numpy.random.rand() > 0.5:
            scale_range         = [0.5, 2.0]
        
        if numpy.random.rand() > 0.5:
            shear_range         = [-0.1, 0.1]

        h = homography_random(translation_range, rotation_range, scale_range, shear_range)

        return h


