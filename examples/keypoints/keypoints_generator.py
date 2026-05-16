import numpy
import cv2

class KeypointsGenerator:
    def __init__(self):
        pass

    def get(self):

        # this will be randomised
        n_points    = 16
        radius      = 0.4
        noise       = 0.2
        hole_scale  = 0.3

        width = 512
        height = 512
        
        outer_points = self._random_polygon(n_points=n_points, radius=radius, noise=noise)

        if numpy.random.rand() < 0.5:
            holes = self._add_hole(outer_points, scale=hole_scale)
        else:
            holes = None

        mask = self._polygon_to_mask(outer_points, width=width, height=height, hole_points=holes)

        points = outer_points.copy()
        if holes is not None:
            points += holes

        keypoints = self._keypoints_to_map(points, width=width, height=height)

        _keypoints_to_heatmap_blur = self._keypoints_to_heatmap_blur(points, width=width, height=height, sigma=5)

        return mask, keypoints, _keypoints_to_heatmap_blur
    
    # polygone points generation 

    def _random_polygon(self, n_points=10, center=(0.5, 0.5), radius=0.4, noise=0.2):
        angles = numpy.sort(numpy.random.rand(n_points) * 2 * numpy.pi)

        radii = radius * (1 + noise * (numpy.random.rand(n_points) - 0.5))

        x = center[0] + radii * numpy.cos(angles)
        y = center[1] + radii * numpy.sin(angles)

        points = numpy.stack([x, y], axis=1)    

        # clip to [0,1]
        points = numpy.clip(points, 0, 0.999)

        return points.tolist()
    

    def _add_hole(self, outer_points, scale=0.3):
        center = numpy.mean(outer_points, axis=0)
        inner = (outer_points - center) * scale + center
        return inner.tolist()
    

    def _polygon_to_mask(self, points, width, height, holes=None):
        mask = numpy.zeros((height, width), dtype=numpy.uint8)

        pts = numpy.array(points) * [width, height]
        pts = pts.astype(numpy.int32)

        cv2.fillPoly(mask, [pts], 1)

        if holes is not None:
            for hole in holes:
                hpts = numpy.array(hole) * [width, height]
                hpts = hpts.astype(numpy.int32)
                cv2.fillPoly(mask, [hpts], 0)

        return mask
    

    # rendering

    def _polygon_to_mask(self, points, width, height, hole_points=None):
        mask = numpy.zeros((height, width), dtype=numpy.float32)

        pts = numpy.array(points) * [width, height]
        pts = pts.astype(numpy.int32)

        cv2.fillPoly(mask, [pts], 1)

        if hole_points is not None: 
            hpts = numpy.array(hole_points) * [width, height]
            hpts = hpts.astype(numpy.int32) 
            cv2.fillPoly(mask, [hpts], 0)

        return mask
    

    def _keypoints_to_map(self, points, width, height):
        kp_map = numpy.zeros((height, width), dtype=numpy.float32)

        pts = numpy.array(points) * [width, height]
        pts = pts.astype(numpy.int32)

        x = pts[:, 0]
        y = pts[:, 1]

        kp_map[y, x] = 1.0  

        return kp_map
    

    def _keypoints_to_heatmap_blur_OLD(self, kp_map, sigma=3):
        ksize = int(5 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        heatmap = cv2.GaussianBlur(kp_map, (ksize, ksize), sigma)

        # normalize so peak = 1
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap
    

    def _keypoints_to_heatmap_blur(self, points, width, height, sigma=3):
        heatmap = numpy.zeros((height, width), dtype=numpy.float32)

        pts = numpy.array(points) * [width, height]
        pts = pts.astype(numpy.int32)

        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        # precompute gaussian kernel
        ax = numpy.arange(ksize) - ksize // 2
        xx, yy = numpy.meshgrid(ax, ax)
        gaussian = numpy.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        for x, y in pts:
            x1 = max(0, x - ksize // 2)
            y1 = max(0, y - ksize // 2)
            x2 = min(width, x + ksize // 2 + 1)
            y2 = min(height, y + ksize // 2 + 1)

            g_x1 = max(0, ksize // 2 - x)
            g_y1 = max(0, ksize // 2 - y)
            g_x2 = g_x1 + (x2 - x1)
            g_y2 = g_y1 + (y2 - y1)

            heatmap[y1:y2, x1:x2] = numpy.maximum(
                heatmap[y1:y2, x1:x2],
                gaussian[g_y1:g_y2, g_x1:g_x2]
            )

        return heatmap
    

if __name__ == "__main__":
    generator = KeypointsGenerator()
    mask, keypoints, heatmap = generator.get()

    result_img = numpy.zeros((mask.shape[0], mask.shape[1], 3), dtype=numpy.float32)
    result_img[mask == 1] = [0.1, 0.1, 0.1]
    #result_img[keypoints == 1] = [0, 1.0, 0]
    result_img[heatmap == 1] = [1.0, 1.0, 1.0]

    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
