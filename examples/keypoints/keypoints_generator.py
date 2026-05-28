import numpy
import cv2

class KeypointsGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get(self):

        # this will be randomised
        n_points    = numpy.random.randint(3, 8)
        radius      = numpy.random.uniform(0.3, 0.7)
        noise       = 0.4
        hole_scale  = radius*numpy.random.uniform(0.1, 0.9)


        outer_points = self._random_polygon(n_points=n_points, radius=radius, noise=noise)

        if numpy.random.rand() < 0.5:
            holes = self._add_hole(outer_points, scale=hole_scale)
        else:
            holes = None

        mask = self._polygon_to_mask(outer_points, width=self.width, height=self.height, hole_points=holes)

        points = outer_points.copy()
        if holes is not None:
            points += holes

        keypoints = self._keypoints_to_map(points, width=self.width, height=self.height)

        _keypoints_to_heatmap_blur = self._keypoints_to_heatmap_blur(points, width=self.width, height=self.height, sigma=3)

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
        
    def _keypoints_to_heatmap_blur(self, points, width, height, sigma=2):
        heatmap = numpy.zeros((height, width), dtype=numpy.float32)

        # Assuming points are normalized (0.0 to 1.0) and in (x, y) format
        pts = numpy.array(points) * [width - 1, height - 1]
        pts = numpy.round(pts).astype(numpy.int32)

        ksize = int(5 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        radius = ksize // 2

        # Precompute Gaussian kernel (peak value is 1.0 at the center)
        ax = numpy.arange(ksize) - radius
        xx, yy = numpy.meshgrid(ax, ax)
        gaussian = numpy.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        for x, y in pts:
            # Ignore points that fall completely outside the heatmap boundaries
            if not (0 <= x < width and 0 <= y < height):
                continue

            # Map out boundaries on the heatmap
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(width, x + radius + 1)
            y2 = min(height, y + radius + 1)

            # Map out corresponding boundaries on the precomputed gaussian kernel
            # Fix: Ensure g_x corresponds to x changes, and g_y to y changes
            g_x1 = max(0, radius - x)
            g_y1 = max(0, radius - y)
            g_x2 = g_x1 + (x2 - x1)
            g_y2 = g_y1 + (y2 - y1)

            # Element-wise maximum handles overlapping/close keypoints cleanly
            heatmap[y1:y2, x1:x2] = numpy.maximum(
                heatmap[y1:y2, x1:x2],
                gaussian[g_y1:g_y2, g_x1:g_x2]
            )

        return heatmap
        

if __name__ == "__main__":
    generator = KeypointsGenerator(512, 512)
    mask, keypoints, heatmap = generator.get()

    result_img = numpy.zeros((mask.shape[0], mask.shape[1], 3), dtype=numpy.float32)
    result_img[mask == 1] = [0.1, 0.1, 0.1]
    result_img[keypoints == 1] = [0.0, 1.0, 0.0]
    #result_img[heatmap == 1] = [1.0, 0.0, 0.0]

    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
