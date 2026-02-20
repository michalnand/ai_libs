import cv2
import numpy

from scipy.interpolate import splprep, splev


def generate_random_lines(image_size=(512, 512), lines_min = 2, lines_max = 30, gray_levels = False):
    h, w = image_size
   
    num_lines = numpy.random.randint(lines_min, lines_max)
    keypoints = []
    
    result = numpy.zeros((h, w), dtype=numpy.float32)

    for _ in range(num_lines):
        
        if gray_levels:
            intensity = numpy.random.uniform(0.75, 1.0)
        else:
            intensity = 1.0
            
        pt1 = numpy.random.randint(0, w, 2)
        pt2 = numpy.random.randint(0, w, 2)
        th  = numpy.random.randint(1, 8)
        result = cv2.line(result, tuple(pt1), tuple(pt2), (intensity, intensity, intensity), th)
        keypoints.extend([pt1, pt2])

    keypoints = numpy.array(keypoints, dtype=int)

    return result, keypoints



def generate_random_polygon(image_size=(512, 512), points_max = 30, gray_levels = False):
    h, w = image_size
    num_pts = numpy.random.randint(3, points_max)

    # Random center
    cx = numpy.random.randint(w // 10, 9 * w // 10)
    cy = numpy.random.randint(h // 10, 9 * h // 10)

    # Generate angles and random radii
    angles = numpy.sort(numpy.random.uniform(0, 2 * numpy.pi, num_pts))
    
    min_r = min(w, h) // 10
    max_r = min(w, h)

    radii = numpy.random.uniform(min_r, max_r, size=num_pts)

    # Calculate coordinates
    x = cx + radii * numpy.cos(angles)
    y = cy + radii * numpy.sin(angles)

    keypoints = numpy.stack([x, y], axis=1).astype(numpy.int32)
    keypoints[:, 0] = numpy.clip(keypoints[:, 0], 0, w - 1)
    keypoints[:, 1] = numpy.clip(keypoints[:, 1], 0, h - 1)

    # Fill polygon safely
    result = numpy.zeros((h, w), dtype=numpy.float32)
    contour = keypoints.reshape((-1, 1, 2))

    if gray_levels:
        intensity = numpy.random.uniform(0.75, 1.0)
    else:
        intensity = 1.0

    cv2.fillPoly(result, [contour], intensity)

    return result, keypoints






def generate_random_polygons(image_size=(512, 512), points_max = 12, polygons_max_count = 20, gray_levels = False):
    h, w = image_size

    result = numpy.zeros((h, w), dtype=numpy.float32)

    num_pts = numpy.random.randint(3, points_max)

    num_polygons = numpy.random.randint(1, polygons_max_count)

    keypoints_all = []
    for n in range(num_polygons):
        cx = numpy.random.randint(0, w-1)
        cy = numpy.random.randint(0, h-1)
        
        #angles = numpy.sort(numpy.random.uniform(0, 2.0*numpy.pi, num_pts))

        angles = numpy.random.uniform(0, 2.0*numpy.pi, num_pts)
       
        radius = numpy.random.randint(min(w, h)//50, min(w, h)//6)
        
        keypoints = numpy.stack([
            cx + radius * numpy.cos(angles),
            cy + radius * numpy.sin(angles)
        ], axis=1).astype(numpy.int32)


        keypoints[:, 0] = numpy.clip(keypoints[:, 0], 0, w-1)
        keypoints[:, 1] = numpy.clip(keypoints[:, 1], 1, h-1)

        if gray_levels:
            intensity = numpy.random.uniform(0.75, 1.0)
        else:
            intensity = 1.0

        result = cv2.fillPoly(result, [keypoints], intensity)
        
        keypoints_all.extend(keypoints)

    keypoints_all = numpy.array(keypoints_all, dtype=int)

    return result, keypoints_all




def generate_curvy_flake(image_size=(512, 512), points_max=20, gray_levels=False):
    h, w = image_size
    num_pts = numpy.random.randint(5, points_max)  # more points for smoother shapes
    
    # Random angles and radii around center

    cx = numpy.random.randint(0, w)
    cy = numpy.random.randint(0, h)

    angles = numpy.sort(numpy.random.uniform(0, 2*numpy.pi, num_pts))
    radii = numpy.random.uniform(min(h, w)//8, min(h, w), num_pts)

    x = cx + radii * numpy.cos(angles)
    y = cy + radii * numpy.sin(angles)

    # Ensure the shape loops back by repeating first point at the end
    x = numpy.append(x, x[0])
    y = numpy.append(y, y[0])

    # Create smooth spline through points
    tck, _ = splprep([x, y], s=0, per=True)
    u_fine = numpy.linspace(0, 1, 1000)
    x_smooth, y_smooth = splev(u_fine, tck)

    # Clip to image boundaries
    x_smooth = numpy.clip(x_smooth, 0, w - 1).astype(numpy.int32)
    y_smooth = numpy.clip(y_smooth, 0, h - 1).astype(numpy.int32)

    # Prepare mask image
    result = numpy.zeros((h, w), dtype=numpy.float32)
    smooth_contour = numpy.stack([x_smooth, y_smooth], axis=1)

    if gray_levels:
        intensity = numpy.random.uniform(0.75, 1.0)
    else:
        intensity = 1.0

    cv2.fillPoly(result, [smooth_contour], intensity)

    # Return result and control points (keypoints)
    keypoints = numpy.stack([x[:-1], y[:-1]], axis=1).astype(numpy.int32)
    return result, keypoints



def generate_random_star(image_size=(512, 512), rays_min = 3, rays_max = 20, gray_levels = False):
    h, w = image_size

    keypoints = []

    cx = numpy.random.randint(w//4, 3*w//4)
    cy = numpy.random.randint(h//4, 3*h//4)

    keypoints.append([cx, cy])

    num_rays = numpy.random.randint(rays_min, rays_max)
    angles = numpy.linspace(0, 2*numpy.pi, num_rays, endpoint=False)
  
    img = numpy.zeros((h, w), dtype=numpy.float32)

    for angle in angles:
        if gray_levels:
            intensity = numpy.random.uniform(0.75, 1.0)
        else:
            intensity = 1.0

        length = numpy.random.randint(min(h, w)//5, min(h, w))

        y = cy + length*numpy.cos(angle)
        x = cx + length*numpy.sin(angle)

        y = numpy.clip(y, 0, h-1)
        x = numpy.clip(x, 0, h-1)

        y = int(y)
        x = int(x)

        th = numpy.random.randint(1, 8)
        
        img = cv2.line(img, (cx, cy), (x, y), (intensity, intensity, intensity), th)

        keypoints.append([x, y])

    keypoints = numpy.array(keypoints, dtype=int)

    return img, keypoints





def generate_random_stars(image_size=(512, 512), rays_min = 3, rays_max = 20, stars_max_count = 20, gray_levels = False):
    h, w = image_size

    keypoints_all = []
    img = numpy.zeros((h, w), dtype=numpy.float32)

    for n in range(5):
        keypoints = []


        if gray_levels:
            intensity = numpy.random.uniform(0.75, 1.0)
        else:
            intensity = 1.0



        num_pts = numpy.random.randint(3, rays_max)

        cx = numpy.random.randint(1, w-1)
        cy = numpy.random.randint(1, h-1)
        
        angles  = numpy.sort(numpy.random.uniform(0, 2.0*numpy.pi, num_pts))
        lengths = numpy.random.uniform(1, min(w, h), num_pts)


        keypoints.append([cx, cy])

        for n in range(num_pts):
            x = cx + lengths[n]*numpy.sin(angles[n])
            y = cy + lengths[n]*numpy.cos(angles[n])
            
            x = numpy.clip(x, 0, w-1).astype(int)
            y = numpy.clip(y, 0, h-1).astype(int)
            

            th = numpy.random.randint(1, 8)   
                        
            img = cv2.line(img, (cx, cy), (x, y), (intensity, intensity, intensity), th)

            keypoints.append([x, y])


        keypoints_all.extend(keypoints)

    keypoints_all = numpy.array(keypoints_all, dtype=int)

    return img, keypoints_all


def generate_random_checkerboard(image_size=(512, 512), max_squares=32, gray_levels=True):
    h, w = image_size
 
    img = numpy.zeros((h, w), dtype=numpy.float32)

    squares = numpy.random.randint(2, max_squares)

    step_x = w // squares
    step_y = h // squares

    # Draw checkerboard pattern
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                if gray_levels:
                    intensity = numpy.random.uniform(0.75, 1.0)
                else:
                    intensity = 1.0

                x0, y0 = i * step_x, j * step_y
                x1, y1 = x0 + step_x, y0 + step_y
                img = cv2.rectangle(img, (x0, y0), (x1, y1), intensity, -1)

    # Generate all internal grid corners only once
    keypoints = []
    for i in range(squares + 1):
        for j in range(squares + 1):
            x = i * step_x
            y = j * step_y
            keypoints.append([x, y])

    keypoints = numpy.array(keypoints, dtype=int)

    return img, keypoints

