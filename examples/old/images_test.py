import image_libs


import numpy

import cv2


def show_img_kp(img, key_points):

    if len(img.shape) == 2:
        img = numpy.dstack([img, img, img])
    else:
        img = numpy.transpose(img, (1, 2, 0))

    img = numpy.array(img, dtype=numpy.float32)


    for n in range(key_points.shape[0]):
        x = int(key_points[n, 0])
        y = int(key_points[n, 1])

        img = cv2.circle(img, center=(x, y), radius=3, color=(0.0, 1.0, 0.0), thickness=-1)
    

    cv2.imshow("image", img)
    cv2.waitKey(0)



def load_image(file_name, size = (512, 512)):
    img = cv2.imread(file_name)

    img = cv2.resize(img, size, interpolation= cv2.INTER_LINEAR)
    img = numpy.array(img/255.0, dtype=numpy.float32)
    img = numpy.transpose(img, (2, 0, 1))

    return numpy.array(img)


def noise_image(size = (512, 512)):
    result = numpy.random.rand(size[0], size[1], 3)

    # random blending noise
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


def generate_random_pair(img_a, img_b):

   
    pattern_id = numpy.random.randint(0, 4)

    pattern_id = 2
    
    if pattern_id == 0:
        mask_orig, key_points_orig = image_libs.generate_random_lines(gray_levels=True)
    elif pattern_id == 1:
        mask_orig, key_points_orig = image_libs.generate_random_polygon(gray_levels=True)
    elif pattern_id == 2:
        mask_orig, key_points_orig = image_libs.generate_random_polygons(gray_levels=True)
    elif pattern_id == 3:
        mask_orig, key_points_orig = image_libs.generate_random_star(gray_levels=True)
    elif pattern_id == 4:
        mask_orig, key_points_orig = image_libs.generate_random_checkerboard(gray_levels=True)

   
    # random smooth blending
    if numpy.random.rand() < 0.5:
        blend_size = 0
    else:
        blend_size = 2*numpy.random.randint(0, 10) + 1

    img_orig = image_libs.blend_images(img_a, img_b, mask_orig, blend_size)

    return img_orig, key_points_orig



if __name__ == "__main__":

    img_a = load_image("img_a.jpg")
    img_b = load_image("img_b.jpg")

    #img_a = noise_image()
    #img_b = noise_image()

    #img_a = numpy.ones((3, 512, 512), dtype=numpy.float32)
    img_b = numpy.ones((3, 512, 512), dtype=numpy.float32)

    translation_range = (-0.25*img_a.shape[1], 0.25*img_a.shape[1])
    h = image_libs.homography_random(translation_range)

    img_orig, key_points_orig = generate_random_pair(img_a, img_b)


    print(h)
    key_points_h = image_libs.apply_h_points(h, 1.0*key_points_orig)
    image_h      = image_libs.apply_h_image(h, img_orig)

    show_img_kp(img_orig, key_points_orig)
    #show_img_kp(image_h, key_points_h)
