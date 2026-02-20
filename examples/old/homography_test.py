import image_libs


import numpy

import cv2

if __name__ == "__main__":
    img = cv2.imread("test_image.jpg")
    img = numpy.array(img/255.0, dtype=numpy.float32)
    img = numpy.transpose(img, (2, 0, 1))


    #h = image_libs.homography_rotation(-45.0*numpy.pi/180.0)
    #h = image_libs.homography_translation(50.0, 50.0)

    
    h = image_libs.homography_random()
    #h = image_libs.homography_perspective(0.0002, 0.0002)



    img_b = image_libs.apply_h_image(h, img)



    img_b = numpy.transpose(img_b, (1, 2, 0))

    cv2.imshow("image", img_b)
    cv2.waitKey(0)
    print(img_b.shape)
    #print(points_h)

    print("done")