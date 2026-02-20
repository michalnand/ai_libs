import numpy
import cv2


'''
    h      : homography matrix, shape (3, 3) 
    points : batch of 2D points, of shape (n_points, 2)
'''
def apply_h_points(h, points):
    # expand into 3D
    ones = numpy.ones((points.shape[0], 1), dtype=numpy.float32)
    tmp  = numpy.concatenate([1.0*points, ones], axis=1)

    # apply homography: (n_points, 3)
    
    transformed_homog = (h @ tmp.T).T
    

    # normalize to get back 2D
    transformed_points = transformed_homog[:, :2] / transformed_homog[:, 2:3]

    return transformed_points.astype(numpy.float32)



'''
    apply homography on rgb image
    image : numpy float32 array of shape (3, height, width)
'''
def apply_h_image(h, image):
    
    if image.shape[0] == 3:
        image_hwc = numpy.transpose(image, (1, 2, 0))  # (H, W, 3)
        h_img, w_img = image_hwc.shape[:2]
    else:
        image_hwc = numpy.array(image[0])
    
    h_img = image_hwc.shape[0]
    w_img = image_hwc.shape[1]


    # Warp using OpenCV
    warped = cv2.warpPerspective(
        image_hwc, h, (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # black padding
    )   

    # Transpose back to CHW format
    if image.shape[0] == 3:
        warped_chw = numpy.transpose(warped, (2, 0, 1))  # (3, H, W)
    else:
        warped_chw = numpy.expand_dims(warped, 0)


    return warped_chw.astype(image.dtype)