import image_libs

import numpy
import cv2



def process_img(img, x_mask, kp_gt, key_points, p):

    if len(img.shape) == 2:
        img = numpy.dstack([img, img, img])
    else:
        img = numpy.transpose(img, (1, 2, 0))



    x_mask = numpy.dstack([x_mask[0], x_mask[0], x_mask[0]])
    kp_gt = numpy.dstack([kp_gt[0], kp_gt[0], kp_gt[0]])



    img = numpy.array(img, dtype=numpy.float32)
    img = numpy.ascontiguousarray(img)

    img = img*0.5
    for n in range(key_points.shape[0]):
        x = int(key_points[n, 0])
        y = int(key_points[n, 1])

        img = cv2.circle(img, center=(x, y), radius=4, color=(0.0, 1.0, 0.0), thickness=-1)

    print(">>>> ", p.shape)

    
    for n in range(p.shape[0]):
        x = int(p[n, 0])
        y = int(p[n, 1])

        img = cv2.circle(img, center=(x, y), radius=8, color=(0.0, 0.0, 1.0), thickness=-1)
    

    img = numpy.concatenate([img, x_mask, kp_gt], axis=1)

    return img


def show_img_all(xa, xa_mask, xa_kp, kpa,   xb, xb_mask, xb_kp, kpb, pa, pb):

    res_a = process_img(xa, xa_mask, xa_kp, kpa, pa)
    res_b = process_img(xb, xb_mask, xb_kp, kpb, pb)

    result = numpy.concatenate([res_a, res_b], axis=0)
    cv2.imshow("image", result)
    cv2.waitKey(0)




if __name__ == "__main__":


    images_loader = image_libs.ImagesLoader("/Users/michal/datasets/textures/")
    images_aug    = image_libs.ImagesAug(images_loader)
    

    dataset = image_libs.TransformationsDataset(images_aug)


        
    xa, xa_mask, xa_kp, kpa,  xb, xb_mask, xb_kp, kpb, pa, pb = dataset.get((600, 800))

    print(xa.shape, xa_mask.shape, xa_kp.shape, kpa.shape)
    print(xb.shape, xb_mask.shape, xb_kp.shape, kpb.shape)
    print(pa.shape, pb.shape)

    show_img_all(xa, xa_mask, xa_kp, kpa,   xb, xb_mask, xb_kp, kpb, pa, pb)


