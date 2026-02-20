import time 

import image_libs
import numpy
import torch

import cv2


def show_result(x_img, y_seg_gt, y_seg_pred, y_kp_gt, y_kp_pred):

    x_img = numpy.transpose(x_img, (1, 2, 0))
    
    x_img = numpy.array(x_img, dtype=numpy.float32)
    x_img = numpy.ascontiguousarray(x_img)

    seq_img = numpy.zeros(x_img.shape)
    seq_img[:, :, 0] = y_seg_gt[:, :]
    seq_img[:, :, 2] = cv2.resize(y_seg_pred[0], (x_img.shape[1], x_img.shape[0]), interpolation = cv2.INTER_NEAREST)


    kp_img = numpy.zeros(x_img.shape)
    kp_img[:, :, 0] = y_kp_gt[:, :]
    kp_img[:, :, 2] = cv2.resize(y_kp_pred[0], (x_img.shape[1], x_img.shape[0]), interpolation = cv2.INTER_NEAREST)


    result = numpy.concatenate([x_img, seq_img, kp_img], axis=1)

    
    cv2.imshow("image", result)
    cv2.waitKey(0)


if __name__ == "__main__":

    dataset_path = "/Users/michal/datasets/textures/"
    #dataset_path = "/home/michal/datasets/textures/"

    images_loader = image_libs.ImagesLoader(dataset_path)
    images_aug    = image_libs.ImagesAug(images_loader)
    dataset       = image_libs.TransformationsDataset(images_aug)

    image_height = 400
    image_width  = 800
    model = image_libs.KPModel((3, image_height, image_width))
    
    #model.load_state_dict(torch.load("model_1.pt", map_location = "cpu"))
    model.eval()

    print(model)

    x_in, kp, y_seg, y_kp_gt = dataset.get_kp((image_height, image_width), blurred_kp=True)

    x_in_t  = torch.from_numpy(x_in).unsqueeze(0)
    
    y_seq_pred, y_kp_pred, y_f_pred = model(x_in_t)
    

    
    y_seq_pred = (torch.sigmoid(y_seq_pred) > 0.5).float()
    y_kp_pred  = (torch.sigmoid(y_kp_pred) > 0.5).float()

    y_seq_pred  = y_seq_pred.squeeze(1).detach().cpu().numpy()
    y_kp_pred   = y_kp_pred.squeeze(1).detach().cpu().numpy()
    y_f_pred    = y_f_pred.squeeze(1).detach().cpu().numpy()

    show_result(x_in, y_seg, y_seq_pred, y_kp_gt, y_kp_pred)
    