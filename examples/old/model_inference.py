import time 

import image_libs
import numpy
import torch
from torchinfo import summary

import cv2


def show_result(x_img, y_seg_pred, y_kp_pred):

    x_img = numpy.transpose(x_img, (1, 2, 0))
    
    x_img = numpy.array(x_img, dtype=numpy.float32)
    x_img = numpy.ascontiguousarray(x_img)

    seq_img = numpy.zeros(x_img.shape)
    tmp = cv2.resize(y_seg_pred[0], (x_img.shape[1], x_img.shape[0]), interpolation = cv2.INTER_LINEAR)
    seq_img[:, :, 0] = tmp
    seq_img[:, :, 1] = tmp
    seq_img[:, :, 2] = tmp


    kp_img = numpy.zeros(x_img.shape)
    tmp = cv2.resize(y_kp_pred[0], (x_img.shape[1], x_img.shape[0]), interpolation = cv2.INTER_LINEAR)
    kp_img[:, :, 0] = tmp
    kp_img[:, :, 1] = tmp
    kp_img[:, :, 2] = tmp

    result = numpy.concatenate([x_img, seq_img, kp_img], axis=1)

    cv2.imshow("image", result)


if __name__ == "__main__":

    device = "mps"

    dataset_path = "/Users/michal/datasets/textures/"
    #dataset_path = "/home/michal/datasets/textures/"

    images_loader = image_libs.ImagesLoader(dataset_path)
    images_aug    = image_libs.ImagesAug(images_loader)
    dataset       = image_libs.TransformationsDataset(images_aug)

    image_height = 400
    image_width  = 800
    model = image_libs.KPModel((3, image_height, image_width))

    
    #example_input_tensor = torch.rand((1, 3, image_height, image_width)).to(device)

    print(model)
    summary(model, input_size=(1, 3, image_height, image_width))

    model.load_state_dict(torch.load("model_1.pt", map_location = "cpu"))
    model.to(device)

    #model = torch.jit.trace(model, example_input_tensor)
    model.eval()
    
    
    cam = cv2.VideoCapture(0)

    threshold = 0.2

    fps = 0.0

    while True:
        # read frame and resize
        ret, frame = cam.read()

        frame = cv2.resize(frame, (image_width, image_height), interpolation = cv2.INTER_NEAREST)
        x_in  = numpy.array(frame/256.0, dtype=numpy.float32)

        x_in = numpy.transpose(x_in, (2, 0, 1))

        time_start = time.time()
        # prediction
        x_in_t  = torch.from_numpy(x_in).unsqueeze(0).to(device)

        y_seq_pred, y_kp_pred, y_f_pred = model(x_in_t)

        # convert to numpy
        #y_seq_pred = (torch.sigmoid(y_seq_pred) > threshold).float()
        #y_kp_pred  = (torch.sigmoid(y_kp_pred) > threshold).float()

        y_seq_pred = torch.sigmoid(y_seq_pred)
        y_kp_pred  = torch.sigmoid(y_kp_pred)

        y_seq_pred  = y_seq_pred.squeeze(1).detach().cpu().numpy()
        y_kp_pred   = y_kp_pred.squeeze(1).detach().cpu().numpy()
        y_f_pred    = y_f_pred.squeeze(1).detach().cpu().numpy()

        time_stop = time.time()

        fps = 0.9*fps + 0.1/(time_stop - time_start)

        print("fps = ", round(fps, 1))

        show_result(x_in, y_seq_pred, y_kp_pred)


        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

   