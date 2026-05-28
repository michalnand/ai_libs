from keypoints_generator import KeypointsGenerator

import torch
import numpy

from unet_model import UnetModel
from keypoints_metric import keypoint_metrics

import cv2

class KeyPointsDataset:
    def __init__(self, width, height, images_dir = None):
        self.width      = width
        self.height     = height
        self.generator  = KeypointsGenerator(width, height)

    def __len__(self):
        return 50000
    

    def get_batch(self, batch_size):
        batch_masks = []
        batch_keypoints = []
        batch_heatmaps = []

        for _ in range(batch_size):
            mask, keypoints, heatmap = self.generator.get()
            batch_masks.append(mask)
            batch_keypoints.append(keypoints)
            batch_heatmaps.append(heatmap)

        

        return numpy.array(batch_masks), numpy.array(batch_keypoints), numpy.array(batch_heatmaps)


if __name__ == "__main__":
    width  = 256
    height = 256

    batch_size  = 32
    num_steps   = 10000
    lr          = 0.001

    device      = "cpu"


    dataset_train = KeyPointsDataset(width, height)

    model = UnetModel(in_ch=1, out_ch=1)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
   
    loss_func_mask = torch.nn.BCELoss()

    for n in range(num_steps):

        mask, keypoints, heatmap = dataset_train.get_batch(batch_size)

        mask        = torch.from_numpy(mask).float().to(device).unsqueeze(1)
        keypoints   = torch.from_numpy(keypoints).float().to(device).unsqueeze(1)
        heatmap     = torch.from_numpy(heatmap).float().to(device).unsqueeze(1)


        keypoints_pred, mask_pred = model(mask)

        mask_pred = torch.sigmoid(mask_pred)
        loss_mask = loss_func_mask(mask_pred, mask)
        loss_kp = ((heatmap - keypoints_pred)**2).mean()

        loss = loss_mask + loss_kp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #keypoints_pred = torch.sigmoid(keypoints_pred)

        metric = keypoint_metrics(heatmap, torch.sigmoid(keypoints_pred))

        print(n, loss.item())
        for k, v in metric.items():
            print(f"    {k}: {v:.4f}")
        print("\n\n")


        x = 0.2*mask[0, 0].detach().cpu().numpy()
        y_gt = heatmap[0, 0].detach().cpu().numpy()
        mask_pred = mask_pred[0, 0].detach().cpu().numpy()
        keypoints_pred = keypoints_pred[0, 0].detach().cpu().numpy()

        # Visualize the heatmap and predictions
        res_image = numpy.zeros((height, width, 3), dtype=numpy.float32)

        res_image[..., 0] = numpy.maximum(x, mask_pred)  # prediction blue
        res_image[..., 1] = x   # mask in green
        res_image[..., 2] = numpy.maximum(x, y_gt) # GT red
        cv2.imshow("Keypoints Heatmap (Red=GT, Green=Mask, Blue=Pred)", res_image)
        cv2.waitKey(1)




        
