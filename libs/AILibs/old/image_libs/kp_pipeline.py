import numpy
import torch

from .metric import *
from .loss_self_supervised import *

class KPPipeline:


    def __init__(self, Model, dataset, device = "cpu"):
        self.dataset    = dataset
        self.device     = device

        self.batch_size   = 32
        self.image_width  = 512     
        self.image_height = 256     


        n_features      = 64

        self.learning_rate = 0.001
        
        self.model = Model((3, self.image_height, self.image_width), n_features)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate)

        print(self.model)
        


    def step(self):
        xa, ya_mask_gt, ya_kp_gt,  xb, yb_mask_gt, yb_kp_gt, pa, pb  = self._sample_batch(self.dataset, self.batch_size, self.image_height, self.image_width)

        
        ya_mask_pred, ya_kp_pred, ya_z = self.model(xa)
        yb_mask_pred, yb_kp_pred, yb_z = self.model(xb)


        # rescale masks to original size
        sf = xa.shape[-1]//ya_mask_pred.shape[-1]

        ya_mask_pred    = torch.nn.functional.interpolate(ya_mask_pred, scale_factor=sf, mode='nearest')
        ya_kp_pred      = torch.nn.functional.interpolate(ya_kp_pred, scale_factor=sf, mode='nearest')
        
        yb_mask_pred    = torch.nn.functional.interpolate(yb_mask_pred, scale_factor=sf, mode='nearest')
        yb_kp_pred      = torch.nn.functional.interpolate(yb_kp_pred, scale_factor=sf, mode='nearest')



        # classification loss for mask and keypoints
        loss_func = torch.nn.BCEWithLogitsLoss()

        loss_seg = loss_func(ya_mask_pred, ya_mask_gt) + loss_func(yb_mask_pred, yb_mask_gt)

        loss_kp = loss_func(ya_kp_pred, ya_kp_gt) + loss_func(yb_kp_pred, yb_kp_gt)



        # self supervised loss for keypoints features matching
        za = self._extract_features(ya_z, pa)
        zb = self._extract_features(yb_z, pb)

        za = za.flatten(1)
        zb = zb.flatten(1)

        
        # features similarity term
        loss_sim = loss_mse_func(za, zb)

        # variance and invariance terms
        loss_std = loss_std_func(za) + loss_std_func(zb)
        loss_cov = loss_cov_func(za) + loss_cov_func(zb)


        # total loss
        loss = loss_seg + loss_kp + 1.0*loss_sim + 1.0*loss_std + (1.0/25.0)*loss_cov


        # optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log loss and metric
        result_log = {}
        result_log["loss"]      = float(loss.item())
        result_log["loss_seg"]  = float(loss_seg.item())
        result_log["loss_kp"]   = float(loss_kp.item())
        result_log["loss_sim"]  = float(loss_sim.item())
        result_log["loss_std"]  = float(loss_std.item())
        result_log["loss_cov"]  = float(loss_cov.item())

        seg_metric = compute_segmentation_metrics(ya_mask_gt, torch.sigmoid(ya_mask_pred))
        kp_metric  = compute_segmentation_metrics(ya_kp_gt, torch.sigmoid(ya_kp_pred))

        result_log["seg_metric"] = seg_metric
        result_log["kp_metric"]  = kp_metric

        return result_log
        

    
    def _sample_batch(self, dataset, batch_size = 32, image_height = 256, image_width = 512, num_z_points = 32):
        xa_batch        = torch.zeros((batch_size, 3, image_height, image_width), dtype=torch.float32)
        ya_mask_batch   = torch.zeros((batch_size, 1, image_height, image_width), dtype=torch.float32)
        ya_kp_batch     = torch.zeros((batch_size, 1, image_height, image_width), dtype=torch.float32)

        xb_batch        = torch.zeros((batch_size, 3, image_height, image_width), dtype=torch.float32)
        yb_mask_batch   = torch.zeros((batch_size, 1, image_height, image_width), dtype=torch.float32)
        yb_kp_batch     = torch.zeros((batch_size, 1, image_height, image_width), dtype=torch.float32)

        pa_batch        = torch.zeros((batch_size, num_z_points, 2), dtype=int)
        pb_batch        = torch.zeros((batch_size, num_z_points, 2), dtype=int)


        for n in range(batch_size):
            xa, ya_mask, ya_kp, kpa,  xb, yb_mask, yb_kp, kpb, pa, pb = dataset.get((image_height, image_width), num_z_points=num_z_points)

            xa_batch[n]       = torch.from_numpy(xa)
            ya_mask_batch[n]  = torch.from_numpy(ya_mask)
            ya_kp_batch[n]    = torch.from_numpy(ya_kp)

            xb_batch[n]       = torch.from_numpy(xb)
            yb_mask_batch[n]  = torch.from_numpy(yb_mask)
            yb_kp_batch[n]    = torch.from_numpy(yb_kp)

            pa_batch[n]       = torch.from_numpy(pa)
            pb_batch[n]       = torch.from_numpy(pb)


        xa_batch        = xa_batch.to(self.device)
        ya_mask_batch   = ya_mask_batch.to(self.device)
        ya_kp_batch     = ya_kp_batch.to(self.device)

        xb_batch        = xb_batch.to(self.device)
        yb_mask_batch   = yb_mask_batch.to(self.device)
        yb_kp_batch     = yb_kp_batch.to(self.device)

        pa_batch        = pa_batch.to(self.device)
        pb_batch        = pb_batch.to(self.device)


        return xa_batch, ya_mask_batch, ya_kp_batch,  xb_batch, yb_mask_batch, yb_kp_batch, pa_batch, pb_batch


    def _extract_features(self, z, p):

        B, C, H, W = z.shape
        N = p.shape[1]

        # Clamp and get integer coordinates
        x = p[:, :, 0].clamp(0, W - 1).long()  # (B, N)
        y = p[:, :, 1].clamp(0, H - 1).long()  # (B, N)

        # Compute linear indices for flattened spatial dimensions (H * W)
        linear_idx = y * W + x  # (B, N)

        # Flatten feature map from (B, C, H, W) -> (B, C, H*W)
        z_flat = z.view(B, C, H * W)

        # Transpose to (B, H*W, C) for easier indexing
        z_flat = torch.transpose(z_flat, 1, 2)

        # Prepare batch indices
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, N)


        # Use advanced indexing to select the right features
        result = z_flat[batch_idx, linear_idx]  # (B, N, C)

        return result

    