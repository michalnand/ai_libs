import numpy
import torch

from .metric import *

class SegmentationPipeline:


    def __init__(self, Model, dataset, device = "cpu"):
        self.dataset    = dataset
        self.device     = device

        self.input_shape = self.dataset.input_shape

        self.batch_size   = 32
        n_outputs         = 1

        self.learning_rate = 0.001
        
        self.model = Model(self.input_shape, n_outputs)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate)

        print(self.model)
        


    def step(self):
        xa, y_gt = self._sample_batch(self.dataset, self.batch_size)

        
        y_pred = self.model(xa)

        # rescale prediction to original size
        sf = xa.shape[-1]//y_pred.shape[-1]
        y_pred    = torch.nn.functional.interpolate(y_pred, scale_factor=sf, mode='nearest')

        # classification loss for mask and keypoints
        loss_func = torch.nn.BCEWithLogitsLoss()

        loss = loss_func(y_pred, y_gt)


        # optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log loss and metric
        result_log = {}
        result_log["loss"] = float(loss.item())
        
        seg_metric = compute_segmentation_metrics(y_gt, torch.sigmoid(y_pred))

        result_log["seg_metric"] = seg_metric

        return result_log
        

    
    def _sample_batch(self, dataset, batch_size = 32):
        x_batch     = torch.zeros((batch_size, 3, self.input_shape[1], self.input_shape[2]), dtype=torch.float32)
        y_gt_batch  = torch.zeros((batch_size, 1, self.input_shape[1], self.input_shape[2]), dtype=torch.float32)


        for n in range(batch_size):
            idx = numpy.random.randint(0, len(dataset))

            x, y = dataset.get(idx)

            x_batch[n]       = torch.from_numpy(x)
            y_gt_batch[n, 0] = torch.from_numpy(y).float()
           
        x_batch      = x_batch.to(self.device)
        y_gt_batch   = y_gt_batch.to(self.device)
      
        return x_batch, y_gt_batch


    