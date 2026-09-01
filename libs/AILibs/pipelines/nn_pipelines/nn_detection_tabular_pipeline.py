import AILibs
import numpy
import os

import torch
import json

from .nn_utils import *

        
class NNDetectionTabularPipeline:
    

    def __init__(self, config): 

        self.config = config

        self.dataset_train = config.dataset_train
        self.dataset_test  = config.dataset_test

        self.batch_size                 = config.batch_size
        self.num_steps                  = config.num_steps


        self.threshold          = config.threshold
        self.model              = config.model

        self.device = "cpu"

        self.result_path = config.result_path

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        print(self.model)

    def run(self):
        self._training()
        m, y_gt, y_pred = self._testing()
        self._save(m, y_gt, y_pred) 

    def _training(self):
        os.makedirs(self.result_path, exist_ok=True)

        f = open(self.result_path + "training.jsonl", "w")
        f.close()
                

        loss_func = torch.nn.BCEWithLogitsLoss() 

        for n in range(self.num_steps):

            x, y_gt = self._sample_random_batch(self.dataset_train, self.batch_size)

            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y_gt).float().to(self.device) 

            x_norm = (x - x.mean(dim=0, keepdim=True))/(x.std(dim=0, keepdim=True) + 1e-10)

            y_pred = self.model.forward(x_norm)
            y_pred = y_pred.squeeze(1)


            loss = loss_func(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # evaluation
            y_pred = torch.nn.functional.sigmoid(y_pred)
            y_pred = y_pred.detach().float().cpu().numpy()  
           
            metrics = AILibs.metrics.detection_evaluation(y_gt, y_pred, th=0.5)

            # log result
            log_result = {}
            log_result["step"]      = n
            log_result["loss"]      = loss.item()


            log_result.update(metrics)

            f = open(self.result_path + "training.jsonl", "a")
            f.write(json.dumps(log_result) + "\n")
            f.close()
            print(log_result)



    def _testing(self):
        y_gt_all   = []
        y_pred_all = []

        for n in range(1000):
            x, y_gt = self._sample_random_batch(self.dataset_test, self.batch_size)
            
            x = torch.from_numpy(x).float().to(self.device)

            x_norm = (x - x.mean(dim=0, keepdim=True))/(x.std(dim=0, keepdim=True) + 1e-10)

            y_hat = self.model.forward(x_norm)
            y_hat = y_hat.squeeze(1)
            y_hat = torch.nn.functional.sigmoid(y_hat)
            y_hat = y_hat.detach().cpu().numpy()


            y_gt_all.extend(list(y_gt))
            y_pred_all.extend(list(y_hat))
            
            
        y_gt   = numpy.array(y_gt_all) 
        y_pred = numpy.array(y_pred_all)

        

        if self.threshold is None:  
            self.threshold = AILibs.metrics.tune_threshold(y_gt, y_pred)
            print("autotune threshold to ", self.threshold)

        metrics = AILibs.metrics.anomaly_evaluation(y_gt, y_pred, th=self.threshold)
        return metrics, y_gt, y_pred


    def predict(self, x):
        return self.forest.score(x)


    def _save(self, metrics, y_gt, y_pred):

        # 2. Save the model, TODO


        nn_save_detector_prediction_distribution(self.result_path, y_gt, y_pred, threshold = self.threshold)
        nn_save_detector_cm(self.result_path, metrics)
        nn_save_detector_metrics(self.result_path, self.config, metrics)


    def _sample_random_batch(self, x_sampler, batch_size):
        if isinstance(x_sampler, AILibs.BatchSampler):
            x_batch, y_batch = x_sampler.sample(batch_size)
        elif isinstance(x_sampler, tuple) or isinstance(x_sampler, list):
            x_data, y_data = x_sampler[0], x_sampler[1]
            indices = numpy.random.randint(0, len(x_data), (batch_size, ))
            x_batch = x_data[indices]
            y_batch = y_data[indices]
        else:
            raise Exception("Unsupported input data type")
        
        return numpy.array(x_batch, dtype=numpy.float32), numpy.array(y_batch, dtype=numpy.float32)
