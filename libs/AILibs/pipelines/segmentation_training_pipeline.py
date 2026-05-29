import os
import time
import json
import numpy
import torch
import AILibs



"""
    config mus contain : 

    config.results_path         = path to reuslts, saving models
    config.device               = "cpu" or "cuda"
    config.model                = fully CNN model
    config.augmentations        = augmentations pipeline
    config.num_steps            = num training steps, one step = one batch processed
    config.batch_size           = batch size typical : 4 .. 64
    config.training_dataset     = instance of dataset
    config.training_dataset     = instance of dataset
"""

class SegmentationTrainingPipeline:


    def __init__(self, config):
        self.config = config

    def run(self):
        optimizer = torch.optim.Adam(self.config.model.parameters(), lr=self.config.learning_rate)

        loss_func = torch.nn.BCEWithLogitsLoss()

        results_path = self.config.results_path
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        f = open(results_path + "/training.log", "w")
        f.close()

        steps_sec = 0.0

        saving_interval = self.config.num_steps//10
        saving_version  = 0
        
        for n in range(self.config.num_steps):
            timing_stats = {}

            x_batch = []
            y_batch = []

            time_main_start = time.time()

            time_start = time.time()
            for j in range(self.config.batch_size):
                idx = numpy.random.randint(0, len(self.config.training_dataset))
                x, y = self.config.training_dataset[idx]
                x, y = self.config.augmentations(x, y)

                x_t = torch.from_numpy(x).float()


                # add channel dim if only matrix is present
                y_t = torch.from_numpy(y).float()


                x_batch.append(x_t) 
                y_batch.append(y_t)

            x_batch = torch.stack(x_batch).to(self.config.device)
            y_batch = torch.stack(y_batch).to(self.config.device)

            time_stop = time.time()

            timing_stats["data_preparation"] = round(time_stop - time_start, 4)

            time_start = time.time()
            y_pred = self.config.model(x_batch)
            time_stop = time.time()

            timing_stats["forward_pass"] = round(time_stop - time_start, 4)

            
            time_start = time.time()

            if y_batch.shape != y_pred.shape:
                raise Exception("y_batch and y_pred shape is not matching, expected " + str(y_batch.shape) + " , got " + str(y_pred.shape))

            loss = loss_func(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_stop = time.time()

            timing_stats["backward_pass"] = round(time_stop - time_start, 4)


            time_start = time.time()

            y_batch = y_batch.detach().cpu().numpy()
            y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()

            metric = AILibs.metrics.detection_evaluation( y_batch, y_pred)
            time_stop = time.time()

            timing_stats["metric_processing"] = round(time_stop - time_start, 4)

            time_main_stop = time.time()


            timing_stats["batch_time"] = round(time_main_stop - time_main_start, 4)
            timing_stats["imgs/sec"]   = round(self.config.batch_size*1.0/(time_main_stop - time_main_start), 2)

            k = 0.9
            steps_sec = k*steps_sec + (1.0 - k)*1.0/(time_main_stop - time_main_start)

            timing_stats["steps/sec"]   = round(steps_sec, 2)   
            timing_stats["eta"]         = round((self.config.num_steps - n)/(steps_sec + 1e-10), 1)


            result_log = {}
            result_log["step"]      = n
            result_log["max_steps"] = self.config.num_steps

            result_log["loss"]      = loss.detach().cpu().numpy().item()
            result_log["iou"]       = metric["iou"]
            result_log["f1_score"]  = metric["f1_score"]
            result_log["dice"]      = metric["dice"]

            result_log["timing"]    = timing_stats
            result_log["metric"]    = metric


            log_str = str(json.dumps(result_log) + "\n")
            f = open(results_path + "/training.log", "a+")
            f.write(log_str)
            f.close()
            
            print(log_str)

            if n%saving_interval == 0:
                file_name = results_path + "/model_" + str(saving_version) + ".pt"
                print("saving model to ", file_name)
                
                torch.save(self.config.model, file_name)
                saving_version+= 1
                
                print("saving done\n\n")