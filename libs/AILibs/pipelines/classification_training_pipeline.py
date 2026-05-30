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

class ClassificationTrainingPipeline:


    def __init__(self, config):
        self.config = config

    def run(self):
        optimizer = torch.optim.Adam(self.config.model.parameters(), lr=self.config.learning_rate)

        loss_func = torch.nn.CrossEntropyLoss()

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
                if hasattr(self.config, "class_balancer"):
                    idx = self.config.class_balancer.sample()
                else:
                    idx = numpy.random.randint(0, len(self.training_dataset))


                x, y = self.config.training_dataset[idx]

                if hasattr(self.config, "augmentations"):
                    x, y = self.config.augmentations(x, y)

                x_t = torch.from_numpy(x).float()

                y_t = torch.from_numpy(y).float()

                x_batch.append(x_t) 
                y_batch.append(y_t)

            x_batch = torch.stack(x_batch).to(self.config.device)
            y_batch = torch.stack(y_batch).long().to(self.config.device)

            time_stop = time.time()

            timing_stats["data_preparation"] = round(time_stop - time_start, 4)

            time_start = time.time()
            y_pred = self.config.model(x_batch)
            time_stop = time.time()

            timing_stats["forward_pass"] = round(time_stop - time_start, 4)

            
            time_start = time.time()

            if y_batch.shape[0] != y_pred.shape[0]:
                raise Exception("y_batch and y_pred shape is not matching, expected " + str(y_batch.shape) + " , got " + str(y_pred.shape))



            loss = loss_func(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_stop = time.time()

            timing_stats["backward_pass"] = round(time_stop - time_start, 4)


            time_start = time.time()

            y_batch = y_batch.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            metric = AILibs.metrics.classification_evaluation(y_batch, y_pred, self.config.num_classes)
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
            result_log["step"]          = n
            result_log["max_steps"]     = self.config.num_steps
            result_log["num_classes"]   = self.config.num_classes

            result_log["loss"]          = loss.detach().cpu().numpy().item()
            result_log["iou"]           = metric["macro_iou"]
            result_log["f1_score"]      = metric["macro_f1_score"]
            result_log["dice"]          = metric["macro_dice"]
            result_log["accuracy"]      = metric["accuracy"]

            result_log["timing"]    = timing_stats
            result_log["metric"]    = metric

            tmp = self.config.num_steps//10000
            if tmp < 1:
                tmp = 1
                
            if (n%tmp) == 0:
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

        if self.config.testing_datset is not None:
            self._testing_model(results_path, self.config.testing_datset, self.config.num_testing_samples)

    def _testing_model(self, results_path, dataset, num_testing_samples):

        print("\n\n\n")
        print("testing")

        indices      = []
        y_batch      = []
        y_pred_batch = []

        timing = []

        self.config.model.eval()

        for n in range(num_testing_samples):

            idx = numpy.random.randint(0, len(dataset))

            x, y = dataset[idx]

            x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.config.device)

            time_start = time.time()


            with torch.no_grad():
                y_pred = self.config.model(x_t)


            time_stop = time.time()

            timing.append(time_stop - time_start)

            indices.append(idx)
            y_batch.append(y)
            y_pred_batch.append(y_pred.squeeze(0).detach().cpu().numpy())  


        timing  = numpy.array(timing)

        indices = numpy.array(indices)
        y_batch = numpy.array(y_batch)
        y_pred_batch = numpy.array(y_pred_batch)


        metric = AILibs.metrics.classification_evaluation(y_batch, y_pred_batch, self.config.num_classes)


        # pipeline timing stats
        timing_stats = {}
        timing_stats["mean"] = round(timing.mean(), 6)
        timing_stats["std"]  = round(timing.std(), 6)
        timing_stats["p68"]  = round(numpy.percentile(timing, 68), 6)
        timing_stats["p95"]  = round(numpy.percentile(timing, 95), 6)
        timing_stats["p997"] = round(numpy.percentile(timing, 99.7), 6)
        timing_stats["max"]  = round(numpy.max(timing), 6)



        entropy_all, entropy_correct, entropy_incorrect = self._calculate_logits_entropy(y_batch, y_pred_batch)

        entropy = {}
        entropy["entropy"]                = float(entropy_all.mean())
        entropy["entropy_correct"]        = float(entropy_correct.mean())
        entropy["entropy_incorrect"]      = float(entropy_incorrect.mean())

        blind_spots = self._identify_model_blindspots(y_batch, y_pred_batch, indices, high_conf_thresh=0.8, low_conf_thresh=0.5)


        result_log = {}
        result_log["num_samples"]   = num_testing_samples
        result_log["num_classes"]   = self.config.num_classes

        result_log["iou"]           = metric["macro_iou"]
        result_log["f1_score"]      = metric["macro_f1_score"]
        result_log["dice"]          = metric["macro_dice"]
        result_log["accuracy"]      = metric["accuracy"]

        result_log["timing"]        = timing_stats
        result_log["metric"]        = metric
        result_log["entropy"]       = entropy
        result_log["blind_spots"]   = blind_spots


        log_str = str(json.dumps(result_log) + "\n")
        f = open(results_path + "/testing.log", "a+")
        f.write(log_str)
        f.close()

        print(log_str)

    
    def _np_to_list(self, x, dp):
        result = []
        for v in x:
            result.append(round(v.item(), dp))

        return result
    
    def _calculate_logits_entropy(self, y_batch, y_pred_batch):
        """
        Calculates the Shannon entropy of predictions from logits.
        
        Parameters:
        - y_batch: array of shape (num_samples,) containing true class indices
        - y_pred_batch: array of shape (num_samples, num_classes) containing raw logits
        """
        # 1. Stable Softmax to get probabilities
        # Subtracting the max per row prevents numerical overflow
        shift_logits = y_pred_batch - numpy.max(y_pred_batch, axis=1, keepdims=True)
        exps = numpy.exp(shift_logits)
        probs = exps / numpy.sum(exps, axis=1, keepdims=True)
        
        # 2. Calculate Shannon Entropy: -Sum(p * log(p))
        # We add a tiny epsilon to avoid log(0)
        epsilon = 1e-15
        entropy = -numpy.sum(probs * numpy.log(probs + epsilon), axis=1)
        
        # 3. Split by correctness for better histogram insights
        predicted_classes = numpy.argmax(y_pred_batch, axis=1)
        correct_mask = (predicted_classes == y_batch)
        
        entropy_correct = entropy[correct_mask]
        entropy_incorrect = entropy[~correct_mask]
        
        return entropy, entropy_correct, entropy_incorrect
    


    def _identify_model_blindspots(self, y_batch, y_pred_batch, indices, high_conf_thresh=0.8, low_conf_thresh=0.5):
        """
        Identifies 'Confidently Wrong' and 'Weakly Correct' samples.
        
        Parameters:
        - y_batch: (num_samples,) ground truth integer labels
        - y_pred_batch: (num_samples, num_classes) raw logits
        - indices: (num_samples,) original dataset indices
        - high_conf_thresh: Float, threshold above which a model is "highly confident"
        - low_conf_thresh: Float, threshold below which a model is "weakly confident"
        """
        # 1. Numerically stable softmax to get probabilities
        shift_logits = y_pred_batch - numpy.max(y_pred_batch, axis=1, keepdims=True)
        exps = numpy.exp(shift_logits)
        probs = exps / numpy.sum(exps, axis=1, keepdims=True)
        
        # 2. Get the predicted class and its associated confidence (probability)
        pred_classes = numpy.argmax(probs, axis=1)
        confidences = numpy.max(probs, axis=1)
        
        # 3. Create boolean masks for correctness
        is_correct = (pred_classes == y_batch)
        is_wrong = ~is_correct
        
        # Category 1: Confidently Wrong (Wrong prediction AND confidence >= high_conf_thresh)
        conf_wrong_mask = is_wrong & (confidences >= high_conf_thresh)
        conf_wrong_indices = indices[conf_wrong_mask]
        conf_wrong_scores = confidences[conf_wrong_mask]
        
        # Category 2: Weakly Correct (Correct prediction AND confidence <= low_conf_thresh)
        weak_correct_mask = is_correct & (confidences <= low_conf_thresh)
        weak_correct_indices = indices[weak_correct_mask]
        weak_correct_scores = confidences[weak_correct_mask]
        
        return {
            "confidently_wrong": {"indices": self._np_to_list(conf_wrong_indices, 0), "confidence": self._np_to_list(conf_wrong_scores, 3)},
            "weakly_correct": {"indices": self._np_to_list(weak_correct_indices, 0), "confidence": self._np_to_list(weak_correct_scores, 3)}
        }
