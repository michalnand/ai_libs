import json
import torch
import numpy
import time

from pathlib import Path

from .augmentations import *
from .loss_self_supervised import *

class ImageFeaturesPipeline:

    def __init__(self, config):
        self.dataset    = config.dataset

        self.num_steps  = config.num_steps
        self.batch_size = config.batch_size

        self.width      = config.width
        self.height     = config.height

        self.num_points = config.num_points
        self.model      = config.model

        # Auto-decide device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # loss term weights
        self.w_sim = config.w_sim
        self.w_ssl = config.w_ssl

        # create path if not exists
        self.result_path = Path(config.result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.result_path / "training_log.jsonl"
        # Clear/Create log file on startup
        with open(self.log_file, 'w') as f:
            pass 

        self.ssl_loss_func = SIGRegLoss()

        print(self.model)

    def run_training(self):
        # Calculate interval to save model 10 times during training
        save_interval = max(1, self.num_steps // 10)

        steps_per_second = 0.0
        k = 0.1
        for step in range(self.num_steps):

            if (step%10) == 0:
                log_enabled = True
            else:
                log_enabled = False

            time_start = time.time()
            x = self.dataset.get_batch(self.batch_size)
            metrics = self.train_batch(x, log_enabled)
            time_stop = time.time() 


            steps_per_second = (1.0 - k)*steps_per_second + k/(time_stop - time_start)

            # Add step counter to metrics
            if log_enabled:
                log_result={}
                log_result["step"]            = step
                log_result["step_per_second"] = round(steps_per_second, 2)

                log_result.update(metrics)

                # JSONL Logging: flush every line
                with open(self.log_file, 'a') as f:
                    str_out = json.dumps(log_result)
                    f.write(str_out + '\n')
                    print(str_out)

            # Save model 10x per training + at final step
            if (step > 0 and step % save_interval == 0) or step == self.num_steps - 1:
                model_save_path = self.result_path / f"model_step_{step}.pt"
                # Saves the entire model architecture + weights
                torch.save(self.model, model_save_path)


        model_save_path = self.result_path / f"model_final.pt"
        torch.save(self.model, model_save_path)


    def train_batch(self, x, log_enabled):
        batch_size = len(x)
        
        x = crop_augmentation(x, 32, 0.5)  

        # all images to fixed size
        x = resize_augmentation(x, self.width, self.height)

        print("x_uint = ", x.shape, x.dtype, x.min(), x.max())

        # convert x from uint8 to float32
        x = numpy.array(x/255.0, dtype=numpy.float32)
        x = torch.from_numpy(x).to(self.device)

        print("x_fp = ", x.shape, x.dtype, x.min(), x.max())

        print()

        # simple colors augmentation, contrast, noise
        x0 = photometric_augmentations(x)
        x1 = photometric_augmentations(x)

        # geometric augmentations
        M, M_inv = generate_affine_matrices(batch_size)
        x1 = affine_augmentation(x1, M_inv.to(self.device))

        x0 = x0.float().to(self.device) 
        x1 = x1.float().to(self.device)

        # obtain model features
        z0 = self.model(x0)
        z1 = self.model(x1)

       
        zs0, zs1, valid_mask = self._sample_matching_features(z0, z1, M.to(self.device), self.num_points)


        # similarity loss term (Fix: Average over channels before applying mask)
        diff = ((zs0 - zs1)**2).mean(dim=-1) # Shape: (B, K)
        valid_count = valid_mask.sum().clamp(min=1)
        loss_sim = (diff * valid_mask).sum() / valid_count

        # self supervised regularization (Fix: Apply ONLY to valid points, on both views)
        # zs0[valid_mask] flattens the valid points into shape (N_valid, C)
        valid_zs0 = zs0[valid_mask]
        valid_zs1 = zs1[valid_mask]

        if False:
            print("\n\n\n")
            
            print("x0     = ", x0.shape)
            print("x1     = ", x1.shape)
            print("z0     = ", z0.shape)
            print("z1     = ", z1.shape)
            print("dx_mag = ", ((x0 - x1)**2).mean())
            print("dz_mag = ", ((z0 - z1)**2).mean())

            
            print("zs0 = ", zs0.shape)
            print("zs1 = ", zs1.shape)
            print("dzs_mag = ", ((zs0 - zs1)**2).mean())

            print("valid_zs0  = ", valid_zs0.shape)
            print("valid_zs1  = ", valid_zs1.shape)
            print("d_valid_zs = ", ((valid_zs0 - valid_zs1)**2).mean())

            print("\n\n\n")

        
        # optional features projector, isolate self supervised loss
        if hasattr(self.model, "projector"):
            valid_zs0_proj = self.model.projector(valid_zs0)
            valid_zs1_proj = self.model.projector(valid_zs1)
        else:   
            valid_zs0_proj = valid_zs0
            valid_zs1_proj = valid_zs1


        loss_ssl = self.ssl_loss_func(valid_zs0_proj) + self.ssl_loss_func(valid_zs1_proj)

        loss = self.w_sim * loss_sim + self.w_ssl * loss_ssl

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics for logging
        if log_enabled:
            with torch.no_grad():
                # Mean, and Standard deviation across features
                z_mag = (valid_zs0**2).mean().item()
                z_std = valid_zs0.std(dim=0).mean().item()

                z_proj_mag = (valid_zs0_proj**2).mean().item()
                z_proj_std = valid_zs0_proj.std(dim=0).mean().item()
                
                # Ratio of points that fell inside the valid frame
                valid_ratio = valid_mask.float().mean().item()
                valid_count = valid_mask.sum().item()

                if valid_count > 1:
                    # Positive match similarity
                    pos_cos = torch.nn.functional.cosine_similarity(valid_zs0, valid_zs1, dim=-1).mean().item()
        
                    # Negative match similarity via random batch permutation
                    perm_idx = torch.randperm(valid_count, device=valid_zs0.device)
                    neg_cos = torch.nn.functional.cosine_similarity(valid_zs0, valid_zs1[perm_idx], dim=-1).mean().item()

                    # Pairs eulcidean distances
                    d_pos    = ((valid_zs0 - valid_zs1)**2).mean(dim=-1)
                    d_neg    = ((valid_zs0 - valid_zs1[perm_idx])**2).mean(dim=-1)

                    pos_dist        = d_pos.mean().item()
                    neg_dist        = d_neg.mean().item()
                    pos_dist_std    = d_pos.std().item()
                    neg_dist_std    = d_neg.std().item()  

                    spectrum        = self._compute_variance_spectrum(valid_zs0)
                else:
                    pos_cos         = 0.0
                    neg_cos         = 0.0

                    pos_dist        = 0.0 
                    neg_dist        = 0.0
                    pos_dist_std    = 0.0
                    neg_dist_std    = 0.0
                    spectrum        = {"exp_var_1s" : -1,  "exp_var_2s" : -1, "exp_var_3s" : -1}

            log_result = {
                "loss_total": round(loss.item(), 5),
                "loss_sim": round(loss_sim.item(), 5),
                "loss_ssl": round(loss_ssl.item(), 5),

                "z_mag": round(z_mag, 5),   
                "z_std": round(z_std, 5),

                "z_proj_mag": round(z_proj_mag, 5),   
                "z_proj_std": round(z_proj_std, 5), 

                "pos_cos": round(pos_cos, 5),
                "neg_cos": round(neg_cos, 5),

                "pos_dist": round(pos_dist, 5),
                "neg_dist": round(neg_dist, 5),

                "pos_dist_std": round(pos_dist_std, 5),
                "neg_dist_std": round(neg_dist_std, 5),

                **spectrum,

                "valid_ratio": round(valid_ratio, 3),
                "valid_count": round(valid_count, 3)
            }   
        else:
            log_result = None

        return log_result


    def _sample_matching_features(self, z0, z1, M, num_points):
        # z0, z1 shape: (B, C, H_f, W_f)
        B, C, H_f, W_f = z0.shape

        # 1. Sample random coordinates P0 in range [-0.7, 0.7] to keep points mostly in-bounds
        p0 = torch.empty((B, num_points, 2), device=z0.device).uniform_(-0.7, 0.7)

        # 2. Convert P0 to homogeneous coordinates (B, num_points, 3, 1)
        p0_hom = torch.ones((B, num_points, 3, 1), device=z0.device)
        p0_hom[:, :, :2, 0] = p0

        # 3. Project points to transformed coordinate space: P1 = M @ P0
        p1_hom = torch.bmm(M, p0_hom.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1)
        p1 = p1_hom[:, :, :2]

        # 4. Extract features using grid_sample (expects grid shape: B, 1, K, 2)
        grid_p0 = p0.unsqueeze(1)
        grid_p1 = p1.unsqueeze(1) 

        feat0 = torch.nn.functional.grid_sample(z0, grid_p0, align_corners=False).squeeze(2).permute(0, 2, 1) # (B, K, C)
        feat1 = torch.nn.functional.grid_sample(z1, grid_p1, align_corners=False).squeeze(2).permute(0, 2, 1) # (B, K, C)

        # 5. Mask out points that landed outside valid normalized image frame [-1, 1]
        valid_mask = (p1[:, :, 0].abs() <= 1.0) & (p1[:, :, 1].abs() <= 1.0) # (B, K)

        return feat0, feat1, valid_mask


    def _compute_variance_spectrum(self, z):
        """
        Computes the number of features required to explain 68%, 95%, and 99.7% 
        of the variance in a batch of representations.
        
        Args:
            z (torch.Tensor): Feature batch of shape (Batch_Size, Dimensions).
            
        Returns:
            dict: Number of dimensions required for each variance threshold.
        """
        z = z.detach()

        # 1. Center the batch data (mean of 0 along the batch dimension)
        z_centered = z - z.mean(dim=0, keepdim=True)

        # 2. Compute singular values (S is returned in descending order)
        # Using svdvals is much faster than torch.svd or torch.linalg.svd
        S = torch.linalg.svdvals(z_centered)
        
        # 3. Variance is proportional to the square of singular values (eigenvalues)
        eigenvalues = S ** 2
        
        # 4. Compute cumulative explained variance ratio
        cum_var_ratio = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
        
        # 5. Helper function to find the minimum number of components for a threshold
        def get_n_features(threshold: float) -> int:
            # Find the first index where cumulative variance is >= threshold
            idx = (cum_var_ratio >= threshold).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                return idx[0].item() + 1 # +1 because indices are 0-based
            return z.shape[1] # Fallback to max dimensions if threshold isn't cleanly met
            
        return {
            "exp_var_1s": get_n_features(0.68),   # 68% variance
            "exp_var_2s": get_n_features(0.95),   # 95% variance
            "exp_var_3s": get_n_features(0.997)   # 99.7% variance
        }