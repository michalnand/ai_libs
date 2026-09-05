import numpy
import cv2
import torch
import random
import numpy
import random

from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 1. Create a persistent thread pool. 
# Adjust max_workers to match your CPU cores (or 2-4x cores for IO/CV tasks).
# Keeping this global or at the class-level means threads stay alive waiting for work.
SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=8) 


def _crop_single(x_tmp, min_dim, prob):
    """Worker function for a single image crop."""
    if numpy.random.rand() < prob:
        height = x_tmp.shape[1]
        width  = x_tmp.shape[2]

        min_original_dim = min(height, width)
        s_min = min(1.0, min_dim / min_original_dim)
        scale = random.uniform(s_min, 1.0)
        
        crop_h = int(round(height * scale))
        crop_w = int(round(width * scale))
        
        crop_h = min(height, max(1, crop_h))
        crop_w = min(width, max(1, crop_w))

        y_start = random.randint(0, height - crop_h)
        x_start = random.randint(0, width - crop_w)

        y_end   = y_start + crop_h
        x_end   = x_start + crop_w  

        # Slice and copy to ensure a clean contiguous array
        y = x_tmp[:, y_start:y_end, x_start:x_end]
        return numpy.array(y)
    else:
        return numpy.array(x_tmp)

def crop_augmentation(x: list, min_dim: int = 64, prob: float = 0.5):
    # partial binds the extra arguments so map() can just pass the image
    worker = partial(_crop_single, min_dim=min_dim, prob=prob)
    
    # map distributes the list across the active threads in the pool
    return list(SHARED_EXECUTOR.map(worker, x))


def _resize_single(x_tmp, width, height):
    """Worker function for a single image resize."""
    x_tmp = numpy.moveaxis(x_tmp, 0, 2)
    y = cv2.resize(x_tmp, (width, height))
    return numpy.moveaxis(y, 2, 0)

def resize_augmentation(x: list, width: int, height: int):
    worker = partial(_resize_single, width=width, height=height)
    return list(SHARED_EXECUTOR.map(worker, x))



def photometric_augmentations(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    B, C, H, W = x.shape
    device = x.device 

    # 1. Random Brightness
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    factors = torch.empty(B, 1, 1, 1, device=device).uniform_(0.1, 2.0)
    x = x * torch.where(mask, factors, 1.0)

    # 2. Random Contrast (centered around per-channel mean)
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    factors = torch.empty(B, 1, 1, 1, device=device).uniform_(0.25, 2.0)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    contrast_x = (x - mean) * factors + mean
    x = torch.where(mask, contrast_x, x)


    # 3. Random Gamma Correction
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    gammas = torch.empty(B, 1, 1, 1, device=device).uniform_(0.5, 2.0)
    gamma_x = torch.clamp(x, 0.0, 1.0) ** gammas
    x = torch.where(mask, gamma_x, x)

    # 4. Random Saturation (RGB to Grayscale blend)
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    factors = torch.empty(B, 1, 1, 1, device=device).uniform_(0.0, 2.0)
    gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    sat_x = factors * x + (1.0 - factors) * gray
    x = torch.where(mask, sat_x, x)

    # 5. Random Channel Permutation
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    perm_indices = torch.rand(B, C, device=device).argsort(dim=1)
    perm_indices = perm_indices.view(B, C, 1, 1).expand(B, C, H, W)
    perm_x = torch.gather(x, dim=1, index=perm_indices)
    x = torch.where(mask, perm_x, x)

    # 6. Random Channel Inversion
    inv_mask = torch.rand(B, C, 1, 1, device=device) < p
    x = torch.where(inv_mask, 1.0 - x, x)

    # 7. Additive Gaussian Noise
    mask = torch.rand(B, 1, 1, 1, device=device) < p
    sigmas = torch.empty(B, 1, 1, 1, device=device).uniform_(0.01, 0.05)
    noise = torch.randn_like(x) * sigmas
    x = x + torch.where(mask, noise, 0.0)
    
    return torch.clamp(x, 0.0, 1.0)






def generate_affine_matrices(batch_size):
    # 1. Scale [0.5, 2.0]
    scale_x = torch.empty(batch_size).uniform_(0.5, 2.0)
    scale_y = torch.empty(batch_size).uniform_(0.5, 2.0)
    
    M_scale = torch.zeros((batch_size, 3, 3))
    M_scale[:, 0, 0] = scale_x
    M_scale[:, 1, 1] = scale_y
    M_scale[:, 2, 2] = 1.0  

    # 2. Shear [-0.25, 0.25]
    sh_x = torch.empty(batch_size).uniform_(-0.25, 0.25)
    sh_y = torch.empty(batch_size).uniform_(-0.25, 0.25)

    M_shear = torch.zeros((batch_size, 3, 3))
    M_shear[:, 0, 0] = 1.0
    M_shear[:, 0, 1] = sh_x
    M_shear[:, 1, 0] = sh_y
    M_shear[:, 1, 1] = 1.0
    M_shear[:, 2, 2] = 1.0

    # 3. Rotation [-pi, pi]
    angles = torch.empty(batch_size).uniform_(-torch.pi, torch.pi)
    cos, sin = torch.cos(angles), torch.sin(angles)

    M_rot = torch.zeros((batch_size, 3, 3))
    M_rot[:, 0, 0] = cos
    M_rot[:, 0, 1] = -sin
    M_rot[:, 1, 0] = sin
    M_rot[:, 1, 1] = cos
    M_rot[:, 2, 2] = 1.0
 
    # 4. Translation / Shift [-0.25, 0.25] (Normalized grid coordinates)
    tx = torch.empty(batch_size).uniform_(-0.25, 0.25)
    ty = torch.empty(batch_size).uniform_(-0.25, 0.25)

    M_trans = torch.eye(3).repeat(batch_size, 1, 1)
    M_trans[:, 0, 2] = tx 
    M_trans[:, 1, 2] = ty

    # Chain transformations: M = M_trans @ M_shear @ M_rot @ M_scale
    M = torch.bmm(M_trans, torch.bmm(M_shear, torch.bmm(M_rot, M_scale)))
    
    # Compute inverse matrix for image warping
    M_inv = torch.linalg.inv(M)

    return M, M_inv


def affine_augmentation(x0, M_inv):
    # Slice to 2x3 affine tensor
    theta = M_inv[:, :2, :]
    
    # Generate spatial sampling grid in normalized range [-1, 1]
    grid = torch.nn.functional.affine_grid(theta, x0.size(), align_corners=False)
    
    # Warp images
    x1 = torch.nn.functional.grid_sample(x0, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return x1 