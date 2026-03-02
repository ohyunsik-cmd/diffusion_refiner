import torch
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from data.crop_shim import rescale_and_crop


def apply_fxfy_scale(K: torch.Tensor, fx_scale: float, fy_scale: float):
    K2 = K.clone()
    K2[0,0] = K2[0,0] * fx_scale
    K2[1,1] = K2[1,1] * fy_scale
    return K2
def estimate_fxfy_from_reproj(w2c34: torch.Tensor, K: torch.Tensor,
                         wp: torch.Tensor, mask: torch.Tensor):
    """
    Find fx_eff, fy_eff such that:
    xs ≈ fx_eff * (X/Z) + cx
    ys ≈ fy_eff * (Y/Z) + cy
    using least squares on masked pixels.
    """
    device = wp.device
    ys, xs = torch.where(mask)
    pts = wp[ys, xs]  # [N,3]

    R = w2c34[:3, :3]
    t = w2c34[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    xs = xs[valid].float()
    ys = ys[valid].float()

    # (중요) pixel-center convention 보정: VGGT가 center 기준이면 이게 더 맞는 경우 많음
    xs = xs + 0.5
    ys = ys + 0.5

    X = cam[:, 0]
    Y = cam[:, 1]
    Z = cam[:, 2]

    cx = K[0, 2]
    cy = K[1, 2]

    rx = X / (Z + 1e-8)
    ry = Y / (Z + 1e-8)

    # LS: (xs-cx) ≈ fx * rx  => fx = ( (xs-cx)·rx ) / (rx·rx)
    fx_eff = ((xs - cx) * rx).mean() / (rx * rx).mean().clamp(min=1e-8)
    fy_eff = ((ys - cy) * ry).mean() / (ry * ry).mean().clamp(min=1e-8)

    return fx_eff, fy_eff

def interp_K_log(K0, K1, t):
    K = K0.clone()
    # log-interp focal
    K[0,0] = torch.exp((1-t)*torch.log(K0[0,0]) + t*torch.log(K1[0,0]))
    K[1,1] = torch.exp((1-t)*torch.log(K0[1,1]) + t*torch.log(K1[1,1]))
    # principal point는 보통 선형 보간 (혹은 그냥 168로 고정)
    K[0,2] = (1-t)*K0[0,2] + t*K1[0,2]
    K[1,2] = (1-t)*K0[1,2] + t*K1[1,2]
    K[2,2] = 1.0
    return K


def cam_center_from_w2c(w2c_3x4):
    R = w2c_3x4[:3,:3]
    t = w2c_3x4[:3,3]
    return -(R.T @ t)

def interp_ratio_from_gt_centers(w2c_gt0, w2c_gt1, w2c_gtT):
    C0 = cam_center_from_w2c(w2c_gt0)
    C1 = cam_center_from_w2c(w2c_gt1)
    CT = cam_center_from_w2c(w2c_gtT)
    v = C1 - C0
    denom = (v @ v).clamp(min=1e-9)
    t = ((CT - C0) @ v) / denom
    return float(t.clamp(0.0, 1.0))
    
