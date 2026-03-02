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

# =========================
# Config
# =========================
DATASET_ROOT = Path("/mnt/hdd1/yunsik/re10k")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
IMAGE_SIZE = 336
out_w = 336
out_h = 336
Z_CLIP = 0.2

USE_POINT_CONF_MASK = True
CONF_THRESH = 0.0
USE_IMAGE_MASK = True # letterbox/black border 제거

# =========================
# Utils: pose / sim3
# =========================

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

def w2c_from_pred_variant(w2c34):
    # variant 1: as-is (현재 네 방식)
    v1 = w2c34

    # variant 2: invert (c2w일 가능성)
    v2 = invert_34(w2c34)

    # variant 3: transpose-rotation + fix t (컨벤션 실수 가능성)
    R = w2c34[:3,:3]
    t = w2c34[:3,3]
    v3 = w2c34.clone()
    v3[:3,:3] = R.T
    v3[:3,3]  = -R.T @ t

    # variant 4: invert(v3)
    v4 = invert_34(v3)

    return [v1, v2, v3, v4]

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

def interp_K_linear(K0, K1, t):
    K = K0.clone()
    K[0,0] = (1-t)*K0[0,0] + t*K1[0,0]  # fx
    K[1,1] = (1-t)*K0[1,1] + t*K1[1,1]  # fy

    return K

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

def T34_to_T44(T34: torch.Tensor) -> torch.Tensor:
    T44 = torch.eye(4, device=T34.device, dtype=T34.dtype)
    T44[:3, :4] = T34
    return T44

def invert_34(T34: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(T34_to_T44(T34))[:3, :4]

def camera_center_from_w2c(w2c_3x4: torch.Tensor) -> torch.Tensor:
    # w2c: X_cam = R X_world + t
    R = w2c_3x4[:3, :3]
    t = w2c_3x4[:3, 3]
    # camera center in world: C = -R^T t
    return -(R.transpose(0, 1) @ t)

def closest_rotation(M: torch.Tensor) -> torch.Tensor:
    # project to SO(3) via SVD
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def estimate_sim3_gt_to_pred(
    w2c_gt0: torch.Tensor, w2c_gt1: torch.Tensor,
    w2c_pr0: torch.Tensor, w2c_pr1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GT world -> Pred(VGGT) world similarity:
      X_pred = s * R_sim * X_gt + t_sim
    Robust with just 2 cams by using camera orientations + baseline ratio.
    """
    # centers
    Cg0 = camera_center_from_w2c(w2c_gt0)
    Cg1 = camera_center_from_w2c(w2c_gt1)
    Cp0 = camera_center_from_w2c(w2c_pr0)
    Cp1 = camera_center_from_w2c(w2c_pr1)

    # scale from baseline ratio
    bg = torch.norm(Cg1 - Cg0).clamp(min=1e-8)
    bp = torch.norm(Cp1 - Cp0).clamp(min=1e-8)
    s = (bp / bg)

    # rotations: use c2w rotations (world orientation of camera)
    # w2c rotation is R_wc? actually w2c is R_cam_world, so c2w rotation is R_world_cam = R^T
    Rg0 = w2c_gt0[:3, :3].transpose(0, 1)
    Rg1 = w2c_gt1[:3, :3].transpose(0, 1)
    Rp0 = w2c_pr0[:3, :3].transpose(0, 1)
    Rp1 = w2c_pr1[:3, :3].transpose(0, 1)

    Rsim0 = Rp0 @ Rg0.transpose(0, 1)
    Rsim1 = Rp1 @ Rg1.transpose(0, 1)
    Rsim = closest_rotation(0.5 * (Rsim0 + Rsim1))

    # translation so that Cg0 maps to Cp0
    t = Cp0 - s * (Rsim @ Cg0)

    return s, Rsim, t

def transform_gt_w2c_to_pred_w2c(w2c_gt: torch.Tensor, s: torch.Tensor, Rsim: torch.Tensor, tsim: torch.Tensor) -> torch.Tensor:
    """
    Convert a GT camera (w2c) defined in GT world into Pred world using:
      X_pred = s Rsim X_gt + tsim

    Camera rotation: R_pred_c2w = Rsim * R_gt_c2w
    Camera center:   C_pred = s Rsim C_gt + tsim
    Then build w2c_pred from (R_pred_c2w, C_pred).
    """
    R_gt_c2w = w2c_gt[:3, :3].transpose(0, 1)
    C_gt = camera_center_from_w2c(w2c_gt)

    R_pr_c2w = Rsim @ R_gt_c2w
    C_pr = s * (Rsim @ C_gt) + tsim

    # build w2c: R = (c2w)^T, t = -R * C
    R_pr_w2c = R_pr_c2w.transpose(0, 1)
    t_pr = -(R_pr_w2c @ C_pr)

    w2c_pr = torch.zeros((3, 4), device=w2c_gt.device, dtype=w2c_gt.dtype)
    w2c_pr[:3, :3] = R_pr_w2c
    w2c_pr[:3, 3] = t_pr
    return w2c_pr

# =========================
# Dataset loader (same as you)
# =========================
to_tensor = tf.ToTensor()
resize = tf.Resize((IMAGE_SIZE, IMAGE_SIZE))

def load_chunk_sample(chunk_path, sample_idx=0):
    chunk = torch.load(chunk_path, weights_only=True)
    item = chunk[sample_idx]
    cameras = item["cameras"]
    num_views = cameras.shape[0]

    def convert_poses(poses):
        b, _ = poses.shape
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
        w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
        return w2c, intrinsics

    extrinsics, intrinsics = convert_poses(cameras)

    # 1) images tensor로 만들기 (원본 해상도 그대로)
    images = []
    for i in range(num_views):
        img = item["images"][i]
        img_pil = Image.open(BytesIO(img.numpy().tobytes())).convert("RGB")
        images.append(to_tensor(img_pil))
    images = torch.stack(images)  # [V,3,H,W] (H,W는 원본)

    '''
    H0, W0 = images.shape[-2], images.shape[-1]
    intrinsics_px = intrinsics.clone()
    intrinsics_px[:, 0, 0] *= W0
    intrinsics_px[:, 1, 1] *= H0
    intrinsics_px[:, 0, 2] *= W0
    intrinsics_px[:, 1, 2] *= H0
    '''

    images, intrinsics_px = rescale_and_crop(
        images, intrinsics, (out_h, out_w)
    )

    return {
        "extrinsics": extrinsics,        # [V,4,4]
        "intrinsics": intrinsics_px,     # [V,3,3]  (이제 336x336 pixel K)
        "images": images,                # [V,3,336,336]
        "scene": item["key"],
    }

def select_views(num_views, min_gap=50, max_gap=150):
    max_gap = min(num_views - 1, max_gap)
    min_gap = max(2, min_gap)
    if max_gap < min_gap:
        min_gap = max(2, max_gap // 2)

    gap = np.random.randint(min_gap, max_gap + 1)
    left = np.random.randint(0, num_views - gap)
    right = left + gap
    target = np.random.randint(left + 1, right)
    return [left, right], target

# =========================
# Render: vectorized z-buffer (near-first)
# =========================
def render_pointcloud_zbuffer(pts_world: torch.Tensor, rgb: torch.Tensor, w2c_tgt: torch.Tensor, K_tgt: torch.Tensor, H: int, W: int):
    """
    pts_world: [N,3]  (in Pred/VGGT world)
    rgb:       [N,3]  (0..1)
    w2c_tgt:   [3,4]  (world->cam)
    K_tgt:     [3,3]
    returns rendered [3,H,W]
    """
    
    def camera_center_from_w2c(w2c: torch.Tensor):
        R = w2c[:3,:3]
        t = w2c[:3,3]
        C = -R.transpose(0,1) @ t
        return C

    C = camera_center_from_w2c(w2c_tgt)
    print("C:", C.tolist(), "||C||:", float(C.norm()))

    R = w2c_tgt[:3, :3]
    t = w2c_tgt[:3, 3]

    cam = pts_world @ R.transpose(0, 1) + t  # row-vector: Xcam = Xw R^T + t
    z = cam[:, 2]
    valid = z > 1e-6  # z가 너무 작으면 수치적으로 불안정하므로 제거 (카메라 뒤에 있거나 너무 가까운 점)
    zv = z[valid]

    if zv.numel() > 64:
        z_floor = torch.quantile(zv, 0.02).clamp(min=0.2)  # 데이터에 맞춰 0.2 유지
    else:
        z_floor = torch.tensor(0.2, device=z.device)
        
    valid = valid & (z >= z_floor)
    if valid.sum() < 10:
        return torch.zeros((3, H, W), device=pts_world.device)
    
    cam_v = cam[valid]
    z = cam_v[:, 2]
    
    x = cam_v[:, 0] / (z + 1e-8) * K_tgt[0, 0] + K_tgt[0, 2]
    y = cam_v[:, 1] / (z + 1e-8) * K_tgt[1, 1] + K_tgt[1, 2]
    
    print("[render debug]",
          "N=", pts_world.shape[0],
          "z(min/mean/max)=", float(z.min()), float(z.mean()), float(z.max()),
          "valid=", int(valid.sum()), "/", int(valid.numel()),
          "x(min/max)=", float(x.min()), float(x.max()),
          "y(min/max)=", float(y.min()), float(y.max()),
          "rgb(min/max)=", float(rgb.min()), float(rgb.max()))
    
    if not valid.any():
        return torch.zeros((3, H, W), device=pts_world.device)

    in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if in_img.sum() < 10:
        return torch.zeros((3, H, W), device=pts_world.device)
    
    x_i = x[in_img].long()
    y_i = y[in_img].long()
    z_i = z[in_img].float()
    c_i = rgb[valid][in_img].float() 

    flat = (y_i * W + x_i).long()

    order = torch.argsort(z_i)          # near -> far
    flat_s = flat[order]
    col_s  = c_i[order]

    keep = torch.ones_like(flat_s, dtype=torch.bool)
    keep[1:] = flat_s[1:] != flat_s[:-1]  # unique_consecutive

    flat_k = flat_s[keep]
    col_k  = col_s[keep]

    out = torch.zeros((H * W, 3), device=pts_world.device, dtype=col_k.dtype)
    out[flat_k] = col_k
    return out.view(H, W, 3).permute(2, 0, 1)  # [3,H,W]

# =========================
# Main
# =========================
print("Loading VGGT pretrained model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
print("Model loaded!")

print("Loading samples from RE10K dataset...")

for sample_num in range(10):
    chunk_idx = np.random.randint(0, 500)
    chunk_path = DATASET_ROOT / "train" / f"{chunk_idx:06d}.torch"
    data = load_chunk_sample(chunk_path, sample_idx=0)

    V = data["images"].shape[0]
    np.random.seed(sample_num)
    context_indices, target_index = select_views(V)

    print(f"\n{'='*60}")
    print(f"SAMPLE {sample_num} (chunk {chunk_idx}) scene={data['scene']}")
    print(f"context={context_indices}, target={target_index}")
    print(f"{'='*60}")

    # 1) VGGT inference on context views only
    context_images = data["images"][context_indices].unsqueeze(0).to(DEVICE)  # [1,2,3,H,W]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=DTYPE):
            pred = model(context_images)
            
    pose0 = pred["pose_enc"][0,0].float()          # [9]
    w2c0, K0 = pose_encoding_to_extri_intri(pred["pose_enc"][:, :1].float(), (out_h,out_w), build_intrinsics=True)
    K0 = K0[0,0].float()
    w2c0 = w2c0[0,0].float()

    # 2) Decode predicted w2c + K for the 2 context views
    # pred["pose_enc"]: [1,2,9]
    w2c_pred, K_pred = pose_encoding_to_extri_intri(pred["pose_enc"], (out_h, out_w), build_intrinsics=True)
    w2c_pred0 = w2c_pred[0, 0].float()  # [3,4] w2c
    w2c_pred1 = w2c_pred[0, 1].float()
    K_pred0   = K_pred[0, 0].float()
    K_pred1   = K_pred[0, 1].float()

    # 3) Use VGGT world_points directly (no unproject, no GT pose mixing)
    wp0 = pred["world_points"][0, 0].float()  # [H,W,3]
    wp1 = pred["world_points"][0, 1].float()  # [H,W,3]
    conf0 = pred["world_points_conf"][0, 0].float()  # [H,W]
    conf1 = pred["world_points_conf"][0, 1].float()

    # colors from images (same pixel indices)
    img0 = context_images[0, 0].permute(1, 2, 0).float()  # [H,W,3]
    img1 = context_images[0, 1].permute(1, 2, 0).float()


    # optional: remove black border areas
    if USE_IMAGE_MASK:
        mask_img0 = img0.mean(-1) > 0.01
        mask_img1 = img1.mean(-1) > 0.01
    else:
        mask_img0 = torch.ones((out_h, out_w), device=DEVICE, dtype=torch.bool)
        mask_img1 = torch.ones((out_h, out_w), device=DEVICE, dtype=torch.bool)

    if USE_POINT_CONF_MASK:
        m0 = (conf0 > CONF_THRESH) & mask_img0
        m1 = (conf1 > CONF_THRESH) & mask_img1
    else:
        m0 = mask_img0
        m1 = mask_img1

    pts0 = wp0[m0]           # [N0,3]
    pts1 = wp1[m1]           # [N1,3]
    col0 = img0[m0]          # [N0,3]
    col1 = img1[m1]          # [N1,3]

    fx_eff0, fy_eff0 = estimate_fxfy_from_reproj(w2c_pred0, K_pred0, wp0, m0)
    fx_scale0 = float(fx_eff0 / K_pred0[0,0])
    fy_scale0 = float(fy_eff0 / K_pred0[1,1])
    print("fx_scale0, fy_scale0:", fx_scale0, fy_scale0)
    
    fx_eff1, fy_eff1 = estimate_fxfy_from_reproj(w2c_pred1, K_pred1, wp1, m1)
    fx_scale1 = float(fx_eff1 / K_pred1[0,0])
    fy_scale1 = float(fy_eff1 / K_pred1[1,1])

    fx_scale = 0.5*(fx_scale0 + fx_scale1)
    fy_scale = 0.5*(fy_scale0 + fy_scale1)
    print("fx_scale, fy_scale:", fx_scale, fy_scale)

    all_pts = torch.cat([pts0, pts1], dim=0)
    all_col = torch.cat([col0, col1], dim=0)
    print(f"pointcloud: {all_pts.shape[0]} points (conf>{CONF_THRESH}={USE_POINT_CONF_MASK})")

    # 4) Convert GT target pose -> Pred(VGGT) world using Sim(3) estimated from 2 context cameras
    # GT w2c from dataset (context0, context1, target)
    w2c_gt0 = data["extrinsics"][context_indices[0]].to(DEVICE)[:3, :4].float()
    w2c_gt1 = data["extrinsics"][context_indices[1]].to(DEVICE)[:3, :4].float()
    w2c_gtT = data["extrinsics"][target_index].to(DEVICE)[:3, :4].float()
    

    s, Rsim, tsim = estimate_sim3_gt_to_pred(w2c_gt0, w2c_gt1, w2c_pred0, w2c_pred1)
    w2c_tgt_in_pred = transform_gt_w2c_to_pred_w2c(w2c_gtT, s, Rsim, tsim).float()

    # intrinsics: 그냥 GT target intrinsics(이미 픽셀로 스케일됨) 사용
    K_tgt = data["intrinsics"][target_index].to(DEVICE).float()
    
    
    print("K_pred1: ", K_pred1)
    print("k_pred0: ", K_pred0)
    
    t = interp_ratio_from_gt_centers(w2c_gt0, w2c_gt1, w2c_gtT)
    K_tgt = interp_K_log(K_pred0, K_pred1, t)
    K_tgt = apply_fxfy_scale(K_tgt, fx_scale, fy_scale)                                     
    
    def reproj_error_stats(w2c34, K, wp, mask, H, W):
        # wp: [H,W,3], mask: [H,W] bool
        ys, xs = torch.where(mask)
        pts = wp[ys, xs]  # [N,3]

        # project
        R = w2c34[:3,:3]
        t = w2c34[:3,3]
        cam = pts @ R.T + t
        z = cam[:,2]
        valid = z > 1e-6
        cam = cam[valid]
        xs = xs[valid].float()
        ys = ys[valid].float()
        z = cam[:,2]

        u = cam[:,0] / (z + 1e-8) * K[0,0] + K[0,2]
        v = cam[:,1] / (z + 1e-8) * K[1,1] + K[1,2]

        # error to original pixel coords
        du = (u - xs).abs()
        dv = (v - ys).abs()
        return {
            "N": int(u.numel()),
            "med_du": float(du.median()),
            "med_dv": float(dv.median()),
            "mean_du": float(du.mean()),
            "mean_dv": float(dv.mean()),
        }

    # 4 variants 
    variants = w2c_from_pred_variant(w2c_pred0)

    for i, w2c_v in enumerate(variants):
        st = reproj_error_stats(w2c_v, K_pred0, wp0, m0, out_h, out_w)
        print(f"[reproj v{i+1}] N={st['N']} med(|du|)={st['med_du']:.3f} med(|dv|)={st['med_dv']:.3f} "
            f"mean(|du|)={st['mean_du']:.3f} mean(|dv|)={st['mean_dv']:.3f}")

    print("K_GT target:", K_tgt)

    # 5) Render to target (GT->Pred transformed pose)
    rendered = render_pointcloud_zbuffer(all_pts, all_col, w2c_tgt_in_pred, K_tgt, out_h, out_w)

    img = rendered  # [3,H,W]
    filled = (img.sum(0, keepdim=True) > 0).float()  # [1,H,W]

    # 주변 3x3에서 가장 가까운 색 가져오기 (간단 dilation)
    img2 = F.max_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    filled2 = F.max_pool2d(filled.unsqueeze(0), 3, 1, 1)[0]

    # 비어있는 픽셀만 채우기
    out = img.clone()
    out[:, filled[0] == 0] = img2[:, filled[0] == 0]
    rendered = out

    rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)


    rendered_np = rendered.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    # visuals
    ctx0_np = context_images[0, 0].permute(1, 2, 0).detach().cpu().numpy()
    ctx1_np = context_images[0, 1].permute(1, 2, 0).detach().cpu().numpy()
    tgt_np  = data["images"][target_index].permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(ctx0_np); axes[0].set_title("Context 0"); axes[0].axis("off")
    axes[1].imshow(ctx1_np); axes[1].set_title("Context 1"); axes[1].axis("off")
    axes[2].imshow(tgt_np);  axes[2].set_title("Target GT"); axes[2].axis("off")
    axes[3].imshow(rendered_np); axes[3].set_title(f"Rendered image by GT pose"); axes[3].axis("off")
    #axes[4].imshow(rendered_forward_np); axes[4].set_title(f"Rendered (alpha={alpha:.2f})"); axes[4].axis("off")
    

    plt.tight_layout()
    out_path = f"vggt_pc_render_{sample_num}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

print("\nDone!")