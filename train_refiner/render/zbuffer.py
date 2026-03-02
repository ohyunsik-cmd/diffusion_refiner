import torch
import torch.nn.functional as F

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

@torch.no_grad()
def dilation_fill(image: torch.Tensor, mode="image_stats", sigma=0.2):
    """
    image: [3,H,W] in 0..1 (assumed)
    mode:
      - "dilation": 기존 방식(맥스풀)
      - "gray_noise": mean=0.5, std=sigma
      - "image_stats": 채워진 픽셀의 채널별 mean/std로 노이즈 생성 (추천)
    """
    img = image
    device = img.device
    C, H, W = img.shape

    filled = (img.sum(0, keepdim=True) > 0)   # [1,H,W] bool
    empty = ~filled[0]                        # [H,W] bool

    if empty.sum() == 0:
        return img

    if mode == "dilation":
        img2 = F.max_pool2d(img.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
        out = img.clone()
        out[:, empty] = img2[:, empty]
        return out

    # ---- noise fill ----
    if mode == "gray_noise":
        noise = torch.randn((C, H, W), device=device) * sigma + 0.5

    elif mode == "image_stats":
        # filled 픽셀이 너무 적으면 fallback
        if filled.sum() < 64:
            mean = torch.full((C, 1, 1), 0.5, device=device)
            std  = torch.full((C, 1, 1), sigma, device=device)
        else:
            vals = img[:, filled[0]]  # [C, N]
            mean = vals.mean(dim=1).view(C, 1, 1)
            std  = vals.std(dim=1, unbiased=False).clamp(min=1e-3).view(C, 1, 1)
        noise = torch.randn((C, H, W), device=device) * std + mean

    else:
        raise ValueError(f"Unknown mode: {mode}")

    noise = noise.clamp(0.0, 1.0)
    out = img.clone()
    out[:, empty] = noise[:, empty]
    return out