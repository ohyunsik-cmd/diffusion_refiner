import torch
import torch.nn.functional as F

def compute_big_mask_from_hit(
    hit: torch.Tensor,            # [H,W] bool
    win: int = 31,                # 윈도우 크기 (홀수 권장)
    min_fill_ratio: float = 0.15, # 윈도우 내 hit 비율이 이 이상이면 big으로
    smooth_iters: int = 1,        # 1~2 정도로 다듬기
):
    """
    빠른 근사 '큰 영역' 마스크:
    - k×k 평균풀로 local density를 구하고
    - density가 threshold 이상인 곳만 big으로 간주
    """
    H, W = hit.shape
    x = hit.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # local density
    pad = win // 2
    dens = F.avg_pool2d(x, kernel_size=win, stride=1, padding=pad)  # [1,1,H,W]
    big = (dens[0, 0] >= min_fill_ratio)

    # optional smoothing (팽창/침식 비슷한 효과)
    # big이 끊기는 걸 줄이려고 maxpool로 살짝 연결해줌
    for _ in range(smooth_iters):
        big = (F.max_pool2d(big.float().unsqueeze(0).unsqueeze(0),
                            kernel_size=3, stride=1, padding=1)[0, 0] > 0)

    return big  # [H,W] bool


def render_pointcloud_zbuffer(
    pts_world: torch.Tensor, rgb: torch.Tensor,
    w2c_tgt: torch.Tensor, K_tgt: torch.Tensor,
    H: int, W: int,
    # depth 컷
    z_quantile: float = 0.005,
    z_min: float = 0.02,
    # fill
    fill_mode: str = "dilation",   # "none"|"gray_noise"|"image_stats"|"dilation"
    fill_sigma: float = 0.4,
    # big mask params
    big_win: int = 31,
    big_min_fill_ratio: float = 0.15,
    big_smooth_iters: int = 1,
    return_hit: bool = False,
):
    device = pts_world.device
    R = w2c_tgt[:3, :3]
    t = w2c_tgt[:3, 3]

    cam = pts_world @ R.transpose(0, 1) + t
    z = cam[:, 2]

    valid = z > 1e-6
    zv = z[valid]

    if zv.numel() > 64:
        z_floor = torch.quantile(zv, z_quantile).clamp(min=z_min)
    else:
        z_floor = torch.tensor(z_min, device=device)

    valid = valid & (z >= z_floor)
    if valid.sum() < 10:
        out = torch.zeros((3, H, W), device=device)
        hit = torch.zeros((H, W), device=device, dtype=torch.bool)
        big = hit.clone()
        if fill_mode != "none":
            out = (torch.randn_like(out) * fill_sigma + 0.5).clamp(0, 1)
        return (out, big, hit) if return_hit else (out, big)

    cam_v = cam[valid]
    z = cam_v[:, 2]

    x = cam_v[:, 0] / (z + 1e-8) * K_tgt[0, 0] + K_tgt[0, 2]
    y = cam_v[:, 1] / (z + 1e-8) * K_tgt[1, 1] + K_tgt[1, 2]

    in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if in_img.sum() < 10:
        out = torch.zeros((3, H, W), device=device)
        hit = torch.zeros((H, W), device=device, dtype=torch.bool)
        big = hit.clone()
        if fill_mode != "none":
            out = (torch.randn_like(out) * fill_sigma + 0.5).clamp(0, 1)
        return (out, big, hit) if return_hit else (out, big)

    x_i = x[in_img].long()
    y_i = y[in_img].long()
    z_i = z[in_img].float()
    c_i = rgb[valid][in_img].float()

    flat = (y_i * W + x_i).long()

    # near-first
    z_norm = z_i.double() / (z_i.max().double() + 1.0)
    sort_keys = flat.double() + z_norm
    order = torch.argsort(sort_keys)
    flat_s = flat[order]
    col_s  = c_i[order]

    keep = torch.ones_like(flat_s, dtype=torch.bool)
    keep[1:] = flat_s[1:] != flat_s[:-1]

    flat_k = flat_s[keep]
    col_k  = col_s[keep]

    out_flat = torch.zeros((H * W, 3), device=device, dtype=col_k.dtype)
    hit_flat = torch.zeros((H * W,), device=device, dtype=torch.bool)

    out_flat[flat_k] = col_k
    hit_flat[flat_k] = True

    out = out_flat.view(H, W, 3).permute(2, 0, 1)
    hit = hit_flat.view(H, W)

    # big mask 계산 (hit 기반)
    big = compute_big_mask_from_hit(
        hit, win=big_win,
        min_fill_ratio=big_min_fill_ratio,
        smooth_iters=big_smooth_iters,
    )

    # fill은 "빈 픽셀"만 채우되, hit 기준으로 판단
    empty = ~hit
    if fill_mode != "none" and empty.sum() > 0:
        if fill_mode == "dilation":
            out2 = F.max_pool2d(out.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
            out[:, empty] = out2[:, empty]

        elif fill_mode == "gray_noise":
            noise = torch.randn((3, H, W), device=device) * fill_sigma + 0.5
            out[:, empty] = noise[:, empty].clamp(0, 1)

        elif fill_mode == "image_stats":
            vals = out[:, hit]  # [3,N]
            if vals.numel() < 3 * 64:
                mean = torch.full((3, 1, 1), 0.5, device=device)
                std  = torch.full((3, 1, 1), fill_sigma, device=device)
            else:
                mean = vals.mean(dim=1).view(3, 1, 1)
                std  = vals.std(dim=1, unbiased=False).clamp(min=1e-3).view(3, 1, 1)
            noise = torch.randn((3, H, W), device=device) * std + mean
            out[:, empty] = noise[:, empty].clamp(0, 1)
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return (out, big, hit) if return_hit else (out, big) 