# models/vggt_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch

from ..vggt.models.vggt import VGGT
from ..util.pose_enc import pose_encoding_to_extri_intri

from ..conf import DEVICE, DTYPE, IMAGE_SIZE
from ..util.align_camerapose import (
    apply_fxfy_scale,
    estimate_fxfy_from_reproj,
    interp_K_log,
    interp_ratio_from_gt_centers,
)
from ..render.zbuffer import render_pointcloud_zbuffer, dilation_fill


# -------------------------
# Geometry helpers
# -------------------------
def closest_rotation(M: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def camera_center_from_w2c(w2c_3x4: torch.Tensor) -> torch.Tensor:
    R = w2c_3x4[:3, :3]
    t = w2c_3x4[:3, 3]
    return -(R.transpose(0, 1) @ t)


def estimate_sim3_gt_to_pred(
    w2c_gt0: torch.Tensor, w2c_gt1: torch.Tensor,
    w2c_pr0: torch.Tensor, w2c_pr1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Cg0 = camera_center_from_w2c(w2c_gt0)
    Cg1 = camera_center_from_w2c(w2c_gt1)
    Cp0 = camera_center_from_w2c(w2c_pr0)
    Cp1 = camera_center_from_w2c(w2c_pr1)

    bg = torch.norm(Cg1 - Cg0).clamp(min=1e-8)
    bp = torch.norm(Cp1 - Cp0).clamp(min=1e-8)
    s = (bp / bg)

    Rg0 = w2c_gt0[:3, :3].transpose(0, 1)  # c2w
    Rg1 = w2c_gt1[:3, :3].transpose(0, 1)
    Rp0 = w2c_pr0[:3, :3].transpose(0, 1)
    Rp1 = w2c_pr1[:3, :3].transpose(0, 1)

    Rsim0 = Rp0 @ Rg0.transpose(0, 1)
    Rsim1 = Rp1 @ Rg1.transpose(0, 1)
    Rsim = closest_rotation(0.5 * (Rsim0 + Rsim1))

    t = Cp0 - s * (Rsim @ Cg0)
    return s, Rsim, t


def transform_gt_w2c_to_pred_w2c(
    w2c_gt: torch.Tensor, s: torch.Tensor, Rsim: torch.Tensor, tsim: torch.Tensor
) -> torch.Tensor:
    R_gt_c2w = w2c_gt[:3, :3].transpose(0, 1)
    C_gt = camera_center_from_w2c(w2c_gt)

    R_pr_c2w = Rsim @ R_gt_c2w
    C_pr = s * (Rsim @ C_gt) + tsim

    R_pr_w2c = R_pr_c2w.transpose(0, 1)
    t_pr = -(R_pr_w2c @ C_pr)

    w2c_pr = torch.zeros((3, 4), device=w2c_gt.device, dtype=w2c_gt.dtype)
    w2c_pr[:3, :3] = R_pr_w2c
    w2c_pr[:3, 3] = t_pr
    return w2c_pr


# -------------------------
# Config for wrapper
# -------------------------
@dataclass
class VGGTWrapperCfg:
    model_id: str = "facebook/VGGT-1B"
    image_size: int = IMAGE_SIZE
    dtype: torch.dtype = DTYPE
    device: str | torch.device = DEVICE

    # masking / point filtering
    use_image_mask: bool = True
    use_point_conf_mask: bool = True
    conf_thresh: float = 0.1
    image_mask_thresh: float = 0.01

    # postprocess
    use_dilation_fill: bool = True

    # perf
    use_amp: bool = True   # only active on cuda


class VGGTWrapper:
    """
    One-stop VGGT wrapper:
      - loads model
      - runs prediction with amp/no_grad
      - provides render_novel_view() for your training loop
    """

    def __init__(self, cfg: Optional[VGGTWrapperCfg] = None):
        self.cfg = cfg or VGGTWrapperCfg()
        self.device = torch.device(self.cfg.device) if not isinstance(self.cfg.device, torch.device) else self.cfg.device
        self.dtype = self.cfg.dtype
        self.image_size = self.cfg.image_size

        self.model = VGGT.from_pretrained(self.cfg.model_id).to(self.device).eval()

    @torch.no_grad()
    def forward_context(self, context_images_b2chw: torch.Tensor) -> Dict[str, Any]:
        """
        context_images_b2chw: [B,2,3,H,W] (usually B=1)
        """
        x = context_images_b2chw.to(self.device)
        amp_ok = (self.device.type == "cuda") and self.cfg.use_amp
        with torch.cuda.amp.autocast(dtype=self.dtype, enabled=amp_ok):
            pred = self.model(x)
        return pred

    @torch.no_grad()
    def render_novel_views(
        self,
        data: Dict[str, Any],
        context_indices: torch.Tensor,  # [B,2] (long)
        target_index: torch.Tensor,     # [B]   (long)
    ) -> Dict[str, torch.Tensor]:
        device = self.device
        S = self.image_size

        # images: [B,V,3,H,W]
        images = data["images"].to(device)
        extri  = data["extrinsics"].to(device)

        B, V, _, H, W = images.shape
        assert H == S and W == S, f"expected {S}x{S}, got {H}x{W}"

        context_indices = context_indices.to(device).long()
        target_index    = target_index.to(device).long()

        b_idx = torch.arange(B, device=device)

        # [B,2,3,H,W]
        context_images = images[b_idx[:, None], context_indices]

        # VGGT forward in batch (중요: 여기서 이득 제일 큼)
        pred = self.forward_context(context_images)  # pred contains batch

        # pose decode
        pred_pose_enc = pred["pose_enc"]  # 보통 [B,2,...] 형태일 거야
        pred_w2c, pred_K = pose_encoding_to_extri_intri(
            pred_pose_enc,
            (S, S),
            build_intrinsics=True,
        )  # w2c: [B,2,3,4], K: [B,2,3,3] (가정)

        # world points
        wp   = pred["world_points"].float()       # [B,2,H,W,3] (가정)
        conf = pred["world_points_conf"].float()  # [B,2,H,W]

        # colors from context
        # context_images: [B,2,3,H,W] -> [B,2,H,W,3]
        img_ctx = context_images.permute(0, 1, 3, 4, 2).float()

        rendered_out = torch.zeros((B, 3, S, S), device=device, dtype=torch.float32)
        valid = torch.zeros((B,), device=device, dtype=torch.bool)

        # target GT
        target = images[b_idx, target_index]  # [B,3,H,W]

        for b in range(B):
            w2c_pred0 = pred_w2c[b, 0].float()
            w2c_pred1 = pred_w2c[b, 1].float()
            K_pred0 = pred_K[b, 0].float()
            K_pred1 = pred_K[b, 1].float()

            wp0 = wp[b, 0]
            wp1 = wp[b, 1]
            conf0 = conf[b, 0]
            conf1 = conf[b, 1]

            img0 = img_ctx[b, 0]
            img1 = img_ctx[b, 1]

            # masks
            if self.cfg.use_image_mask:
                mask_img0 = img0.mean(-1) > self.cfg.image_mask_thresh
                mask_img1 = img1.mean(-1) > self.cfg.image_mask_thresh
            else:
                mask_img0 = torch.ones((S, S), device=device, dtype=torch.bool)
                mask_img1 = torch.ones((S, S), device=device, dtype=torch.bool)

            if self.cfg.use_point_conf_mask:
                m0 = (conf0 > self.cfg.conf_thresh) & mask_img0
                m1 = (conf1 > self.cfg.conf_thresh) & mask_img1
            else:
                m0, m1 = mask_img0, mask_img1

            pts0, col0 = wp0[m0], img0[m0]
            pts1, col1 = wp1[m1], img1[m1]

            if pts0.numel() == 0 or pts1.numel() == 0:
                continue

            # fx/fy correction
            fx_eff0, fy_eff0 = estimate_fxfy_from_reproj(w2c_pred0, K_pred0, wp0, m0)
            fx_eff1, fy_eff1 = estimate_fxfy_from_reproj(w2c_pred1, K_pred1, wp1, m1)

            fx_scale = float(0.5 * (fx_eff0 / K_pred0[0, 0] + fx_eff1 / K_pred1[0, 0]))
            fy_scale = float(0.5 * (fy_eff0 / K_pred0[1, 1] + fy_eff1 / K_pred1[1, 1]))

            all_pts = torch.cat([pts0, pts1], dim=0)
            all_col = torch.cat([col0, col1], dim=0)

            # GT extrinsics for Sim(3)
            gt_ctx0 = extri[b, context_indices[b, 0]][:3, :4].float()
            gt_ctx1 = extri[b, context_indices[b, 1]][:3, :4].float()
            gt_tgt  = extri[b, target_index[b]][:3, :4].float()

            s, Rsim, tsim = estimate_sim3_gt_to_pred(gt_ctx0, gt_ctx1, w2c_pred0, w2c_pred1)
            w2c_tgt_in_pred = transform_gt_w2c_to_pred_w2c(gt_tgt, s, Rsim, tsim).float()

            # intrinsics for target (interp + scale)
            ratio = interp_ratio_from_gt_centers(gt_ctx0, gt_ctx1, gt_tgt)
            K_tgt = interp_K_log(K_pred0, K_pred1, ratio)
            K_tgt = apply_fxfy_scale(K_tgt, fx_scale, fy_scale)

            rendered = render_pointcloud_zbuffer(all_pts, all_col, w2c_tgt_in_pred, K_tgt, S, S)
            if self.cfg.use_dilation_fill:
                rendered = dilation_fill(rendered)  # or dilation_fill_batched(rendered.unsqueeze(0))[0]

            rendered_out[b] = rendered
            valid[b] = True

        return {
            "rendered": rendered_out,     # [B,3,H,W]
            "target": target,             # [B,3,H,W]
            "context0": context_images[:, 0],  # [B,3,H,W]
            "context1": context_images[:, 1],  # [B,3,H,W]
            "valid": valid,               # [B]
        }
