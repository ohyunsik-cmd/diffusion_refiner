# train_refiner/models/sdxl_refiner.py
import os, re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from peft import LoraConfig, inject_adapter_in_model
from .models.model import make_sched
from .models.vae_skip import VAEEncoderHook, setup_vae_skip_connections
from .models.sdxl_conditioning import encode_prompt_refinement


# -------------------------
# Utils (moved from train_sdxl_controlnet.py)
# -------------------------

def denoise_ksteps(unet, scheduler, x, timesteps, prompt_embeds, pooled_prompt_embeds, add_time_ids):
    for t in timesteps:
        t_in = t.view(1)
        noise_pred = unet(
            x,
            t_in,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
        ).sample
        x = scheduler.step(noise_pred, t_in, x).prev_sample
    return x


def prepare_scheduler_and_prompts(base_model: str, pipe: StableDiffusionXLImg2ImgPipeline, device, dtype):
    scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")
    prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_prompt_refinement(pipe, device)
    return (
        scheduler,
        prompt_embeds.to(dtype=dtype),
        pooled_prompt_embeds.to(dtype=dtype),
        add_time_ids.to(dtype=dtype),
    )


def setup_vae_trainable_parts(vae, lora_config: LoraConfig, device, include_gamma=True):
    # LoRA injection (optional)
    try:
        vae = inject_adapter_in_model(lora_config, vae, adapter_name="vae_lora")
    except Exception:
        print("[WARN] VAE LoRA injection failed. Continuing without it.")

    # skip convs
    if hasattr(vae.decoder, "skip_convs"):
        for p in vae.decoder.skip_convs.parameters():
            p.requires_grad_(True)

    # decoder last layers
    for p in vae.decoder.conv_out.parameters():
        p.requires_grad_(True)

    if hasattr(vae.decoder, "conv_norm_out") and vae.decoder.conv_norm_out is not None:
        for p in vae.decoder.conv_norm_out.parameters():
            p.requires_grad_(True)

    # gamma
    if include_gamma and hasattr(vae.decoder, "gamma") and isinstance(vae.decoder.gamma, torch.nn.Parameter):
        vae.decoder.gamma.requires_grad_(True)

    vae_dtype = next(vae.parameters()).dtype
    if hasattr(vae.decoder, "skip_convs"):
        vae.decoder.skip_convs = vae.decoder.skip_convs.to(device=device, dtype=vae_dtype)

    vae.to(device=device, dtype=vae_dtype)
    vae.train()
    return vae


def collect_trainable_params(unet, vae, include_gamma=True) -> Dict[str, List[torch.nn.Parameter]]:
    unet_params = [p for p in unet.parameters() if p.requires_grad]

    vae_trainables = [(n, p) for n, p in vae.named_parameters() if p.requires_grad]
    vae_lora_params = [p for n, p in vae_trainables if "lora" in n.lower()]
    vae_last_params = [p for n, p in vae_trainables if ("conv_out" in n or "conv_norm_out" in n)]
    vae_skip_params = [p for n, p in vae_trainables if ("skip_convs" in n or "skip_conv" in n)]

    vae_gamma_params = []
    if include_gamma and hasattr(vae.decoder, "gamma") and isinstance(vae.decoder.gamma, torch.nn.Parameter):
        vae_gamma_params = [vae.decoder.gamma]

    all_params = unet_params + vae_lora_params + vae_last_params + vae_skip_params + vae_gamma_params

    unique, seen = [], set()
    for p in all_params:
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)

    return {
        "unet": unet_params,
        "vae_lora": vae_lora_params,
        "vae_last": vae_last_params,
        "vae_skip": vae_skip_params,
        "vae_gamma": vae_gamma_params,
        "unique": unique,
    }


def load_refiner_checkpoint(
    ckpt_path: str,
    *,
    unet,
    vae,
    lr_scheduler=None,
    optimizer=None,
    device="cuda",
    dtype=torch.float16,
    verbose=True,
) -> Tuple[Optional[dict], int]:
    resume_step = 0
    if (not ckpt_path) or (not os.path.exists(ckpt_path)):
        if verbose:
            print("[RESUME] No checkpoint found -> start from scratch")
        return None, resume_step

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if verbose:
        print("[CKPT]", ckpt_path)
        print("[CKPT KEYS]", list(ckpt.keys()))

    if "global_step" in ckpt:
        resume_step = int(ckpt["global_step"])
    else:
        m = re.search(r"refiner_step_(\d+)", os.path.basename(ckpt_path))
        if m:
            resume_step = int(m.group(1))

    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        except Exception as e:
            if verbose:
                print("[RESUME] lr_scheduler load failed:", e)

    if "vae_skip_convs" in ckpt and hasattr(vae.decoder, "skip_convs"):
        try:
            vae.decoder.skip_convs.load_state_dict(ckpt["vae_skip_convs"], strict=True)
        except Exception as e:
            if verbose:
                print("[RESUME] vae_skip_convs load failed:", e)

    if "unet" in ckpt:
        unet.load_state_dict(ckpt["unet"], strict=False)

    if "vae" in ckpt:
        vae.load_state_dict(ckpt["vae"], strict=False)

    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            if verbose:
                print("[RESUME] optimizer load failed:", e)

    if "gamma" in ckpt and hasattr(vae.decoder, "gamma") and isinstance(vae.decoder.gamma, torch.nn.Parameter):
        try:
            with torch.no_grad():
                vae.decoder.gamma.copy_(torch.tensor(float(ckpt["gamma"]), device=device, dtype=dtype))
        except Exception as e:
            if verbose:
                print("[RESUME] gamma load failed:", e)

    if verbose:
        print("[RESUME] global_step =", resume_step)
    return ckpt, resume_step


# -------------------------
# Model class (Pix2Pix_Turbo style)
# -------------------------

@dataclass
class SDXLRefinerCfg:
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # LoRA
    unet_lora_r: int = 16
    unet_lora_alpha: int = 16
    unet_lora_targets: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")

    vae_lora_r: int = 16
    vae_lora_alpha: int = 16
    vae_lora_targets: Tuple[str, ...] = ("conv_in", "conv_out", "conv1", "conv2", "down_block", "up_block")

    include_gamma: bool = True
    


class SDXLRefiner(torch.nn.Module):
    """
    너 train_sdxl_controlnet.py에서 하던 SDXL+LoRA+VAE-skip 기반 refiner를
    pix2pix_turbo.py의 Pix2Pix_Turbo처럼 '모델 객체'로 정리한 버전.
    """
    def __init__(self, cfg: SDXLRefinerCfg):
        super().__init__()
        self.cfg = cfg

        # --- load pipe
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            cfg.base_model,
            torch_dtype=cfg.dtype,
            variant="fp16" if cfg.dtype in (torch.float16, torch.bfloat16) else None,
        ).to(cfg.device)

        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2

        # --- freeze everything first
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # --- VAE skip connections (Pix2Pix-Turbo style)
        self.vae = setup_vae_skip_connections(self.vae, cfg.device)
        self.vae_encoder_hook = VAEEncoderHook(self.vae)
        self.vae_encoder_hook.register_hooks()

        # --- VAE LoRA + trainables
        vae_lora_cfg = LoraConfig(
            r=cfg.vae_lora_r,
            lora_alpha=cfg.vae_lora_alpha,
            target_modules=list(cfg.vae_lora_targets),
            lora_dropout=0.0,
        )
        self.vae = setup_vae_trainable_parts(self.vae, vae_lora_cfg, cfg.device, include_gamma=cfg.include_gamma)

        # --- UNet LoRA
        unet_lora_cfg = LoraConfig(
            r=cfg.unet_lora_r,
            lora_alpha=cfg.unet_lora_alpha,
            target_modules=list(cfg.unet_lora_targets),
            lora_dropout=0.0,
        )
        self.unet = inject_adapter_in_model(unet_lora_cfg, self.unet)
        self.unet.train()

        # --- scheduler + fixed prompt embeds
        self.scheduler, self.prompt_embeds, self.pooled_prompt_embeds, self.add_time_ids = \
            prepare_scheduler_and_prompts(cfg.base_model, self.pipe, cfg.device, cfg.dtype)
        
        self.latent_scale = self.vae.config.scaling_factor

    def trainable_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        return collect_trainable_params(self.unet, self.vae, include_gamma=self.cfg.include_gamma)

    @torch.no_grad()
    def encode_image_to_latent(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        img_01: [B,3,H,W] in 0..1
        return latent: [B,4,h,w]
        """
        vae_dtype = next(self.vae.parameters()).dtype
        x = (img_01 * 2 - 1).to(device=self.cfg.device, dtype=vae_dtype)
        lat = self.vae.encode(x).latent_dist.sample() * self.latent_scale
        return lat

    def forward_refine(
        self,
        rendered_01: torch.Tensor,
        *,
        num_steps: int = 1,
        fixed_timesteps: Optional[List[int]] = None,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        rendered_01: [B,3,H,W] 0..1
        - num_steps: K-step DDIM 스타일로 '짧은 denoise'
        - fixed_timesteps: 예) [600,400,200] 처럼 직접 주면 그걸 사용
        """
        assert rendered_01.dim() == 4, "rendered_01 must be [B,3,H,W]"
        B = rendered_01.shape[0]

        # 1) encode rendered + grab encoder feats for VAE-skip
        self.vae_encoder_hook.clear_features()
        rendered_latent = self.encode_image_to_latent(rendered_01)
        enc_feats = self.vae_encoder_hook.get_features_for_decoder()
        self.vae_encoder_hook.clear_features()

        # 2) make short denoise trajectory
        if fixed_timesteps is not None:
            # mimic your code: set_timesteps(K) then use scheduler.timesteps
            # but 여기서는 fixed_timesteps를 "실제 timestep index"로 간주
            K = len(fixed_timesteps)
            self.scheduler.set_timesteps(K, device=self.cfg.device)
            timesteps = torch.tensor(fixed_timesteps, device=self.cfg.device, dtype=torch.long)
            self.scheduler.timesteps = timesteps
        else:
            self.scheduler.set_timesteps(num_steps, device=self.cfg.device)
            timesteps = self.scheduler.timesteps  # shape [K]
            print("Timesteps:", timesteps)

        eps = torch.randn_like(rendered_latent)
        x = self.scheduler.add_noise(rendered_latent, eps, timesteps[0].view(1))

        # 2-3) K-step denoise 
        x = x.to(dtype=next(self.unet.parameters()).dtype)
        x_denoised = denoise_ksteps(
            self.unet,
            self.scheduler,
            x,
            timesteps,
            self.prompt_embeds,
            self.pooled_prompt_embeds,
            self.add_time_ids,
        )
        
        # 3) decode with VAE-skip
        self.vae.decoder.incoming_skip_acts = enc_feats
        pred = self.vae.decode(x_denoised / self.latent_scale).sample
        pred_01 = (pred / 2 + 0.5).clamp(0, 1)

        out = {"pred_01": pred_01}
        if return_latents:
            out.update({"x_denoised": x_denoised, "rendered_latent": rendered_latent})
        return out

    def save_checkpoint(
        self,
        path: str,
        *,
        optimizer=None,
        lr_scheduler=None,
        global_step: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "unet": self.unet.state_dict(),
            "vae": self.vae.state_dict(),
            "vae_skip_convs": self.vae.decoder.skip_convs.state_dict() if hasattr(self.vae.decoder, "skip_convs") else None,
            "gamma": float(self.vae.decoder.gamma.detach().float().cpu()) if hasattr(self.vae.decoder, "gamma") else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "global_step": int(global_step),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: str,
        *,
        optimizer=None,
        lr_scheduler=None,
        verbose=True,
    ) -> int:
        _, resume_step = load_refiner_checkpoint(
            path,
            unet=self.unet,
            vae=self.vae,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
            verbose=verbose,
        )
        return resume_step