import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as tf
import numpy as np
import wandb
from tqdm import tqdm
import os, re
from functools import cache
from copy import deepcopy
from train_refiner.models.model_wrapper import VGGTWrapper, VGGTWrapperCfg
from train_refiner.util.metrics import compute_psnr, compute_ssim, get_lpips
from train_refiner.util.logging import chw_to_wandb, debug_lora
from train_refiner.data.re10k_dataset import RE10KDataset, build_chunk_index, make_index_cache_path
from train_refiner.data.view_sampler import select_views
from train_refiner.models.vae_skip import VAEEncoderHook, setup_vae_skip_connections
from train_refiner.models.sdxl_conditioning import encode_prompt_refinement
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, inject_adapter_in_model
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from train_refiner.conf import *

os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'

def load_refiner_checkpoint(
    ckpt_path: str,
    lr_scheduler=None,
    *,
    unet,
    vae,
    optimizer=None,
    device="cuda",
    dtype=torch.float16,
    verbose=True,
):
    """
    Returns:
        ckpt (dict or None), resume_step (int)
    """
    resume_step = 0
    if (not ckpt_path) or (not os.path.exists(ckpt_path)):
        if verbose:
            print("[RESUME] No checkpoint found -> start from scratch")
        return None, resume_step

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if verbose:
        print("[CKPT]", ckpt_path)
        print("[CKPT KEYS]", list(ckpt.keys()))
        
    # --- global_step ---
    if "global_step" in ckpt:
        resume_step = int(ckpt["global_step"])
        if verbose: print("[RESUME] global_step from ckpt =", resume_step)
    else:
        m = re.search(r"refiner_step_(\d+)", os.path.basename(ckpt_path))
        if m:
            resume_step = int(m.group(1))
            if verbose: print("[RESUME] global_step parsed from filename =", resume_step)
    
    #-- lr_scheduler ---
    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if verbose: print("[RESUME] lr_scheduler loaded.")
        except Exception as e:
            if verbose: print("[RESUME] lr_scheduler load failed:", e)
    
    # --- VAE skip convs ---
    if "vae_skip_convs" in ckpt and hasattr(vae.decoder, "skip_convs"):
        try:
            vae.decoder.skip_convs.load_state_dict(ckpt["vae_skip_convs"], strict=True)
            if verbose: print("[RESUME] vae_skip_convs loaded.")
        except Exception as e:
            if verbose: print("[RESUME] vae_skip_convs load failed:", e)
    else:
        if verbose: print("[RESUME] vae_skip_convs not found or skip_convs missing -> skip")

    # --- UNet ---
    if "unet" in ckpt:
        missing, unexpected = unet.load_state_dict(ckpt["unet"], strict=False)
        if verbose:
            print("[RESUME] UNet loaded.", "missing:", len(missing), "unexpected:", len(unexpected))
    else:
        if verbose: print("[RESUME] UNet key not found -> skip UNet load")

    # --- VAE ---
    if "vae" in ckpt:
        missing, unexpected = vae.load_state_dict(ckpt["vae"], strict=False)
        if verbose:
            print("[RESUME] VAE loaded.", "missing:", len(missing), "unexpected:", len(unexpected))
    else:
        if verbose: print("[RESUME] VAE key not found -> skip VAE load")

    
    # --- optimizer ---
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if verbose: print("[RESUME] optimizer loaded.")
        except Exception as e:
            if verbose: print("[RESUME] optimizer load failed -> skip:", e)
    
    # --- gamma ---
    if "gamma" in ckpt and hasattr(vae.decoder, "gamma"):
        try:
            with torch.no_grad():
                vae.decoder.gamma.copy_(
                    torch.tensor(float(ckpt["gamma"]), device=device, dtype=dtype)
                )
            if verbose: print("[RESUME] gamma =", float(ckpt["gamma"]))
        except Exception as e:
            if verbose: print("[RESUME] gamma load failed:", e)

    return ckpt, resume_step

def predict_x0_from_eps(latents_noisy, eps_pred, timestep, scheduler):
    """
    Compute x0 from noisy latent and predicted noise using:
    x0 = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    
    Args:
        latents_noisy: noisy latent at timestep t [B, C, H, W]
        eps_pred: predicted noise from UNet [B, C, H, W]
        timestep: current timestep (int)
        scheduler: DDIMScheduler with alphas_cumprod
    
    Returns:
        x0: predicted clean latent
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device=latents_noisy.device, dtype=latents_noisy.dtype)
    
    alpha_bar_t = alphas_cumprod[timestep]
    
    sqrt_alpha_bar = alpha_bar_t.sqrt()
    sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
    
    x0 = (latents_noisy - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
    
    x0 = torch.clamp(x0, -10, 10)
    
    return x0

def inspect_vae_for_lora(vae, max_print=200):
    """
    VAE 내부 Conv2d 모듈 이름을 수집해서 출력.
    LoRA target_modules 후보를 잡을 때 쓰는 1회성 디버깅.
    """
    print("Finding Conv2D modules in VAE for LoRA injection...")
    conv_modules = [name for name, m in vae.named_modules() if isinstance(m, torch.nn.Conv2d)]
    print(f"Found {len(conv_modules)} Conv2D modules:")
    for name in conv_modules[:max_print]:
        print(f"  {name}")
    if len(conv_modules) > max_print:
        print(f"  ... (+{len(conv_modules) - max_print} more)")

    # 패턴 요약(중복 제거 + 보기 좋게)
    common_tokens = [
        "conv_in", "conv_out", "conv_norm_out", "down_blocks", "up_blocks", "mid_block",
        "encoder", "decoder", "quant_conv", "post_quant_conv"
    ]
    hits = {t: any(t in n for n in conv_modules) for t in common_tokens}
    print("\n[Heuristic token presence]")
    for t, ok in hits.items():
        print(f"  {t:14s}: {'YES' if ok else 'no'}")

    print("\n[Note]")
    print("PEFT target_modules는 '정확히 일치'라기보다 이름 매칭(부분/접두)로 걸리는 경우가 많아서,")
    print("위 리스트 보고 공통 prefix/substring을 target_modules로 주는 게 보통 제일 안전함.\n")

    return conv_modules

def prepare_scheduler_and_prompts(base_model, pipe, device, dtype):
    scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

    prompt_embeds, pooled_prompt_embeds, add_time_ids = encode_prompt_refinement(pipe, device)
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
    add_time_ids = add_time_ids.to(dtype=dtype)

    return scheduler, prompt_embeds, pooled_prompt_embeds, add_time_ids

def setup_vae_trainable_parts(vae, config, device = None):

    assert device is not None, "Device must be specified for setup_vae_trainable_parts"
        
    # LoRA injection
    try:
        vae = inject_adapter_in_model(config, vae, adapter_name="vae_lora")
    except Exception:
        print("[WARN] VAE LoRA injection failed. Continuing without it.")

    # skip convs
    if hasattr(vae.decoder, "skip_convs"):
        for p in vae.decoder.skip_convs.parameters():
            p.requires_grad_(True)

    # decoder final layers
    for p in vae.decoder.conv_out.parameters():
        p.requires_grad_(True)

    # decoder conv_norm_out (if exists)
    if hasattr(vae.decoder, "conv_norm_out") and vae.decoder.conv_norm_out is not None:
        for p in vae.decoder.conv_norm_out.parameters():
            p.requires_grad_(True)
    
    # gamma
    if hasattr(vae.decoder, "gamma") and isinstance(vae.decoder.gamma, torch.nn.Parameter):
        vae.decoder.gamma.requires_grad_(True)
    
    print("Setting VAE to training mode...")
    vae_dtype = next(vae.parameters()).dtype
    vae.decoder.skip_convs = vae.decoder.skip_convs.to(device=device, dtype=vae_dtype)

    vae.to(device=device, dtype=vae_dtype)
    vae.train()

    return vae

def debug_vae_trainable_state(vae):
    print("---- VAE Trainable Check ----")

    if hasattr(vae.decoder, "skip_convs"):
        print("skip_convs trainable:",
              all(p.requires_grad for p in vae.decoder.skip_convs.parameters()))

    print("conv_out trainable:",
          all(p.requires_grad for p in vae.decoder.conv_out.parameters()))

    if hasattr(vae.decoder, "conv_norm_out") and vae.decoder.conv_norm_out is not None:
        print("conv_norm_out trainable:",
              all(p.requires_grad for p in vae.decoder.conv_norm_out.parameters()))

def collect_trainable_params(unet, vae, include_gamma=True):
    # UNet: requires_grad True인 것들 (LoRA만 True일 확률 높음)
    unet_params = [p for p in unet.parameters() if p.requires_grad]

    # VAE: 이름 기반으로 분류
    vae_trainables = [(n, p) for n, p in vae.named_parameters() if p.requires_grad]
    vae_lora_params = [p for n, p in vae_trainables if "lora" in n.lower()]
    vae_last_params = [p for n, p in vae_trainables if ("conv_out" in n or "conv_norm_out" in n)]
    vae_skip_params = [p for n, p in vae_trainables if ("skip_convs" in n or "skip_conv" in n)]

    vae_gamma_params = []
    if include_gamma and hasattr(vae.decoder, "gamma"):
        vae_gamma_params = [vae.decoder.gamma]

    # 합치고 dedup
    all_params = unet_params + vae_lora_params + vae_last_params + vae_skip_params + vae_gamma_params
    unique = []
    seen = set()
    for p in all_params:
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)

    # 요약도 같이 반환해두면 main이 더 깔끔
    groups = {
        "unet": unet_params,
        "vae_lora": vae_lora_params,
        "vae_last": vae_last_params,
        "vae_skip": vae_skip_params,
        "vae_gamma": vae_gamma_params,
        "vae_trainables_named": vae_trainables,  # 필요 없으면 빼도 됨
        "unique": unique,
    }
    return groups


def main():
    wandb.init(project="vggt-sdxl-splatdiff", name="vggt-sdxl-refiner", mode="offline")
    
    print("Loading VGGT...")
    vggt_model = VGGTWrapper(VGGTWrapperCfg(conf_thresh=0.1, use_dilation_fill=True))
    
    #------------------------------
    #------SDXL Img2Img load-------
    #------------------------------
    print("Loading SDXL img2img components...")
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype=DTYPE,
        variant="fp16",
    )
    pipe = pipe.to(DEVICE)
    
    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    
    # Freeze all parameters 
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    
    # set VAE decoder final layer to trainable, skip connections, and gamma
    print("Setting up VAE skip connections (Pix2Pix-Turbo style)...")
    vae = setup_vae_skip_connections(vae, DEVICE)
    vae_encoder_hook = VAEEncoderHook(vae)
    vae_encoder_hook.register_hooks()
    
    # debug lora VAE (나중에 필요 없으면 지워 버리자)
    inspect_vae_for_lora(vae)
    
    #------------------------------
    #---- VAE LoRA injection  -----
    #------------------------------
    target_candidates = ["conv_in", "conv_out", "conv1","conv2", "down_block", "up_block"]
    vae_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_candidates,
        lora_dropout=0.0,
    )
    # LoRA injection into VAE + set trainable parts
    vae = setup_vae_trainable_parts(vae, vae_lora_config, device=DEVICE)
    
    # debug VAE parameters after LoRA injection
    debug_vae_trainable_state(vae)
    
    #------------------------------
    #---- UNET LoRA injection  ----
    #------------------------------  
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    
    unet = inject_adapter_in_model(lora_config, unet)
    unet.train()
    
    #-----------------------------------
    #---- collect trainable params  ----
    #----------------------------------- 
    scheduler, prompt_embeds, pooled_prompt_embeds, add_time_ids = prepare_scheduler_and_prompts(base_model, pipe, DEVICE, DTYPE)
    
    groups = collect_trainable_params(unet, vae, include_gamma=True)
    unet_lora_params = groups["unet"]
    vae_lora_params = groups["vae_lora"]
    vae_last_params = groups["vae_last"]
    vae_skip_params = groups["vae_skip"]
    vae_gamma_params = groups["vae_gamma"]
    unique_params = groups["unique"]
    
    # debug parameter groups
    print(f"Training UNet LoRA parameters: {sum(p.numel() for p in unet_lora_params):,}")
    print(f"Training VAE LoRA parameters: {sum(p.numel() for p in vae_lora_params):,}")
    print(f"Training VAE decoder last layer parameters: {sum(p.numel() for p in vae_last_params):,}")
    print(f"Training VAE skip conv parameters: {sum(p.numel() for p in vae_skip_params):,}")
    print(f"Training VAE gamma parameters: {sum(p.numel() for p in vae_gamma_params):,}")
    print(f"Total trainable parameters (unique): {sum(p.numel() for p in unique_params):,}")
    
    # optimizer는 모든 trainable 파라미터를 포함하는 리스트로 생성
    optimizer = torch.optim.AdamW(unique_params, lr=5e-6)

    LATENT_SCALE = vae.config.scaling_factor
    print(f"Using latent scale: {LATENT_SCALE}")
    
    lpips_fn = get_lpips(DEVICE)
    lpips_fn.requires_grad_(False)
    
    # SSIM
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    # load train & val datasets with caching
    print("Loading datasets...")
    train_index_cache_path = make_index_cache_path(RE10K_ROOT, "train", TRAIN_CHUNKS, IMAGE_SIZE)
    val_index_cache_path   = make_index_cache_path(RE10K_ROOT, "test",  VAL_CHUNKS,  IMAGE_SIZE)

    # --- train index ---
    if train_index_cache_path.exists():
        print(f"Loading TRAIN index cache: {train_index_cache_path}")
        index_data = torch.load(train_index_cache_path, weights_only=False)
        train_index = [(Path(p), idx) for p, idx in index_data["index"]]
    else:
        print("Building TRAIN index (this may take a few minutes)...")
        train_index = build_chunk_index(RE10K_ROOT, "train", TRAIN_CHUNKS)
        train_index_serializable = [(str(p), idx) for p, idx in train_index]
        torch.save({"index": train_index_serializable}, train_index_cache_path)
        print(f"Saved TRAIN index cache to {train_index_cache_path}")

    # --- val index ---
    if val_index_cache_path.exists():
        print(f"Loading VAL index cache: {val_index_cache_path}")
        index_data = torch.load(val_index_cache_path, weights_only=False)
        val_index = [(Path(p), idx) for p, idx in index_data["index"]]
    else:
        print("Building VAL index (this may take a few minutes)...")
        val_index = build_chunk_index(RE10K_ROOT, "test", VAL_CHUNKS)
        val_index_serializable = [(str(p), idx) for p, idx in val_index]
        torch.save({"index": val_index_serializable}, val_index_cache_path)
        print(f"Saved VAL index cache to {val_index_cache_path}")
        

    train_dataset = RE10KDataset(TRAIN_CHUNKS, folder="train", precomputed_index=train_index, use_cache=True)
    val_dataset = RE10KDataset(VAL_CHUNKS, folder="test", precomputed_index=val_index, use_cache=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    vae_dtype = next(vae.parameters()).dtype
    
    debug_lora(vae, "VAE")
    
    print("=== Decoder parameters ===")
    for n, p in pipe.vae.named_parameters():
        if n.startswith("decoder"):
            print(n)
            
    val_iter = iter(val_loader)        
    
    num_epochs = 3
    num_update_steps_per_epoch = (len(train_loader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
    max_train_steps = num_epochs * num_update_steps_per_epoch
    warmup_steps = 50

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(warmup_steps, max_train_steps),
        num_training_steps=max_train_steps,
    )
    
        # resume from checkpoint if available
    ckpt, resume_step = load_refiner_checkpoint(
        RESUME_CKPT,
        lr_scheduler=lr_scheduler,
        unet=unet,
        vae=vae,
        optimizer=optimizer,
        device=DEVICE,
        dtype=DTYPE,
        verbose=True,
    )
    '''
    # ---- Force gamma initial value ----
    GAMMA_INIT = 0.05

    with torch.no_grad():
        vae.decoder.gamma.fill_(GAMMA_INIT)

    vae.decoder.gamma.requires_grad_(True)  # 이후에는 학습되게

    print("[GAMMA] Initialized to", float(vae.decoder.gamma.detach().cpu()))
    '''
    
    global_step = max(resume_step, 0)
    update_step = global_step // ACCUMULATION_STEPS

    for epoch in range(3):
        unet.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            data = {
                k: (v[0] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
                for k, v in batch.items()
            }
            
            num_views = data["images"].shape[0]
            context_indices, target_index = select_views(num_views)
            
            if context_indices is None or target_index is None:
                continue
            
            np.random.seed(global_step)
            
            try:
                render_result = vggt_model.render_novel_view(data, context_indices, target_index)
                if render_result is None:
                    continue
            except Exception as e:
                print(f"Error rendering: {e}")
                continue
            
            rendered = render_result["rendered"]
            target = render_result["target"]
            
            rendered_vae = (rendered * 2 - 1).unsqueeze(0).to(device=DEVICE, dtype=vae_dtype)
            target_vae = (target * 2 - 1).unsqueeze(0).to(device=DEVICE, dtype=vae_dtype)
            
            vae_encoder_hook.clear_features()
            
            with torch.no_grad():
                rendered_latent = vae.encode(rendered_vae).latent_dist.sample() * LATENT_SCALE
            
            enc_feats_rendered = vae_encoder_hook.get_features_for_decoder()
            vae_encoder_hook.clear_features()
            
            with torch.no_grad():
                target_latent = vae.encode(target_vae).latent_dist.sample() * LATENT_SCALE
            
            vae_encoder_hook.clear_features()
            
            # --- manual timesteps ---
            t = torch.tensor([200], device=DEVICE, dtype=torch.long)
            
            eps = torch.randn_like(rendered_latent)
            x_t = scheduler.add_noise(rendered_latent, eps, t)

            x_t = x_t.to(dtype=next(unet.parameters()).dtype)
            noise_pred = unet(
                x_t,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample

            # x0 estimate (your current 방식 유지)
            x0_hat = predict_x0_from_eps(x_t, noise_pred, int(t.item()), scheduler)
            
            loss_latent = F.mse_loss(x0_hat, target_latent, reduction='mean')
            
            vae.decoder.incoming_skip_acts = enc_feats_rendered
            pred_image = vae.decode(x0_hat / LATENT_SCALE).sample
            pred_image = (pred_image / 2 + 0.5).clamp(0, 1)
            
            target_image_for_loss = target.unsqueeze(0).to(DEVICE)
            
            loss_img = F.l1_loss(pred_image, target_image_for_loss, reduction='mean')
            
            loss_lpips = lpips_fn(pred_image, target_image_for_loss, normalize=True).mean()
            loss_ssim = 1.0 - ssim(pred_image.float(), target_image_for_loss.float())
            
            loss = (L_LATENT_WEIGHT * loss_latent + 
                    L_IMG_WEIGHT * loss_img + 
                    L_LPIPS_WEIGHT * loss_lpips +
                    L_SSIM_WEIGHT * loss_ssim)
            
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if vae.decoder.gamma.grad is None:
                print("[WARN] gamma.grad is None (graph disconnected)")
            else:
                print("gamma.grad mean =", float(vae.decoder.gamma.grad.abs().mean().cpu())) 
            
            if (global_step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                update_step += 1
                
            
            g = float(vae.decoder.gamma.detach().cpu()) 
            
            wandb.log({
                "train/loss": loss.item() * ACCUMULATION_STEPS,
                "train/loss_latent": loss_latent.item(),
                "train/loss_img": loss_img.item(),
                "train/loss_lpips": loss_lpips.item(),
                "train/gamma": g,
                "step": global_step,
                "lr": optimizer.param_groups[0]['lr'],
                "train/timestep": 200,
            })
            
            pbar.set_postfix({
                "loss": loss.item() * ACCUMULATION_STEPS,
                "loss_latent": loss_latent.item(),
                "gamma": vae.decoder.gamma.detach().float().item(),
            })
            global_step += 1
            
            if global_step % 1000 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "unet": unet.state_dict(),
                    "vae": vae.state_dict(),
                    "vae_skip_convs": vae.decoder.skip_convs.state_dict(),
                    "vae_decoder_conv_out": vae.decoder.conv_out.state_dict(),
                    "vae_decoder_conv_norm_out": vae.decoder.conv_norm_out.state_dict() if hasattr(vae.decoder, 'conv_norm_out') and vae.decoder.conv_norm_out is not None else None,
                    "gamma": vae.decoder.gamma.detach().float().item(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "update_Step": update_step,
                }, f"checkpoints/refiner_step_{global_step}.pt")
                print(f"Saved checkpoint at step {global_step}")
                print(f"  - unet: LoRA weights")
                print(f"  - vae: full VAE state (including LoRA if injected + decoder layers)")
                print(f"  - vae_skip_convs: skip connection convs")
                print(f"  - vae_decoder_conv_out: fully trained decoder output conv")
                print(f"  - vae_decoder_conv_norm_out: fully trained decoder output norm")
            
            if global_step % 100 == 0:
                unet.eval()
                
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                    
                val_data = {
                    k: (v[0] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
                    for k, v in val_batch.items()
                }
                 
                val_num_views = val_data["images"].shape[0]
                val_context_indices, val_target_index = select_views(val_num_views)
                
                if val_context_indices is None or val_target_index is None:
                    continue
                
                try:
                    val_result = vggt_model.render_novel_view(val_data, val_context_indices, val_target_index)
                    if val_result is None:
                        continue
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
                
                val_rendered = val_result["rendered"]
                val_target = val_result["target"]
                val_ctx0 = val_result["context0"]
                val_ctx1 = val_result["context1"]
                
                val_rendered_vae = (val_rendered * 2 - 1).unsqueeze(0).to(device=DEVICE, dtype=vae_dtype)
                
                vae_encoder_hook.clear_features()
                
                with torch.no_grad():
                    val_rendered_latent = vae.encode(val_rendered_vae).latent_dist.sample() * LATENT_SCALE
                    
                    val_enc_feats = vae_encoder_hook.get_features_for_decoder()
                    vae_encoder_hook.clear_features()
                    
                    t_val = torch.tensor([200], device=DEVICE, dtype=torch.long)
                    
                    noise_val = torch.randn_like(val_rendered_latent)
                    x_t_val = scheduler.add_noise(val_rendered_latent, noise_val, t_val)

                    x_t_val = x_t_val.to(dtype=next(unet.parameters()).dtype)
                    noise_pred_val = unet(
                        x_t_val,
                        t_val,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids,
                        },
                    ).sample

                    x0_hat_val = predict_x0_from_eps(x_t_val, noise_pred_val, int(t_val.item()), scheduler)
                    
                    vae.decoder.incoming_skip_acts = val_enc_feats
                    pred_val_image = vae.decode(x0_hat_val / LATENT_SCALE).sample
                    pred_val_image = (pred_val_image / 2 + 0.5).clamp(0, 1)
                
                val_target_display = val_target.to(DEVICE).clamp(0, 1)
                val_rendered_display = val_rendered.to(DEVICE).clamp(0, 1)
                
                val_psnr = compute_psnr(pred_val_image[0].unsqueeze(0), val_target_display.unsqueeze(0)).item()
                val_ssim = compute_ssim(pred_val_image[0].cpu(), val_target_display.cpu())
                val_lpips = lpips_fn(pred_val_image[0].unsqueeze(0), val_target_display.unsqueeze(0), normalize=True).mean().item()
                
                rendered_psnr = compute_psnr(val_rendered_display.unsqueeze(0), val_target_display.unsqueeze(0)).item()
                rendered_ssim = compute_ssim(val_rendered_display.cpu(), val_target_display.cpu())
                rendered_lpips = lpips_fn(val_rendered_display.unsqueeze(0), val_target_display.unsqueeze(0), normalize=True).mean().item()
                
                delta_psnr = val_psnr - rendered_psnr
                delta_ssim = val_ssim - rendered_ssim
                delta_lpips = rendered_lpips - val_lpips
                
                wandb.log({
                    "val/rendered": chw_to_wandb(val_rendered),
                    "val/context0": chw_to_wandb(val_ctx0),
                    "val/context1": chw_to_wandb(val_ctx1),
                    "val/gt": chw_to_wandb(val_target),
                    "val/predicted": chw_to_wandb(pred_val_image[0]),
                    "val/psnr": val_psnr,
                    "val/ssim": val_ssim,
                    "val/lpips": val_lpips,
                    "val/baseline_psnr": rendered_psnr,
                    "val/baseline_ssim": rendered_ssim,
                    "val/baseline_lpips": rendered_lpips,
                    "val/delta_psnr": delta_psnr,
                    "val/delta_ssim": delta_ssim,
                    "val/delta_lpips": delta_lpips,
                    "val/gamma": g,
                    "step": global_step,
                    "train/timestep": 200,
                })
                
                unet.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
