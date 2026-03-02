# train.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_refiner.util.logging import chw_to_wandb
import wandb
import random
import numpy as np
from pathlib import Path
from collections import deque
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms
import lpips

from train_refiner.models.model import Difix, load_ckpt_from_state_dict, save_ckpt
from diffusers.optimization import get_scheduler
from train_refiner.models.model_wrapper import VGGTWrapper, VGGTWrapperCfg
from train_refiner.data.re10k_dataset import RE10KDataset, build_chunk_index, make_index_cache_path
from train_refiner.data.view_sampler import select_views
from train_refiner.util.metrics import compute_psnr, compute_ssim
from train_refiner.util.loss import gram_loss
from train_refiner.models.pipeline_difix import DifixPipeline
from DifixTrainWrapper import DifixTrainWrapper
from train_refiner.conf import *

os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 재현성 강하게 (속도 조금 손해)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def squeeze_batch_dim(batch):
    return {
        k: (v[0] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
        for k, v in batch.items()
    }
    
class MovingAvg:
    def __init__(self, window=50):
        self.buf = deque(maxlen=window)

    def update(self, x: float):
        self.buf.append(float(x))

    @property
    def value(self) -> float:
        return float(sum(self.buf) / max(1, len(self.buf)))
    
def params_from_optimizer(optim):
    for group in optim.param_groups:
        for p in group["params"]:
            if p is not None and getattr(p, "requires_grad", False):
                yield p
    
def to_grid_wandb(images_chw, nrow=4):
    """
    images_chw: list[Tensor] each [3,H,W] in 0..1
    """
    imgs = [img.detach().cpu().clamp(0, 1) for img in images_chw]
    grid = make_grid(imgs, nrow=nrow)  # [3,H,W]
    return chw_to_wandb(grid)

# NOTE: 디버깅용으로 텐서 정보 출력할 때 사용
def finfo(x, name):
    x = x.detach()
    print(f"[{name}] dtype={x.dtype} shape={tuple(x.shape)} "
          f"min={x.min().item():.4f} max={x.max().item():.4f} "
          f"nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()}")
    
def unique_params(params):
    seen = set()
    out = []
    for p in params:
        if p.requires_grad and id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


# -------------------------
# Main
# -------------------------

def main():

    seed_everything(SEED)
    wandb.init(project="vggt-sdxl-refiner", name=RUN_NAME)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (DEVICE.type == "cuda")
    use_fp16 = (DTYPE == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and use_fp16))

    # -------------------------
    # VGGT
    # -------------------------
    print("Loading VGGT...")
    vggt_model = VGGTWrapper(VGGTWrapperCfg(conf_thresh=0.1, use_dilation_fill=True))

    # -------------------------
    # SDXL Refiner
    # -------------------------
    print("Loading Difix...")
    assert torch.cuda.is_available(), "Difix(model.py) is CUDA-only in current implementation."
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to("cuda")
    
    model = DifixTrainWrapper(pipe, timestep=199).to(DEVICE)
    model.set_trainable(train_skipconv_base=True)
    
    net = model
    net.set_train()
    net = net.to(DEVICE)
    
    if ENABLE_XFORMERS:
        try:
            net.unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] xformers enable failed: {e}")

    if ENABLE_GRAD_CKPT:
        try:
            net.unet.enable_gradient_checkpointing()
        except Exception as e:
            print(f"[WARN] grad ckpt enable failed: {e}")

    if ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # LPIPS
    net_lpips = lpips.LPIPS(net='vgg').to(DEVICE).float()
    net_lpips.requires_grad_(False)
    
    # VGG for Gram loss
    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.to(DEVICE).float()
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))
 
    # -------------------------
    # Optimizer
    # -------------------------  
    
    layers_to_opt = unique_params(list(net.parameters()))

    optimizer = torch.optim.AdamW(layers_to_opt, lr=BASE_LR,
        betas=(ADAM_BETA1, ADAM_BETA2), weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPS,)

    # -------------------------
    # Dataset
    # -------------------------
    print("Loading dataset...")
    # load train & val datasets with caching
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

    train_dataset = RE10KDataset(TRAIN_CHUNKS, folder="train", precomputed_index=train_index)
    val_dataset = RE10KDataset(VAL_CHUNKS, folder="test", precomputed_index=val_index)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BS, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # -------------------------
    # Scheduler (optimizer-step 기준)
    # -------------------------
    steps_per_epoch = len(train_loader)
    max_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = 500

    lr_scheduler = get_scheduler(
        name = "constant",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles = 1,
    )

    # --- checkpoint config ---
    CKPT_DIR = "checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)
    LATEST_PATH = os.path.join(CKPT_DIR, "latest.pkl")
    BEST_PATH   = os.path.join(CKPT_DIR, "best.pkl")
    LATEST_META = os.path.join(CKPT_DIR, "latest.pt")
    BEST_META = os.path.join(CKPT_DIR, "best.pt")
    
    # -------------------------
    # tokenizer
    # -------------------------   
    with torch.no_grad():
        empty_tokens = net.tokenizer(
            [""],
            max_length=net.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
    
    # -------------------------
    # Resume
    # -------------------------
    global_step = 0  # batch-step
    best_psnr = -1e9

    if os.path.exists(LATEST_PATH):
        print(f"[RESUME] loading Difix ckpt: {LATEST_PATH}")
        net, optimizer = load_ckpt_from_state_dict(net, optimizer, LATEST_PATH)

    if os.path.exists(LATEST_META):
        meta = torch.load(LATEST_META, map_location="cpu")
        global_step = int(meta.get("global_step", 0))
        best_psnr = float(meta.get("best_psnr", -1e9))
        try:
            lr_scheduler.load_state_dict(meta["lr_scheduler"])
        except Exception as e:
            print(f"[WARN] lr_scheduler state load failed: {e}")
        print(f"[RESUME] global_step={global_step} best_psnr={best_psnr}")
    else:
        print("[RESUME] no checkpoint -> start from scratch")

    # val iterator 유지
    val_iter = iter(val_loader)

    # moving averages
    ma_loss = MovingAvg(window=50)
    ma_psnr = MovingAvg(window=20)

    # -------------------------
    # Training Loop
    # -------------------------

    opt_step = 0  # optimizer step count (원하면 global_step이랑 분리)

    for epoch in range(NUM_EPOCHS):
        net.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for batch in pbar:
            # batch["images"]: [B, V, 3, H, W] 라고 가정 (너 코드상 num_views = data["images"].shape[0] 이었으니)
            B = batch["images"].shape[0]

            # -------------------------
            # 1) 샘플별로 select_views만 반복
            # -------------------------

            ctx = [0, 1]
            tgt = 2
                
            try:
                B = batch["images"].shape[0]
                device = vggt_model.device  # or DEVICE
                context_indices = torch.tensor([0, 1], device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)  # [B,2]
                target_index    = torch.full((B,), 2, device=device, dtype=torch.long)                              # [B]

                render_result = vggt_model.render_novel_views(batch, context_indices, target_index)
                if render_result is None:
                    print("[render] got None")
                    raise RuntimeError("Render failed, got None")
                    continue
            except Exception as e:
                print("[render Exception]", type(e), e)
                raise
                #continue

            rendered = render_result["rendered"]  # [Bv,3,H,W] 0..1
            target   = render_result["target"]    # [Bv,3,H,W] 0..1

            rendered_01 = rendered.to(DEVICE, non_blocking=True)
            target_01   = target.to(DEVICE, non_blocking=True)
            # NOTE: debug
            finfo(rendered_01, "rendered_01")
            finfo(target_01,   "target_01")

            # Difix 입력은 -1..1 + view dim (V=1)
            x_src = (rendered_01 * 2 - 1).unsqueeze(1)  # [Bv,1,3,H,W]
            x_tgt = (target_01   * 2 - 1).unsqueeze(1)  # [Bv,1,3,H,W]
            
            # NOTE: debug
            print("rendered:", rendered_01.shape, "target:", target_01.shape)  # [B,3,H,W]
            print("x_src:", x_src.shape, "x_tgt:", x_tgt.shape)                # [B,1,3,H,W]

            # -------------------------
            # 3) Forward (배치 단위)
            # -------------------------
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=DTYPE):
                x_pred = net(x_src, prompt_tokens=empty_tokens)  # [Bv,1,3,H,W] -1..1
                # NOTE: debug
                finfo(x_pred, "x_pred(-1..1)")
                pred_01 = (x_pred[:, 0] * 0.5 + 0.5).clamp(0, 1).float()   # [Bv,3,H,W]
                # NOTE: debug
                finfo(pred_01, "pred_01(0..1)")
                tgt_01  = target_01.clamp(0, 1).float()                    # [Bv,3,H,W]

                loss_img   = F.mse_loss(pred_01, tgt_01)  # 평균
                
                pred_m11 = (pred_01 * 2 - 1)
                tgt_m11  = (tgt_01  * 2 - 1)
                
                loss_lpips = net_lpips(pred_m11, tgt_m11).mean()
                loss = (L_IMG_WEIGHT * loss_img) + (L_LPIPS_WEIGHT * loss_lpips)

                # --- Gram (warmup 이후) ---
                loss_gram = torch.tensor(0.0, device=DEVICE)
                if (global_step > GRAM_WARMUP_STEPS) and (L_GRAM_WEIGHT > 0):
                    with torch.cuda.amp.autocast(enabled=False):
                        p = t_vgg_renorm(pred_01.float())
                        t = t_vgg_renorm(tgt_01.float())

                        # 배치 전체에 동일 crop (간단/빠름). 샘플별 랜덤 crop 원하면 for문 돌려도 됨.
                        crop_h = crop_w = 400
                        _, _, H, W = p.shape
                        if H >= crop_h and W >= crop_w:
                            top  = random.randint(0, H - crop_h)
                            left = random.randint(0, W - crop_w)
                            p = TF.crop(p, top, left, crop_h, crop_w)
                            t = TF.crop(t, top, left, crop_h, crop_w)

                        loss_gram = gram_loss(p, t, vgg) * L_GRAM_WEIGHT
                    loss = loss + loss_gram

            # -------------------------
            # 4) Backward + step (배치 당 1회)
            # -------------------------
            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(params_from_optimizer(optimizer)), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(params_from_optimizer(optimizer)), MAX_GRAD_NORM)
                optimizer.step()

            lr_scheduler.step()
            opt_step += 1

            # -------------------------
            # 5) Logging
            # -------------------------
            true_loss = float(loss.detach().item())
            ma_loss.update(true_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            if (global_step % LOG_EVERY) == 0:
                wandb.log(
                    {
                        "train/loss": true_loss,
                        "train/loss_ma": ma_loss.value,
                        "train/loss_img": float(loss_img.detach().item()),
                        "train/loss_lpips": float(loss_lpips.detach().item()),
                        "train/loss_gram": float(loss_gram.detach().item()),
                        "train/lr": current_lr,
                    },
                    step=global_step,
                )

            pbar.set_postfix(
                {
                    "loss": f"{true_loss:.4f}",
                    "ma": f"{ma_loss.value:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            global_step += 1
            
            # -------------------------
            # Validation 
            # -------------------------
            if global_step % VAL_EVERY == 0:
                net.eval()
                psnrs_pred, ssims_pred = [], []
                psnrs_rnd, ssims_rnd = [], []

                grid_logged = False

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=DTYPE):
                    for vi in range(NUM_VAL_STEPS):
                        try:
                            vb = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            vb = next(val_iter)

                        vd = squeeze_batch_dim(vb)
                        vnum = vd["images"].shape[0]
                        vctx, vtgt = select_views(vnum)
                        if vctx is None or vtgt is None:
                            continue

                        v_context_indices = torch.tensor([vctx], dtype=torch.long, device=DEVICE)
                        v_target_index = torch.tensor([vtgt], dtype=torch.long, device=DEVICE)

                        try:
                            with torch.amp.autocast("cuda", enabled=False):
                                vres = vggt_model.render_novel_views(vd, v_context_indices, v_target_index)
                            if vres is None:
                                continue
                        except Exception as e:
                            print(f"[VAL] render error: {e}")
                            continue

                        v_rendered = vres["rendered"][0]  # [3,H,W] - take first batch
                        v_target = vres["target"][0]      # [3,H,W]
                        v_ctx0 = vres.get("context0", None)
                        v_ctx1 = vres.get("context1", None)

                        v_rendered_01 = v_rendered.unsqueeze(0).to(DEVICE, non_blocking=True)
                        v_target_01 = v_target.unsqueeze(0).to(DEVICE, non_blocking=True)
                        
                        # --- Difix refine ---
                        vx_src = (v_rendered_01 * 2 - 1).unsqueeze(1)  # [1,1,3,H,W] -1..1
                        with torch.amp.autocast("cuda", enabled=use_amp, dtype=DTYPE):
                            vx_pred = net(vx_src, prompt_tokens=empty_tokens)  # [1,1,3,H,W] -1..1
                            v_pred_01 = (vx_pred[:, 0] * 0.5 + 0.5).clamp(0, 1)  # [1,3,H,W] 0..1

                        # metrics: rendered->gt (baseline) vs pred->gt
                        psnr_rnd = float(compute_psnr(v_rendered_01, v_target_01).item())
                        ssim_rnd = float(compute_ssim(v_rendered_01[0].detach().cpu(), v_target_01[0].detach().cpu()))
                        psnr_pred = float(compute_psnr(v_pred_01, v_target_01).item())
                        ssim_pred = float(compute_ssim(v_pred_01[0].detach().cpu(), v_target_01[0].detach().cpu()))

                        psnrs_rnd.append(psnr_rnd)
                        ssims_rnd.append(ssim_rnd)
                        psnrs_pred.append(psnr_pred)
                        ssims_pred.append(ssim_pred)

                        if (not grid_logged) and (v_ctx0 is not None) and (v_ctx1 is not None):
                            grid = to_grid_wandb(
                                [v_ctx0[0], v_ctx1[0], v_rendered, v_target, v_pred_01[0].detach().cpu()],
                                nrow=5,
                            )
                            if grid is not None:
                                wandb.log({"val/grid_ctx_rnd_gt_pred": grid,}, step = global_step)
                            grid_logged = True

                if len(psnrs_pred) > 0:
                    val_psnr_mean = float(np.mean(psnrs_pred))
                    val_ssim_mean = float(np.mean(ssims_pred))
                    val_psnr_rnd_mean = float(np.mean(psnrs_rnd)) if len(psnrs_rnd) else 0.0
                    val_ssim_rnd_mean = float(np.mean(ssims_rnd)) if len(ssims_rnd) else 0.0
                    val_gain = val_psnr_mean - val_psnr_rnd_mean

                    ma_psnr.update(val_psnr_mean)

                    wandb.log(
                        {
                            "val/rendered": chw_to_wandb(v_rendered),
                            "val/context0": chw_to_wandb(v_ctx0[0]),
                            "val/context1": chw_to_wandb(v_ctx1[0]),
                            "val/gt": chw_to_wandb(v_target),
                            "val/predicted": chw_to_wandb(v_pred_01[0]),
                            "val/psnr_mean": val_psnr_mean,
                            "val/ssim_mean": val_ssim_mean,
                            "val/rendered_psnr_mean": val_psnr_rnd_mean,
                            "val/rendered_ssim_mean": val_ssim_rnd_mean,
                            "val/psnr_gain_over_rendered": val_gain,
                            "val/psnr_ma": ma_psnr.value,
                        } , step = global_step
                    )

                    # best checkpoint
                    if val_psnr_mean > best_psnr:
                        best_psnr = val_psnr_mean
                        
                        save_ckpt(net, optimizer, BEST_PATH)
                        torch.save(
                             {
                                 "epoch": epoch,
                                 "global_step": global_step,
                                 "opt_step": opt_step,
                                 "best_psnr": best_psnr,
                                 "lr_scheduler": lr_scheduler.state_dict(),
                             },
                             BEST_META,
                         )
                        print(f"[CKPT] best saved: {BEST_PATH} (best_psnr={best_psnr:.4f})")
                        print(f"[CKPT] best saved: {BEST_PATH} (best_psnr={best_psnr:.4f})")

                net.train()
            
            # -------------------------
            # CheckPoint
            # -------------------------
            if (global_step % SAVE_EVERY) == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"model_{global_step}.pkl")
                meta_path = os.path.join(CKPT_DIR, f"model_{global_step}_meta.pt")

                save_ckpt(net, optimizer, ckpt_path)
                save_ckpt(net, optimizer, LATEST_PATH)

                meta = {
                     "epoch": epoch,
                     "global_step": global_step,
                     "opt_step": opt_step,
                     "best_psnr": best_psnr,
                     "lr_scheduler": lr_scheduler.state_dict(),
                 }
                torch.save(meta, meta_path)
                torch.save(meta, LATEST_META)

                print(f"[CKPT] saved: {ckpt_path}")


    wandb.finish()


if __name__ == "__main__":
    main()