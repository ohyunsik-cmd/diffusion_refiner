import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange
from torch.utils.data import DataLoader
from pathlib import Path

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from train_refiner.models.model import Difix, load_ckpt_from_state_dict, save_ckpt
from train_refiner.models.model_wrapper import VGGTWrapper, VGGTWrapperCfg
from train_refiner.data.re10k_dataset import RE10KDataset, build_chunk_index, make_index_cache_path
from train_refiner.data.view_sampler import select_views
from train_refiner.util.loss import gram_loss
from train_refiner.util.metrics import compute_batch_psnr, compute_batch_ssim
from train_refiner.conf import *

from collections import defaultdict

def find_duplicate_params(model, layers_to_opt, topk=50):
    # model: net_difix or accelerator.unwrap_model(net_difix)
    # layers_to_opt: optimizer에 넣으려는 파라미터 리스트

    # 1) model 전체 파라미터의 "객체 id -> 이름" 매핑 만들기
    id2names = defaultdict(list)
    for name, p in model.named_parameters():
        id2names[id(p)].append(name)

    # 2) layers_to_opt에서 중복 카운트
    id2count = defaultdict(int)
    for p in layers_to_opt:
        id2count[id(p)] += 1

    dups = [(pid, c) for pid, c in id2count.items() if c > 1]
    dups.sort(key=lambda x: -x[1])

    print(f"[dup-check] layers_to_opt size={len(layers_to_opt)} unique={len(id2count)} dups={len(dups)}")
    if not dups:
        return

    print("[dup-check] duplicated params (count, names):")
    for pid, c in dups[:topk]:
        names = id2names.get(pid, ["<NOT FOUND IN model.named_parameters()>"])
        print(f"  x{c}  id={pid}  names={names}")

def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae, 
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()

    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    layers_to_opt += list(net_difix.unet.parameters())
   
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
        
    raw_model = accelerator.unwrap_model(net_difix) if "accelerator" in locals() else net_difix
    find_duplicate_params(raw_model, layers_to_opt)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    # -------------------------
    # Dataset (RE10K + index cache)
    # -------------------------
    args.re10k_root = Path(args.re10k_root)
    if args.train_chunks is None:
        args.train_chunks = list(range(4866))
    if args.val_chunks is None:
        args.val_chunks = list(range(542))
    train_index_cache_path = make_index_cache_path(args.re10k_root, "train", args.train_chunks, args.image_size)
    val_index_cache_path   = make_index_cache_path(args.re10k_root, "test",  args.val_chunks,  args.image_size)

    if train_index_cache_path.exists():
        index_data = torch.load(train_index_cache_path, weights_only=False)
        train_index = [(Path(p), idx) for p, idx in index_data["index"]]
    else:
        train_index = build_chunk_index(args.re10k_root, "train", args.train_chunks)
        torch.save({"index": [(str(p), idx) for p, idx in train_index]}, train_index_cache_path)

    if val_index_cache_path.exists():
        index_data = torch.load(val_index_cache_path, weights_only=False)
        val_index = [(Path(p), idx) for p, idx in index_data["index"]]
    else:
        val_index = build_chunk_index(args.re10k_root, "test", args.val_chunks)
        torch.save({"index": [(str(p), idx) for p, idx in val_index]}, val_index_cache_path)

    dataset_train = RE10KDataset(args.train_chunks, folder="train", precomputed_index=train_index)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    dataset_val = RE10KDataset(args.val_chunks, folder="test", precomputed_index=val_index)
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    #get the vggt model for rendering
    vggt_model = VGGTWrapper(VGGTWrapperCfg(conf_thresh=0.1))
    
    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with our `accelerator`.
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                # batch는 RE10KDataset output dict (train.py랑 동일하게 batch["images"]가 있어야 함)
                B = batch["images"].shape[0]

                # context/target index (일단 하드코딩 가능: 0,1 -> 2)
                # 더 안전하게 하려면 select_views(vnum) 써라 (아래 주석 참고)
                context_indices = torch.tensor([0, 1], device=accelerator.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)  # [B,2]
                target_index    = torch.full((B,), 2, device=accelerator.device, dtype=torch.long)                             # [B]

                # VGGT 렌더 (fp32 + no_grad)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=False):
                        render_result = vggt_model.render_novel_views(batch, context_indices, target_index)

                rendered = render_result["rendered"]  # [B,3,H,W], 0..1
                target   = render_result["target"]    # [B,3,H,W], 0..1
                mask     = render_result["mask"]      # [B,1,H,W], 0..1 (렌더링이 잘 안된 영역이 1)

                # Difix는 [-1,1] + view dim
                x_src = (rendered * 2 - 1).unsqueeze(1).to(accelerator.device, dtype=weight_dtype)  # [B,1,3,H,W]
                x_tgt = (target   * 2 - 1).unsqueeze(1).to(accelerator.device, dtype=weight_dtype)  # [B,1,3,H,W]
                
                H, W = 336, 336
                x_mask = mask.view(B, 1, 1, H, W).to(accelerator.device, dtype=weight_dtype)

                # prompt tokens: empty prompt를 B에 맞게
                empty_tokens = accelerator.unwrap_model(net_difix).tokenizer(
                    [""] * B,
                    max_length=accelerator.unwrap_model(net_difix).tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                # forward
                x_tgt_pred = net_difix(x_src, mask = x_mask, prompt_tokens=empty_tokens)  # [B,1,3,H,W]
                V = 1    
                
                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                         
                # compute PSNR/SSIM before training step
                with torch.no_grad():
                    psnr_pred = float(compute_batch_psnr(x_tgt_pred, x_tgt).mean().item())
                    ssim_pred = float(compute_batch_ssim(x_tgt_pred.detach().cpu(), x_tgt.detach().cpu()).mean().item())
                         
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        _, _, H, W = x_tgt_pred.shape
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h = min(H, 400)
                        crop_w = min(W, 400)
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    logs["lr"] = optimizer.param_groups[0]["lr"]
                    logs["psnr_pred"] = psnr_pred
                    logs["ssim_pred"] = ssim_pred
                    progress_bar.set_postfix(**logs)
                    

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_difix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)

                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break

                            batch_val = to_device(batch_val, accelerator.device)
                            B = batch_val["images"].shape[0]  # eval은 1
                            vnum = batch_val["images"].shape[1]

                            ctx, tgt = select_views(vnum)
                            if ctx is None:
                                continue

                            context_indices = torch.tensor(ctx, device=accelerator.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
                            target_index    = torch.full((B,), int(tgt), device=accelerator.device, dtype=torch.long)

                            with torch.no_grad():
                                with torch.amp.autocast("cuda", enabled=False):
                                    vres = vggt_model.render_novel_views(batch_val, context_indices, target_index)

                                rendered = vres["rendered"]  # [1,3,H,W]
                                target   = vres["target"]    # [1,3,H,W]

                                vx_src = (rendered * 2 - 1).unsqueeze(1).to(dtype=weight_dtype)  # [1,1,3,H,W]
                                empty_tokens = accelerator.unwrap_model(net_difix).tokenizer(
                                    [""],
                                    max_length=accelerator.unwrap_model(net_difix).tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt",
                                ).input_ids.to(accelerator.device)

                                vx_pred = accelerator.unwrap_model(net_difix)(vx_src, prompt_tokens=empty_tokens)  # [1,1,3,H,W]

                                pred_01 = (vx_pred[:,0] * 0.5 + 0.5).clamp(0,1)
                                tgt_01  = target.clamp(0,1)
                                
                                psnr_pred = float(compute_batch_psnr(pred_01, tgt_01).item())
                                ssim_pred = float(compute_batch_ssim(pred_01.detach().cpu(), tgt_01.detach().cpu()).item())
                                

                                loss_l2 = F.mse_loss(pred_01.float(), tgt_01.float(), reduction="mean")
                                pred_m11 = pred_01 * 2 - 1
                                tgt_m11  = tgt_01  * 2 - 1
                                loss_lpips = net_lpips(pred_m11.float(), tgt_m11.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                        logs["val/psnr"] = psnr_pred
                        logs["val/ssim"] = ssim_pred
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=0.5, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="difix", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True, default="Train_difix", help="The name of the wandb run to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=336,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=8,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # Dataset-specific args (RE10K)
    parser.add_argument("--re10k_root", type=str, required=True, default = RE10K_ROOT)
    parser.add_argument("--train_chunks", type=int, nargs="+", default=None)
    parser.add_argument("--val_chunks", type=int, nargs="+", default=None)
    parser.add_argument("--n_context", type=int, default=2)
    parser.add_argument("--n_target", type=int, default=1)  # 보통 1로 시작
    parser.add_argument("--image_size", type=int, default=336)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)