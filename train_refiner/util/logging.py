import wandb


def chw_to_wandb(img_chw):
    img = img_chw.detach().float().clamp(0, 1).cpu()
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return wandb.Image(img)

def debug_lora(model, tag="model", max_print=100):
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower()]
        print(f"[{tag}] LoRA params tensors: {len(lora_params)} | scalars: {sum(p.numel() for _, p in lora_params):,}")
        for n, _ in lora_params[:max_print]:
            print("  ", n)