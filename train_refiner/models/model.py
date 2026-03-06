import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .sd_turbo_masked import SDTurboMaskedConvIn

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
        
    _sd_unet = net_difix.unet.state_dict()
    for k, v in sd["state_dict_unet"].items():
        # ✨ 핵심: 기존 conv_in 키를 새로운 wrapper 내부의 conv1 키로 리다이렉트합니다.
        target_key = k
        if k == "conv_in.weight":
            target_key = "conv_in.conv1.weight"
        elif k == "conv_in.bias":
            target_key = "conv_in.conv1.bias"
        
        # 모델에 실제 존재하는 키인 경우에만 가중치 대입
        if target_key in _sd_unet:
            _sd_unet[target_key] = v
        else:
            # 새로 추가한 conv1_mask는 이전 체크포인트에 없으므로 여기서 건너뛰게 됩니다.
            print(f"Skipping key: {k} (mapping to {target_key} failed or not present)")
    net_difix.unet.load_state_dict(_sd_unet, strict=False)
    
    try:
        optimizer.load_state_dict(sd["optimizer"])
        print("✅ Optimizer state loaded successfully.")
    except (ValueError, KeyError, RuntimeError) as e:
        # 새로운 레이어 추가로 파라미터 개수가 달라졌을 때 이쪽으로 들어옵니다.
        print(f"⚠️ Optimizer state mismatch: {e}")
        print("💡 Architecture changed (Mask Branch added). Skipping optimizer state load.")
        print("💡 Training will proceed with fresh optimizer states for new layers.")
    
    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}
    
    sd["optimizer"] = optimizer.state_dict()   
    
    torch.save(sd, outf)


class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, mv_unet=False, timestep=999):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            
        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")        

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.text_encoder.requires_grad_(False)
        
        # -------------------------
        # Load difix_ref VAE weights (safetensors) into our VAE
        # -------------------------
        ckpt_path_vae = hf_hub_download(
            repo_id="nvidia/difix_ref",
            filename="diffusion_pytorch_model.safetensors",
            subfolder="vae",
        )
        
        ckpt_path_unet = hf_hub_download(
            repo_id="nvidia/difix_ref",
            filename="diffusion_pytorch_model.safetensors",
            subfolder="unet",
        )
        
        ckpt_path_text_encoder = hf_hub_download(
            repo_id="nvidia/difix_ref",
            filename="model.safetensors",
            subfolder="text_encoder",
        )

        sd_vae = load_file(ckpt_path_vae, device="cpu")
        sd_unet = load_file(ckpt_path_unet, device="cpu")
        sd_text_encoder = load_file(ckpt_path_text_encoder, device="cpu")    
        print("Current VAE keys:", list(vae.state_dict().keys())[:5])
        print("Downloaded weights keys:", list(sd_vae.keys())[:5])
        
        missing, unexpected = unet.load_state_dict(sd_unet, strict=False)
        print(f"[UNet difix_ref load] missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > 0:
            print("  unexpected examples:", unexpected[:20])
        if len(missing) > 0:
            print("  missing examples:", missing[:20])  
        missing, unexpected = self.text_encoder.load_state_dict(sd_text_encoder, strict=False)
        print(f"[Text Encoder difix_ref load] missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > 0:
            print("  unexpected examples:", unexpected[:20])
        if len(missing) > 0:
            print("  missing examples:", missing[:20])  
        missing, unexpected = vae.load_state_dict(sd_vae, strict=False)
        print(f"[VAE difix_ref load] missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > 0:
            print("  unexpected examples:", unexpected[:20])
        if len(missing) > 0:
            print("  missing examples:", missing[:20])
            
        unet_lora_config = LoraConfig(r=4, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        unet.add_adapter(unet_lora_config, adapter_name="unet_lora")    
        
        unet.conv_in = SDTurboMaskedConvIn(unet.conv_in).cuda()
        
        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(False)
        
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        self.unet.conv_in.conv1_mask.requires_grad_(True)

    def forward(self, x, mask=None, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
                                
        num_views = x.shape[1]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        
        if mask is not None:
            import torch.nn.functional as F
            # 마스크도 이미지처럼 [b, v, 1, h, w] -> [b*v, 1, h, w] 로 폅니다.
            mask = rearrange(mask, 'b v c h w -> (b v) c h w')
            
            # Latent 크기(예: 42x42)에 맞게 줄입니다.
            mask_latent = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False)
            
            # UNet의 conv_in 모듈 안에 몰래 넣어둡니다.
            self.unet.conv_in.current_mask = mask_latent
        else:
            self.unet.conv_in.current_mask = None
        
        self.sched.set_timesteps(1, device="cuda")
        t_fixed = self.sched.timesteps[0]
        t_unet = t_fixed.expand(z.shape[0]) 
        
        unet_input = z
        
        model_pred = self.unet(unet_input, t_unet, encoder_hidden_states=caption_enc,).sample
        z_denoised = self.sched.step(model_pred, t_fixed, z, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        
        return output_image
    
    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()
        
        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        
        sd["optimizer"] = optimizer.state_dict()
        
        torch.save(sd, outf)