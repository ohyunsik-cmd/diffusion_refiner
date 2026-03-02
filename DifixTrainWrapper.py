import torch
import torch.nn as nn
from einops import rearrange, repeat

class DifixTrainWrapper(nn.Module):
    def __init__(self, pipe, timestep=199):
        super().__init__()
        # pipe에서 꺼내기
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.sched = pipe.scheduler

        # ⭐ 1-step + timestep=199 컨셉 유지
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        # 이게 없으면 네가 봤던 IndexError 다시 남
        self.sched.timesteps = self.timesteps

        # 보통 text_encoder는 고정
        self.text_encoder.requires_grad_(False)

    def set_trainable(self, train_skipconv_base=True):
        # 기본 다 freeze
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        # LoRA만 train
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)

        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)

        # skip conv base도 학습할거면 켜기 (선택)
        if train_skipconv_base:
            for k in ["skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"]:
                m = getattr(self.vae.decoder, k, None)
                if m is None:
                    continue
                # PEFT가 base_layer로 감싼 경우
                if hasattr(m, "base_layer"):
                    m.base_layer.requires_grad_(True)
                else:
                    m.requires_grad_(True)

        self.unet.train()
        self.vae.train()
        self.text_encoder.eval()

    def forward(self, x_src, prompt_tokens):
        """
        x_src: [B, V, 3, H, W] in [-1,1]
        prompt_tokens: [B, L]
        return: x_pred [B, V, 3, H, W] in [-1,1]
        """
        B, V = x_src.shape[0], x_src.shape[1]

        with torch.no_grad():
            caption_enc = self.text_encoder(prompt_tokens)[0]  # [B,L,C]

        x = rearrange(x_src, "b v c h w -> (b v) c h w")
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

        caption_enc = repeat(caption_enc, "b n c -> (b v) n c", v=V)

        model_pred = self.unet(z, self.timesteps, encoder_hidden_states=caption_enc).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample

        out = self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample
        out = out.clamp(-1, 1)
        out = rearrange(out, "(b v) c h w -> b v c h w", b=B, v=V)
        return out