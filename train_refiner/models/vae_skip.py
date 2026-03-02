import torch
from ..conf import DTYPE, SKIP_CONV_INIT
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoderHook:
    """Hook to capture VAE encoder intermediate features."""
    
    def __init__(self, vae):
        self.vae = vae
        self.features = []
        self.handles = []
    
    def register_hooks(self):
        """Register forward hooks on VAE encoder down_blocks only (not mid_block)."""
        self.remove_hooks()
        self.features = []
        
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, (tuple, list)):
                    self.features.append(output[0])
                else:
                    self.features.append(output)
            return hook
        
        # encoder의 모든 downblock feature를 저장
        for i, down_block in enumerate(self.vae.encoder.down_blocks):
            handle = down_block.register_forward_hook(get_hook(f'down_{i}'))
            self.handles.append(handle)
    
    # feature 리스트 초기화
    def clear_features(self):
        """Clear captured features."""
        self.features = []
        
    # 저장한 handle 모두 삭제
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    # 저장한 feature를 decoder로 넘기기 위해 복사본 생성 및 길이 조절
    def get_features_for_decoder(self, num_skip_levels=None):
        """Return features suitable for decoder skip connections.
        
        Args:
            num_skip_levels: Number of skip levels expected by decoder.
                           If None, returns all captured features.
        
        Returns:
            List of features (trimmed to match num_skip_levels if specified)
        """
        features = self.features.copy()
        if num_skip_levels is not None and len(features) > num_skip_levels:
            features = features[:num_skip_levels]
        return features
    

def setup_vae_skip_connections(vae, device):
    """
    Add Pix2Pix-Turbo style skip connections to VAE decoder.
    - Add skip_conv_1..4 (1x1 convs)
    - Add gamma parameter for gating
    - Add incoming_skip_acts storage
    
    Skip features: only from encoder.down_blocks (not mid_block)
    Channel alignment: encoder (shallow→deep) → reversed decoder (shallow→deep)
    """
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = nn.Parameter(torch.tensor(1e-5, device=device, dtype=DTYPE))
    vae.decoder.incoming_skip_acts = None
    
    # encoder down block 개수만큼 skip level 사용 -> 결국 down block의 feature를 다 가져오겠다
    num_down_blocks = len(vae.encoder.down_blocks)
    vae.decoder.num_skip_levels = num_down_blocks
    
    # encoder 출력 채널수 리스트 만들기
    def get_encoder_channels():
        """Get encoder channels from down_blocks resnets (shallow→deep order)."""
        channels = []
        for down_block in vae.encoder.down_blocks:
            try:
                ch = down_block.resnets[-1].out_channels
            except:
                ch = down_block.out_channels if hasattr(down_block, 'out_channels') else 128
            channels.append(ch)
        return channels
    
    # decoder 출력 채널수 리스트 만들기
    def get_decoder_channels():
        """Get decoder channels from up_blocks resnets (deep→shallow order)."""
        channels = []
        for up_block in vae.decoder.up_blocks:
            try:
                ch = up_block.resnets[-1].out_channels
            except:
                ch = up_block.out_channels if hasattr(up_block, 'out_channels') else 512
            channels.append(ch)
        return channels
    
    enc_ch_list = get_encoder_channels()
    dec_ch_list = get_decoder_channels()
    dec_ch_aligned = list(reversed(dec_ch_list))
    
    print(f"VAE Skip Connections: encoder channels = {enc_ch_list}, decoder channels (aligned) = {dec_ch_aligned}")
    
    skip_convs = nn.ModuleList()
    for i, (enc_ch, dec_ch) in enumerate(zip(enc_ch_list, dec_ch_aligned)):
        skip_conv = nn.Conv2d(enc_ch, dec_ch, kernel_size=1, stride=1, bias=False).to(device=device, dtype=DTYPE)
        with torch.no_grad():
            skip_conv.weight.fill_(SKIP_CONV_INIT)
        skip_convs.append(skip_conv)
        print(f"skip_conv[{i}]: {enc_ch} → {dec_ch}")
    
    vae.decoder.skip_convs = skip_convs
    vae.decoder.skip_enc_channels = enc_ch_list
    vae.decoder.skip_dec_channels = dec_ch_aligned
    
    def decoder_forward_with_skip(sample, latent_embeds=None):
        sample = vae.decoder.conv_in(sample)
        upscale_dtype = next(iter(vae.decoder.up_blocks.parameters())).dtype
        
        if vae.decoder.mid_block is not None:
            sample = vae.decoder.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
        
        if not vae.decoder.ignore_skip and vae.decoder.incoming_skip_acts is not None:
            # encoder에서 뽑아 넣어준 feature list
            skip_acts = vae.decoder.incoming_skip_acts
            gamma = vae.decoder.gamma.to(dtype=sample.dtype)
            # skip level 수
            num_skip = len(skip_acts)
            
            # up block에 skip 주입
            for idx, up_block in enumerate(vae.decoder.up_blocks):
                sample = up_block(sample, latent_embeds)
                
                skip_idx = num_skip - 1 - idx
                if skip_idx >= 0 and skip_idx < len(vae.decoder.skip_convs):
                    skip_feat = skip_acts[skip_idx]
                    skip_conv = vae.decoder.skip_convs[skip_idx]
                    
                    if skip_feat is not None:
                        enc_ch = vae.decoder.skip_enc_channels[skip_idx]
                        if skip_feat.shape[1] != enc_ch:
                            raise RuntimeError(
                                f"Skip feature channel mismatch at idx={skip_idx}: "
                                f"expected {enc_ch} channels from encoder, got {skip_feat.shape[1]}"
                            )
                        
                        if skip_feat.shape[2:] != sample.shape[2:]:
                            skip_feat = F.interpolate(skip_feat, size=sample.shape[2:], mode='bilinear', align_corners=False)
                        
                        skip_feat = skip_conv(skip_feat)
                        skip_feat = skip_feat.to(dtype=sample.dtype)
                        
                        sample = sample + gamma * skip_feat
        else:
            for idx, up_block in enumerate(vae.decoder.up_blocks):
                sample = up_block(sample, latent_embeds)
        
        if latent_embeds is None:
            sample = vae.decoder.conv_norm_out(sample)
        else:
            sample = vae.decoder.conv_norm_out(sample, latent_embeds)
        sample = vae.decoder.conv_act(sample)
        sample = vae.decoder.conv_out(sample)
        return sample
    
    vae.decoder.forward = decoder_forward_with_skip
    
    return vae
