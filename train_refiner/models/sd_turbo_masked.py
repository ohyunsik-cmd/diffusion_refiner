import torch
import torch.nn as nn

class SDTurboMaskedConvIn(torch.nn.Module):
    def __init__(self, original_conv_in):
        super().__init__()
        self.conv1 = original_conv_in
        self.conv1_mask = torch.nn.Conv2d(1, 320, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.conv1_mask.weight)
        if self.conv1_mask.bias is not None:
            torch.nn.init.zeros_(self.conv1_mask.bias)
            
        # ✨ 핵심: 마스크를 임시로 저장할 공간을 만듭니다.
        self.current_mask = None 

    def forward(self, x):
        # UNet은 x(latent)만 넘겨주지만, 우리는 저장해둔 마스크를 꺼내 씁니다!
        out = self.conv1(x)
        if self.current_mask is not None:
            out = out + self.conv1_mask(self.current_mask)
        return out