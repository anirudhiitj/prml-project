import torch
import torch.nn as nn
from asteroid.models import ConvTasNet

class SpeechSeparator(nn.Module):
    def __init__(self, n_src=2, sample_rate=8000, 
                 n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512):
        super().__init__()
        # Initialize ConvTasNet using Asteroid which provides a robust implementation
        # n_src specifies the number of output speakers.
        self.model = ConvTasNet(
            n_src=n_src, 
            sample_rate=sample_rate,
            n_filters=512,
            kernel_size=16,
            stride=8,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            in_channels=1
        )
        
    def forward(self, mix):
        """
        Args:
            mix: Tensor of shape (batch, time) or (batch, channels, time)
        Returns:
            separated: Tensor of shape (batch, n_src, time)
        """
        # Asteroid's ConvTasNet expects (batch, time) or (batch, 1, time)
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
            
        estimates = self.model(mix)
        return estimates

if __name__ == "__main__":
    # Quick test
    dummy_mix = torch.randn(2, 16000) # 2 samples, 2 seconds at 8kHz
    model = SpeechSeparator(n_src=2)
    sep = model(dummy_mix)
    print(f"Mix shape: {dummy_mix.shape}")
    print(f"Separated shape: {sep.shape}")
