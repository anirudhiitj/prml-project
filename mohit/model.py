"""
RCNN Model for Cocktail Party Audio Source Separation.

Architecture:
    1. Conv2D Encoder   — extracts local spectral features from magnitude STFT
    2. BiLSTM Core      — captures long-range temporal dependencies
    3. ConvTranspose2D Decoder — reconstructs per-source separation masks
    4. Mask Application — sigmoid masks applied to mixture spectrogram

The model operates in the time-frequency domain (STFT magnitude spectrograms).
Input:  (batch, 1, freq_bins, time_frames) — mixture magnitude spectrogram
Output: (batch, n_sources, freq_bins, time_frames) — estimated masks per source
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """ConvTranspose2D → BatchNorm → ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class RCNNSeparator(nn.Module):
    """
    Recurrent Convolutional Neural Network (RCNN) for audio source separation.

    The model follows an encoder-recurrent-decoder paradigm:
        - Encoder: Stack of 2D convolutional layers to extract spectral features.
        - Recurrent: Bidirectional LSTM to model temporal dependencies.
        - Decoder: Stack of transposed convolutions to reconstruct masks.

    Args:
        n_fft (int): FFT size used for STFT. freq_bins = n_fft // 2 + 1.
        n_sources (int): Number of sources to separate (default: 2).
        encoder_channels (list): Channel sizes for encoder conv layers.
        lstm_hidden (int): Hidden size per direction in BiLSTM.
        lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for LSTM.
    """

    def __init__(
        self,
        n_fft=512,
        n_sources=2,
        encoder_channels=None,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [1, 32, 64, 128]

        self.n_fft = n_fft
        self.n_sources = n_sources
        self.freq_bins = n_fft // 2 + 1  # 257 for n_fft=512

        # ─── Encoder ───
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            # Use stride=2 in frequency dim for downsampling, keep time dim
            stride = (2, 1)
            padding = (1, 1)
            encoder_layers.append(ConvBlock(in_ch, out_ch, kernel_size=3,
                                           stride=stride, padding=padding))
        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the frequency dimension after encoder downsampling
        freq_after_enc = self.freq_bins
        for _ in range(len(encoder_channels) - 1):
            freq_after_enc = (freq_after_enc + 2 * 1 - 3) // 2 + 1  # stride=2, kernel=3, pad=1

        self.freq_after_enc = freq_after_enc
        self.enc_out_channels = encoder_channels[-1]

        # ─── Recurrent Core (BiLSTM) ───
        # Reshape: (batch, channels, freq', time) → (batch, time, channels * freq')
        lstm_input_size = self.enc_out_channels * self.freq_after_enc

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Project LSTM output back to encoder feature space
        self.lstm_proj = nn.Linear(lstm_hidden * 2, lstm_input_size)

        # ─── Decoder ───
        decoder_channels = list(reversed(encoder_channels))
        # We multiply the first decoder input channels by 1 (just from LSTM output)
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            stride = (2, 1)
            padding = (1, 1)
            output_padding = (1, 0)
            if i == len(decoder_channels) - 2:
                # Last decoder layer: output raw features (no batchnorm/relu applied later)
                decoder_layers.append(DeconvBlock(in_ch, out_ch, kernel_size=3,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding))
            else:
                decoder_layers.append(DeconvBlock(in_ch, out_ch, kernel_size=3,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding))
        self.decoder = nn.Sequential(*decoder_layers)

        # ─── Mask Generation ───
        # 1 channel from decoder → n_sources masks
        self.mask_conv = nn.Conv2d(1, n_sources, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, freq_bins, time_frames) — mixture magnitude spectrogram

        Returns:
            masks: (batch, n_sources, freq_bins, time_frames) — estimated masks
        """
        batch_size, _, freq, time = x.shape

        # ─── Encoder ───
        enc_out = self.encoder(x)  # (batch, C, freq', time)

        # ─── Reshape for LSTM ───
        B, C, F_enc, T = enc_out.shape
        # (batch, C, freq', time) → (batch, time, C * freq')
        lstm_in = enc_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F_enc)

        # ─── Recurrent Core ───
        lstm_out, _ = self.lstm(lstm_in)  # (batch, time, 2 * lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)  # (batch, time, C * freq')

        # ─── Reshape back ───
        # (batch, time, C * freq') → (batch, C, freq', time)
        dec_in = lstm_out.view(B, T, C, F_enc).permute(0, 2, 3, 1).contiguous()

        # ─── Add skip connection from encoder ───
        dec_in = dec_in + enc_out

        # ─── Decoder ───
        dec_out = self.decoder(dec_in)  # (batch, 1, freq_reconstructed, time)

        # ─── Adjust frequency dimension to match input ───
        # The decoder output frequency may differ slightly from input due to
        # stride/padding arithmetic; we crop or pad to match.
        freq_dec = dec_out.shape[2]
        if freq_dec > freq:
            dec_out = dec_out[:, :, :freq, :]
        elif freq_dec < freq:
            pad_size = freq - freq_dec
            dec_out = F.pad(dec_out, (0, 0, 0, pad_size))

        # ─── Mask Generation ───
        masks = self.mask_conv(dec_out)  # (batch, n_sources, freq, time)
        masks = torch.sigmoid(masks)  # Soft mask in [0, 1]

        return masks


class RCNNSeparatorWaveform(nn.Module):
    """
    End-to-end wrapper that takes waveforms, computes STFT internally,
    applies the RCNN separator, and returns separated waveforms.

    This is used during inference for convenience.
    """

    def __init__(self, separator, stft_helper):
        super().__init__()
        self.separator = separator
        self.stft_helper = stft_helper

    def forward(self, mixture_waveform):
        """
        Args:
            mixture_waveform: (batch, samples) or (samples,)

        Returns:
            separated_waveforms: (batch, n_sources, samples) or (n_sources, samples)
        """
        single = mixture_waveform.dim() == 1
        if single:
            mixture_waveform = mixture_waveform.unsqueeze(0)

        length = mixture_waveform.shape[-1]

        # Compute STFT
        mag, phase = self.stft_helper.stft(mixture_waveform)
        # (batch, freq, time) → (batch, 1, freq, time)
        mag_input = mag.unsqueeze(1)

        # Get masks
        masks = self.separator(mag_input)  # (batch, n_sources, freq, time)

        # Apply masks and reconstruct
        separated = []
        for s in range(masks.shape[1]):
            est_mag = masks[:, s] * mag  # (batch, freq, time)
            est_waveform = self.stft_helper.istft(est_mag, phase, length=length)
            separated.append(est_waveform)

        separated = torch.stack(separated, dim=1)  # (batch, n_sources, samples)

        if single:
            separated = separated.squeeze(0)

        return separated


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick test
    model = RCNNSeparator(n_fft=512, n_sources=2)
    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    # Test forward pass
    x = torch.randn(2, 1, 257, 250)  # batch=2, 1 channel, 257 freq bins, 250 time frames
    masks = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {masks.shape}")
    print(f"Mask range:   [{masks.min().item():.4f}, {masks.max().item():.4f}]")
