"""
loss.py — Multi-scale Spectral Loss

Compares STFT magnitudes at multiple scales (512, 1024, 2048 FFT sizes).
Uses log magnitude (perceptually closer to human hearing) and L1 distance.
"""

import torch
import torch.nn as nn


class MultiScaleSpectralLoss(nn.Module):
    """Compute spectral loss across multiple STFT scales.
    
    This loss compares the spectral characteristics of target and predicted audio
    at multiple frequency resolutions, providing a perceptually-motivated measure
    of similarity.
    
    Parameters
    ----------
    fft_sizes : list[int]
        FFT sizes to use for STFT computation. Default: [512, 1024, 2048]
    hop_length_ratio : float
        Hop length as a fraction of FFT size. Default: 0.25 (25% overlap)
    """
    
    def __init__(
        self,
        fft_sizes: list[int] = None,
        hop_length_ratio: float = 0.25,
    ):
        super().__init__()
        
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        
        self.fft_sizes = fft_sizes
        self.hop_length_ratio = hop_length_ratio
        
        # Pre-compute hop lengths
        self.hop_lengths = [int(fft_size * hop_length_ratio) for fft_size in fft_sizes]
    
    def forward(self, target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale spectral loss.
        
        Parameters
        ----------
        target : Tensor
            Target audio, shape (batch_size, num_samples) or (num_samples,)
        predicted : Tensor
            Predicted audio, shape (batch_size, num_samples) or (num_samples,)
        
        Returns
        -------
        Tensor
            Scalar loss value (average across scales)
        """
        # Handle unbatched inputs
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if predicted.ndim == 1:
            predicted = predicted.unsqueeze(0)
        
        total_loss = 0.0
        
        for fft_size, hop_length in zip(self.fft_sizes, self.hop_lengths):
            # Compute STFT
            target_spec = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(fft_size, device=target.device),
            )
            predicted_spec = torch.stft(
                predicted,
                n_fft=fft_size,
                hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(fft_size, device=predicted.device),
            )
            
            # Magnitude
            target_mag = torch.abs(target_spec)
            predicted_mag = torch.abs(predicted_spec)
            
            # Log magnitude (with small epsilon to avoid log(0))
            epsilon = 1e-7
            target_log_mag = torch.log(target_mag + epsilon)
            predicted_log_mag = torch.log(predicted_mag + epsilon)
            
            # L1 loss
            scale_loss = torch.mean(torch.abs(target_log_mag - predicted_log_mag))
            total_loss = total_loss + scale_loss
        
        # Average across scales
        return total_loss / len(self.fft_sizes)


# Convenience function for creating the default loss
def create_spectral_loss() -> MultiScaleSpectralLoss:
    """Create a multi-scale spectral loss with default settings."""
    return MultiScaleSpectralLoss(
        fft_sizes=[512, 1024, 2048],
        hop_length_ratio=0.25,
    )


if __name__ == "__main__":
    # Test the loss function
    loss_fn = create_spectral_loss()
    print(f"MultiScaleSpectralLoss created with FFT sizes: {loss_fn.fft_sizes}")
    
    # Test with dummy audio
    batch_size = 4
    num_samples = 44100
    target = torch.randn(batch_size, num_samples)
    predicted = target + 0.1 * torch.randn_like(target)
    
    loss = loss_fn(target, predicted)
    print(f"Test loss shape: {loss.shape}")
    print(f"Test loss value: {loss.item():.4f}")
    
    # Test with unbatched input
    loss_unbatched = loss_fn(target[0], predicted[0])
    print(f"Unbatched loss: {loss_unbatched.item():.4f}")
