"""
train.py — Stage 1 Pre-training on Synthetic Data

Loads cached embeddings and parameters, trains the ParameterPredictor MLP
to predict synth parameters from audio embeddings using multi-scale spectral loss.
"""

import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", message=".*was resized.*")
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import ParameterPredictor
from loss import create_spectral_loss
from synth import FMSynth


def get_device() -> tuple[torch.device, bool]:
    """
    Detect the best device for training.
    
    Returns
    -------
    tuple[torch.device, bool]
        (device, fft_on_cpu) where fft_on_cpu indicates if FFT ops need to run on CPU
    """
    # Prefer CUDA if available
    if torch.cuda.is_available():
        return torch.device("cuda"), False
    
    # Check if MPS is available and actually works on this hardware
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Try a quick test to see if MPS actually works
        try:
            # Test if FFT works on MPS
            test_tensor = torch.randn(2, 100, device="mps")
            torch.fft.rfft(test_tensor)
            # If we get here, FFT works on MPS
            return torch.device("mps"), False
        except (RuntimeError, NotImplementedError):
            # MPS is available but doesn't support FFT (Intel Mac)
            # Use CPU for main ops but mark that FFT needs special handling
            return torch.device("cpu"), False
    
    # Fallback to CPU
    return torch.device("cpu"), False


def load_dataset(dataset_path: str, train_split: float = 0.8, batch_size: int = 64) -> tuple:
    """Load cached dataset and split into train/val.
    
    Parameters
    ----------
    dataset_path : str
        Path to the .pt file containing embeddings and parameters
    train_split : float
        Fraction of data to use for training (rest goes to validation)
    batch_size : int
        Mini-batch size for DataLoaders
    
    Returns
    -------
    tuple
        (train_loader, val_loader, synth, scaler_info)
    """
    print(f"\nLoading dataset from {dataset_path}...")
    
    # Load data
    dataset_dict = torch.load(dataset_path, map_location="cpu")
    embeddings = dataset_dict["embeddings"]
    params = dataset_dict["params"]
    param_names = dataset_dict.get("param_names", None)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Parameters shape: {params.shape}")
    
    # Train/val split
    num_samples = embeddings.shape[0]
    num_train = int(num_samples * train_split)
    
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_embeddings = embeddings[train_indices]
    train_params = params[train_indices]
    val_embeddings = embeddings[val_indices]
    val_params = params[val_indices]
    
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(train_embeddings, train_params)
    val_dataset = TensorDataset(val_embeddings, val_params)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create synth for rendering
    synth = FMSynth(sample_rate=44100)
    
    return train_loader, val_loader, synth, param_names


def train_epoch(
    model: nn.Module,
    synth: FMSynth,
    spectral_loss_fn: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.
    
    Returns
    -------
    float
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
    for batch_idx, (embeddings, target_params) in pbar:
        embeddings = embeddings.to(device)
        target_params = target_params.to(device)
        
        predicted_params = model(embeddings)
        
        with torch.no_grad():
            target_audio_batch = synth.forward_batch(target_params)
        predicted_audio_batch = synth.forward_batch(predicted_params)
        
        # Pad to same time-axis length (durations may differ)
        t_len = target_audio_batch.shape[1]
        p_len = predicted_audio_batch.shape[1]
        if t_len < p_len:
            target_audio_batch = nn.functional.pad(target_audio_batch, (0, p_len - t_len))
        elif p_len < t_len:
            predicted_audio_batch = nn.functional.pad(predicted_audio_batch, (0, t_len - p_len))
        
        # STFT crashes on MPS; run spectral loss on CPU (free on Apple
        # Silicon unified memory, and gradients propagate back automatically)
        loss = spectral_loss_fn(target_audio_batch.cpu(), predicted_audio_batch.cpu())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    synth: FMSynth,
    spectral_loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate on held-out set.
    
    Returns
    -------
    float
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_idx, (embeddings, target_params) in enumerate(pbar):
            embeddings = embeddings.to(device)
            target_params = target_params.to(device)
            
            predicted_params = model(embeddings)
            
            target_audio_batch = synth.forward_batch(target_params)
            predicted_audio_batch = synth.forward_batch(predicted_params)
            
            t_len = target_audio_batch.shape[1]
            p_len = predicted_audio_batch.shape[1]
            if t_len < p_len:
                target_audio_batch = nn.functional.pad(target_audio_batch, (0, p_len - t_len))
            elif p_len < t_len:
                predicted_audio_batch = nn.functional.pad(predicted_audio_batch, (0, t_len - p_len))
            
            loss = spectral_loss_fn(target_audio_batch.cpu(), predicted_audio_batch.cpu())
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train parameter predictor (Stage 1)")
    parser.add_argument("--dataset", type=str, default="data/synthetic_5k.pt", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    args = parser.parse_args()
    
    device, _ = get_device()
    print(f"\nUsing device: {device}\n")
    
    # Load data
    train_loader, val_loader, synth, param_names = load_dataset(args.dataset, batch_size=args.batch_size)
    
    # Create model
    model = ParameterPredictor(embedding_dim=512, dropout=0.1)
    model = model.to(device)
    synth = synth.to(device)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    spectral_loss_fn = create_spectral_loss()
    spectral_loss_fn = spectral_loss_fn.to(device)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}")
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, synth, spectral_loss_fn, train_loader, optimizer, device)
        val_loss = validate(model, synth, spectral_loss_fn, val_loader, device)
        
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  |  "
              f"Train: {train_loss:.4f}  |  Val: {val_loss:.4f}", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(" ✓ (best)")
        else:
            print()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
