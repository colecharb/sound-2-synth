"""
train.py — Stage 1 Pre-training on Synthetic Data

Loads cached embeddings and parameters, trains the ParameterPredictor MLP
to predict synth parameters from audio embeddings.

Uses a two-phase loss strategy:
  1. Early: parameter-space MSE dominates (smooth gradients to bootstrap)
  2. Late:  spectral loss dominates (perceptual fine-tuning)
The parameter-space weight is linearly annealed from `param_weight_start` to 0
over `param_anneal_epochs` epochs.
"""

import argparse
import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", message=".*was resized.*")
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import ParameterPredictor, PARAM_NAMES, PARAM_RANGES, LOG_SCALE_PARAMS
from loss import create_spectral_loss
from synth import FMSynth


def normalize_params(params: torch.Tensor) -> torch.Tensor:
    """Normalize parameters to [0, 1] per-dimension for comparable MSE.

    Log-scale parameters are normalized in log space.
    """
    normed = torch.zeros_like(params)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        col = params[:, i]
        if name in LOG_SCALE_PARAMS:
            col = torch.log(col.clamp(min=lo))
            lo, hi = math.log(lo), math.log(hi)
        normed[:, i] = (col - lo) / (hi - lo)
    return normed


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
    param_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Train for one epoch.
    
    Returns
    -------
    tuple[float, float, float]
        (total_loss, spectral_loss, param_mse_loss) averaged over batches
    """
    model.train()
    total_loss = 0.0
    total_spec = 0.0
    total_param = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
    for batch_idx, (embeddings, target_params) in pbar:
        embeddings = embeddings.to(device)
        target_params = target_params.to(device)
        
        predicted_params = model(embeddings)
        
        # --- parameter-space MSE (always cheap, no synth render needed) ---
        param_loss = torch.tensor(0.0, device=device)
        if param_weight > 0:
            pred_norm = normalize_params(predicted_params)
            tgt_norm = normalize_params(target_params)
            param_loss = nn.functional.mse_loss(pred_norm, tgt_norm)
        
        # --- spectral loss (render through synth) ---
        with torch.no_grad():
            target_audio_batch = synth.forward_batch(target_params)
        predicted_audio_batch = synth.forward_batch(predicted_params)
        
        min_len = min(target_audio_batch.shape[1], predicted_audio_batch.shape[1])
        target_audio_batch = target_audio_batch[:, :min_len]
        predicted_audio_batch = predicted_audio_batch[:, :min_len]
        
        # STFT crashes on MPS; run spectral loss on CPU (free on Apple
        # Silicon unified memory, and gradients propagate back automatically)
        spec_loss = spectral_loss_fn(target_audio_batch.cpu(), predicted_audio_batch.cpu())
        
        loss = spec_loss + param_weight * param_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_spec += spec_loss.item()
        total_param += param_loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    n = len(train_loader)
    return total_loss / n, total_spec / n, total_param / n


def validate(
    model: nn.Module,
    synth: FMSynth,
    spectral_loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    param_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Validate on held-out set.
    
    Returns
    -------
    tuple[float, float, float]
        (total_loss, spectral_loss, param_mse_loss) averaged over batches
    """
    model.eval()
    total_loss = 0.0
    total_spec = 0.0
    total_param = 0.0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_idx, (embeddings, target_params) in enumerate(pbar):
            embeddings = embeddings.to(device)
            target_params = target_params.to(device)
            
            predicted_params = model(embeddings)
            
            param_loss = torch.tensor(0.0, device=device)
            if param_weight > 0:
                pred_norm = normalize_params(predicted_params)
                tgt_norm = normalize_params(target_params)
                param_loss = nn.functional.mse_loss(pred_norm, tgt_norm)
            
            target_audio_batch = synth.forward_batch(target_params)
            predicted_audio_batch = synth.forward_batch(predicted_params)
            
            min_len = min(target_audio_batch.shape[1], predicted_audio_batch.shape[1])
            target_audio_batch = target_audio_batch[:, :min_len]
            predicted_audio_batch = predicted_audio_batch[:, :min_len]
            
            spec_loss = spectral_loss_fn(target_audio_batch.cpu(), predicted_audio_batch.cpu())
            
            loss = spec_loss + param_weight * param_loss
            total_loss += loss.item()
            total_spec += spec_loss.item()
            total_param += param_loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    n = len(val_loader)
    return total_loss / n, total_spec / n, total_param / n


def main():
    parser = argparse.ArgumentParser(description="Train parameter predictor (Stage 1)")
    parser.add_argument("--dataset", type=str, default="data/synthetic_5k.pt", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    parser.add_argument("--param_weight", type=float, default=10.0,
                        help="Initial weight for parameter-space MSE loss")
    parser.add_argument("--param_anneal_epochs", type=int, default=30,
                        help="Linearly anneal param_weight to 0 over this many epochs")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}\n")
    
    # Load data
    train_loader, val_loader, synth, param_names = load_dataset(args.dataset, batch_size=args.batch_size)
    
    # Create model
    model = ParameterPredictor(embedding_dim=512, dropout=0.1)
    model = model.to(device)
    synth = synth.to(device)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer, scheduler, and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    spectral_loss_fn = create_spectral_loss()
    spectral_loss_fn = spectral_loss_fn.to(device)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"  Param-MSE weight: {args.param_weight} → 0 over {args.param_anneal_epochs} epochs")
    print(f"{'='*60}")
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Linear anneal: param_weight → 0 over param_anneal_epochs
        if args.param_anneal_epochs > 0 and epoch < args.param_anneal_epochs:
            pw = args.param_weight * (1.0 - epoch / args.param_anneal_epochs)
        else:
            pw = 0.0
        
        train_loss, train_spec, train_param = train_epoch(
            model, synth, spectral_loss_fn, train_loader, optimizer, device,
            param_weight=pw,
        )
        val_loss, val_spec, val_param = validate(
            model, synth, spectral_loss_fn, val_loader, device,
            param_weight=pw,
        )
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}  |  "
            f"Train: {train_loss:.4f} (spec={train_spec:.3f} param={train_param:.3f})  |  "
            f"Val: {val_loss:.4f} (spec={val_spec:.3f})  |  "
            f"pw={pw:.1f}  LR={current_lr:.2e}",
            end="",
        )
        
        # Track best by spectral-only val loss (the true objective)
        if val_spec < best_val_loss:
            best_val_loss = val_spec
            best_epoch = epoch
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(" ✓ (best)")
        else:
            print()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best val spectral loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
