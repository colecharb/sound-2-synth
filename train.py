"""
train.py — Stage 1 Pre-training on Synthetic Data

Loads cached embeddings and parameters, trains the ParameterPredictor MLP
to predict synth parameters from audio embeddings using multi-scale spectral loss.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from model import ParameterPredictor
from loss import create_spectral_loss
from synth import FMSynth


def load_dataset(dataset_path: str, train_split: float = 0.8) -> tuple:
    """Load cached dataset and split into train/val.
    
    Parameters
    ----------
    dataset_path : str
        Path to the .pt file containing embeddings and parameters
    train_split : float
        Fraction of data to use for training (rest goes to validation)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
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
    
    for batch_idx, (embeddings, target_params) in enumerate(train_loader):
        embeddings = embeddings.to(device)
        target_params = target_params.to(device)
        
        # Predict parameters
        predicted_params = model(embeddings)
        
        # Render audio from both target and predicted parameters
        target_audio_list = []
        predicted_audio_list = []
        
        for i in range(target_params.shape[0]):
            # Target audio
            with torch.no_grad():
                target_audio = synth.forward(
                    carrier_freq=float(target_params[i, 0]),
                    mod_ratio=float(target_params[i, 1]),
                    mod_index=float(target_params[i, 2]),
                    attack=float(target_params[i, 3]),
                    decay=float(target_params[i, 4]),
                    sustain=float(target_params[i, 5]),
                    release=float(target_params[i, 6]),
                    note_duration=float(target_params[i, 7]),
                    lowpass_freq=float(target_params[i, 8]),
                    highpass_freq=float(target_params[i, 9]),
                )
            target_audio_list.append(target_audio)
            
            # Predicted audio
            predicted_audio = synth.forward(
                carrier_freq=float(predicted_params[i, 0]),
                mod_ratio=float(predicted_params[i, 1]),
                mod_index=float(predicted_params[i, 2]),
                attack=float(predicted_params[i, 3]),
                decay=float(predicted_params[i, 4]),
                sustain=float(predicted_params[i, 5]),
                release=float(predicted_params[i, 6]),
                note_duration=float(predicted_params[i, 7]),
                lowpass_freq=float(predicted_params[i, 8]),
                highpass_freq=float(predicted_params[i, 9]),
            )
            predicted_audio_list.append(predicted_audio)
        
        # Pad to same length (use max length)
        max_len = max(a.shape[0] for a in target_audio_list + predicted_audio_list)
        target_audio_padded = []
        predicted_audio_padded = []
        
        for target_a, pred_a in zip(target_audio_list, predicted_audio_list):
            pad_target = torch.nn.functional.pad(target_a, (0, max_len - target_a.shape[0]))
            pad_pred = torch.nn.functional.pad(pred_a, (0, max_len - pred_a.shape[0]))
            target_audio_padded.append(pad_target)
            predicted_audio_padded.append(pad_pred)
        
        target_audio_batch = torch.stack(target_audio_padded).to(device)
        predicted_audio_batch = torch.stack(predicted_audio_padded).to(device)
        
        # Compute loss
        loss = spectral_loss_fn(target_audio_batch, predicted_audio_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
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
    
    with torch.no_grad():
        for embeddings, target_params in val_loader:
            embeddings = embeddings.to(device)
            target_params = target_params.to(device)
            
            # Predict parameters
            predicted_params = model(embeddings)
            
            # Render audio (same as in training)
            target_audio_list = []
            predicted_audio_list = []
            
            for i in range(target_params.shape[0]):
                target_audio = synth.forward(
                    carrier_freq=float(target_params[i, 0]),
                    mod_ratio=float(target_params[i, 1]),
                    mod_index=float(target_params[i, 2]),
                    attack=float(target_params[i, 3]),
                    decay=float(target_params[i, 4]),
                    sustain=float(target_params[i, 5]),
                    release=float(target_params[i, 6]),
                    note_duration=float(target_params[i, 7]),
                    lowpass_freq=float(target_params[i, 8]),
                    highpass_freq=float(target_params[i, 9]),
                )
                target_audio_list.append(target_audio)
                
                predicted_audio = synth.forward(
                    carrier_freq=float(predicted_params[i, 0]),
                    mod_ratio=float(predicted_params[i, 1]),
                    mod_index=float(predicted_params[i, 2]),
                    attack=float(predicted_params[i, 3]),
                    decay=float(predicted_params[i, 4]),
                    sustain=float(predicted_params[i, 5]),
                    release=float(predicted_params[i, 6]),
                    note_duration=float(predicted_params[i, 7]),
                    lowpass_freq=float(predicted_params[i, 8]),
                    highpass_freq=float(predicted_params[i, 9]),
                )
                predicted_audio_list.append(predicted_audio)
            
            # Pad
            max_len = max(a.shape[0] for a in target_audio_list + predicted_audio_list)
            target_audio_padded = []
            predicted_audio_padded = []
            
            for target_a, pred_a in zip(target_audio_list, predicted_audio_list):
                pad_target = torch.nn.functional.pad(target_a, (0, max_len - target_a.shape[0]))
                pad_pred = torch.nn.functional.pad(pred_a, (0, max_len - pred_a.shape[0]))
                target_audio_padded.append(pad_target)
                predicted_audio_padded.append(pad_pred)
            
            target_audio_batch = torch.stack(target_audio_padded).to(device)
            predicted_audio_batch = torch.stack(predicted_audio_padded).to(device)
            
            loss = spectral_loss_fn(target_audio_batch, predicted_audio_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train parameter predictor (Stage 1)")
    parser.add_argument("--dataset", type=str, default="data/synthetic_5k.pt", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Load data
    train_loader, val_loader, synth, param_names = load_dataset(args.dataset)
    
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
