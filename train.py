"""
train.py — Two-stage training for the parameter predictor

Stage A: Pure parameter-space MSE regression (fast, no synth rendering).
         Trains the MLP to predict normalised synth params from OpenL3
         embeddings.  Runs in seconds per epoch.

Stage B: Spectral fine-tuning.  Loads a Stage A checkpoint, renders
         predicted audio through the differentiable synth, and compares
         against precomputed target log-magnitude STFTs.  This refines
         the perceptual quality beyond what param MSE alone can achieve.

Usage:
  # Stage A — fast param regression
  python train.py --stage A --dataset data/synthetic_20k.pt --epochs 80

  # Stage B — spectral fine-tuning (loads Stage A checkpoint)
  python train.py --stage B --dataset data/synthetic_20k.pt --epochs 40 \\
      --checkpoint checkpoints/stage_a.pt
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

import wandb

from model import ParameterPredictor, PARAM_NAMES, PARAM_RANGES, LOG_SCALE_PARAMS
from synth import FMSynth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_params(params: torch.Tensor) -> torch.Tensor:
    """Normalize parameters to [0, 1] per-dimension for comparable MSE."""
    normed = torch.zeros_like(params)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        col = params[:, i]
        if name in LOG_SCALE_PARAMS:
            col = torch.log(col.clamp(min=lo))
            lo, hi = math.log(lo), math.log(hi)
        normed[:, i] = (col - lo) / (hi - lo)
    return normed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SAMPLE_RATE = 44100
NUM_AUDIO_SAMPLES = 4
SPEC_FFT_SIZES = [512, 1024, 2048]
SPEC_HOP_RATIO = 0.25


def compute_spectral_loss(
    target_audio: torch.Tensor,
    predicted_audio: torch.Tensor,
    fft_sizes: list[int] = None,
    hop_ratio: float = SPEC_HOP_RATIO,
) -> torch.Tensor:
    """Multi-scale spectral loss on raw audio pairs (no precomputed specs)."""
    if fft_sizes is None:
        fft_sizes = SPEC_FFT_SIZES

    min_len = min(target_audio.shape[-1], predicted_audio.shape[-1])
    target_audio = target_audio[..., :min_len]
    predicted_audio = predicted_audio[..., :min_len]

    total = 0.0
    for fft_size in fft_sizes:
        hop = int(fft_size * hop_ratio)
        window = torch.hann_window(fft_size, device=target_audio.device)
        eps_sq = 1e-14

        t_stft = torch.stft(target_audio, n_fft=fft_size, hop_length=hop,
                            window=window, return_complex=True)
        p_stft = torch.stft(predicted_audio, n_fft=fft_size, hop_length=hop,
                            window=window, return_complex=True)

        t_lm = torch.log((t_stft.real ** 2 + t_stft.imag ** 2 + eps_sq).sqrt())
        p_lm = torch.log((p_stft.real ** 2 + p_stft.imag ** 2 + eps_sq).sqrt())
        total = total + torch.mean(torch.abs(t_lm - p_lm))

    return total / len(fft_sizes)


@torch.no_grad()
def log_audio_to_wandb(
    model: nn.Module,
    synth: FMSynth,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    num_samples: int = NUM_AUDIO_SAMPLES,
) -> float:
    """Render val examples, log audio to W&B, return spectral loss on those samples."""
    model.eval()

    batch = next(iter(val_loader))
    embeddings = batch[0][:num_samples].to(device)
    target_params = batch[1][:num_samples].to(device)

    pred_params = model(embeddings)

    target_audio = synth.forward_batch(target_params).cpu()
    pred_audio = synth.forward_batch(pred_params).cpu()

    # Spectral loss on the rendered samples
    spec_loss = compute_spectral_loss(target_audio, pred_audio).item()

    audio_logs = {}
    for i in range(min(num_samples, embeddings.shape[0])):
        tgt = target_audio[i].float().numpy()
        pred = pred_audio[i].float().numpy()

        tp = target_params[i].cpu()
        pp = pred_params[i].cpu()
        caption = (
            f"carrier: {tp[0]:.0f}→{pp[0]:.0f}Hz  "
            f"ratio: {tp[1]:.2f}→{pp[1]:.2f}  "
            f"index: {tp[2]:.1f}→{pp[2]:.1f}"
        )

        audio_logs[f"audio/sample_{i}_target"] = wandb.Audio(
            tgt, sample_rate=SAMPLE_RATE, caption=f"[target] {caption}",
        )
        audio_logs[f"audio/sample_{i}_predicted"] = wandb.Audio(
            pred, sample_rate=SAMPLE_RATE, caption=f"[predicted] {caption}",
        )

    audio_logs["val/spectral_loss_preview"] = spec_loss
    wandb.log(audio_logs, step=epoch + 1)

    return spec_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_path: str,
    train_split: float = 0.8,
    batch_size: int = 128,
    stage: str = "A",
):
    """Load dataset and build dataloaders.

    For Stage A only embeddings + params are needed.
    For Stage B the precomputed target specs are also loaded.
    """
    print(f"\nLoading dataset from {dataset_path}...")
    data = torch.load(dataset_path, map_location="cpu")
    embeddings = data["embeddings"]
    params = data["params"]

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Params:     {params.shape}")

    n = embeddings.shape[0]
    n_train = int(n * train_split)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    if stage == "A":
        train_ds = TensorDataset(embeddings[train_idx], params[train_idx])
        val_ds = TensorDataset(embeddings[val_idx], params[val_idx])
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            None,  # no target specs for stage A
        )

    # Stage B — also load precomputed target specs
    target_specs = data.get("target_specs", None)
    if target_specs is None:
        raise ValueError(
            "Dataset has no precomputed target_specs.  "
            "Regenerate with the updated dataset.py."
        )

    # Build tensors: (embeddings, params, spec_512, spec_1024, spec_2048)
    spec_tensors_train = [target_specs[s][train_idx] for s in sorted(target_specs)]
    spec_tensors_val = [target_specs[s][val_idx] for s in sorted(target_specs)]

    train_ds = TensorDataset(
        embeddings[train_idx], params[train_idx], *spec_tensors_train,
    )
    val_ds = TensorDataset(
        embeddings[val_idx], params[val_idx], *spec_tensors_val,
    )

    fft_sizes = sorted(target_specs.keys())
    hop_ratio = data.get("spec_hop_ratio", 0.25)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Cached target specs: FFT sizes {fft_sizes}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        {"fft_sizes": fft_sizes, "hop_ratio": hop_ratio},
    )


# ---------------------------------------------------------------------------
# Stage A — param-space regression
# ---------------------------------------------------------------------------

def train_epoch_A(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for embeddings, target_params in loader:
        embeddings = embeddings.to(device)
        target_params = target_params.to(device)
        pred = model(embeddings)
        loss = nn.functional.mse_loss(
            normalize_params(pred), normalize_params(target_params),
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate_A(model, loader, device, synth=None, compute_spec=False):
    """Validate Stage A.  Optionally renders audio and computes spectral loss."""
    model.eval()
    total_mse = 0.0
    total_spec = 0.0
    n_batches = 0

    for embeddings, target_params in loader:
        embeddings = embeddings.to(device)
        target_params = target_params.to(device)
        pred = model(embeddings)
        total_mse += nn.functional.mse_loss(
            normalize_params(pred), normalize_params(target_params),
        ).item()

        if compute_spec and synth is not None:
            tgt_audio = synth.forward_batch(target_params).cpu()
            pred_audio = synth.forward_batch(pred).cpu()
            total_spec += compute_spectral_loss(tgt_audio, pred_audio).item()

        n_batches += 1

    avg_mse = total_mse / n_batches
    avg_spec = total_spec / n_batches if compute_spec else None
    return avg_mse, avg_spec


def init_wandb(args, stage: str):
    """Initialise a W&B run if --wandb is set, otherwise return None."""
    if not args.wandb:
        return None
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "stage": stage,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "param_weight": args.param_weight if stage == "B" else None,
            "param_anneal_epochs": args.param_anneal_epochs if stage == "B" else None,
        },
    )


def run_stage_a(args):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = load_dataset(
        args.dataset, batch_size=args.batch_size, stage="A",
    )

    model = ParameterPredictor(embedding_dim=512, dropout=0.1).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_epoch = 0
    output = args.output or "checkpoints/stage_a.pt"

    run = init_wandb(args, stage="A")
    synth_for_audio = FMSynth(sample_rate=SAMPLE_RATE).to(device)

    print(f"\n{'=' * 60}")
    print(f"Stage A — param-space regression for {args.epochs} epochs")
    print(f"{'=' * 60}")

    spec_interval = 5

    for epoch in range(args.epochs):
        t_loss = train_epoch_A(model, train_loader, optimizer, device)

        # Full spectral val every spec_interval epochs (rendering is expensive)
        do_spec = (epoch + 1) % spec_interval == 0 or epoch == 0
        v_mse, v_spec = validate_A(
            model, val_loader, device,
            synth=synth_for_audio, compute_spec=do_spec,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        is_best = False
        if v_mse < best_val:
            best_val = v_mse
            best_epoch = epoch
            is_best = True
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)

        spec_str = f"  spec={v_spec:.3f}" if v_spec is not None else ""
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  |  "
            f"Train: {t_loss:.6f}  Val: {v_mse:.6f}{spec_str}  "
            f"LR={lr:.2e}{' ✓' if is_best else ''}"
        )

        if run:
            log_dict = {
                "epoch": epoch + 1,
                "train/param_loss": t_loss,
                "val/param_loss": v_mse,
                "lr": lr,
                "best_val_param_loss": best_val,
            }
            if v_spec is not None:
                log_dict["val/spectral_loss"] = v_spec
            wandb.log(log_dict)

            if is_best or (epoch + 1) % 10 == 0:
                log_audio_to_wandb(
                    model, synth_for_audio, val_loader, device, epoch,
                )

    if run:
        wandb.summary["best_val_mse"] = best_val
        wandb.summary["best_epoch"] = best_epoch + 1
        art = wandb.Artifact(f"stage-a-best", type="model")
        art.add_file(output)
        wandb.log_artifact(art)
        wandb.finish()

    print(f"\nBest val MSE: {best_val:.6f} (epoch {best_epoch+1})")
    print(f"Checkpoint: {output}\n")


# ---------------------------------------------------------------------------
# Stage B — spectral fine-tuning
# ---------------------------------------------------------------------------

def spectral_loss_with_cached_targets(
    predicted_audio: torch.Tensor,
    target_log_mags: list[torch.Tensor],
    fft_sizes: list[int],
    hop_ratio: float,
) -> torch.Tensor:
    """Compute multi-scale spectral loss using precomputed target log-mags.

    Only the predicted audio needs an STFT — the target side is free.
    """
    total = 0.0
    for target_lm, fft_size in zip(target_log_mags, fft_sizes):
        hop = int(fft_size * hop_ratio)
        window = torch.hann_window(fft_size, device=predicted_audio.device)
        pred_stft = torch.stft(
            predicted_audio, n_fft=fft_size, hop_length=hop,
            window=window, return_complex=True,
        )
        eps_sq = 1e-14
        pred_mag = (pred_stft.real ** 2 + pred_stft.imag ** 2 + eps_sq).sqrt()
        pred_lm = torch.log(pred_mag)

        # Trim to shorter time axis (predicted duration may differ)
        t_min = min(pred_lm.shape[-1], target_lm.shape[-1])
        total = total + torch.mean(
            torch.abs(pred_lm[:, :, :t_min] - target_lm[:, :, :t_min])
        )
    return total / len(fft_sizes)


def train_epoch_B(
    model, synth, loader, optimizer, device,
    fft_sizes, hop_ratio, param_weight,
):
    model.train()
    total_loss = 0.0
    total_spec = 0.0
    total_param = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        embeddings = batch[0].to(device)
        target_params = batch[1].to(device)
        target_lms = [batch[2 + i].to("cpu") for i in range(len(fft_sizes))]

        pred_params = model(embeddings)

        # Param MSE (cheap)
        p_loss = torch.tensor(0.0, device=device)
        if param_weight > 0:
            p_loss = nn.functional.mse_loss(
                normalize_params(pred_params), normalize_params(target_params),
            )

        # Render predicted audio through synth
        pred_audio = synth.forward_batch(pred_params)

        # STFT on CPU (MPS doesn't support it; unified memory = free copy)
        s_loss = spectral_loss_with_cached_targets(
            pred_audio.cpu(), target_lms, fft_sizes, hop_ratio,
        )

        loss = s_loss + param_weight * p_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_spec += s_loss.item()
        total_param += p_loss.item()
        pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}")

    n = len(loader)
    return total_loss / n, total_spec / n, total_param / n


@torch.no_grad()
def validate_B(model, synth, loader, device, fft_sizes, hop_ratio, param_weight):
    model.eval()
    total_loss = 0.0
    total_spec = 0.0
    total_param = 0.0

    for batch in loader:
        embeddings = batch[0].to(device)
        target_params = batch[1].to(device)
        target_lms = [batch[2 + i].to("cpu") for i in range(len(fft_sizes))]

        pred_params = model(embeddings)

        p_loss = torch.tensor(0.0)
        if param_weight > 0:
            p_loss = nn.functional.mse_loss(
                normalize_params(pred_params), normalize_params(target_params),
            )

        pred_audio = synth.forward_batch(pred_params)
        s_loss = spectral_loss_with_cached_targets(
            pred_audio.cpu(), target_lms, fft_sizes, hop_ratio,
        )

        loss = s_loss + param_weight * p_loss
        total_loss += loss.item()
        total_spec += s_loss.item()
        total_param += p_loss.item()

    n = len(loader)
    return total_loss / n, total_spec / n, total_param / n


def run_stage_b(args):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, spec_info = load_dataset(
        args.dataset, batch_size=args.batch_size, stage="B",
    )
    fft_sizes = spec_info["fft_sizes"]
    hop_ratio = spec_info["hop_ratio"]

    model = ParameterPredictor(embedding_dim=512, dropout=0.1).to(device)

    # Load Stage A checkpoint
    ckpt = args.checkpoint or "checkpoints/stage_a.pt"
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: no checkpoint at {ckpt} — training from scratch")

    synth = FMSynth(sample_rate=44100).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pw_start = args.param_weight
    pw_anneal = args.param_anneal_epochs
    output = args.output or "checkpoints/stage_b.pt"

    best_val_spec = float("inf")
    best_epoch = 0

    run = init_wandb(args, stage="B")

    print(f"\n{'=' * 60}")
    print(f"Stage B — spectral fine-tuning for {args.epochs} epochs")
    print(f"  Param-MSE weight: {pw_start} → 0 over {pw_anneal} epochs")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        pw = pw_start * max(0.0, 1.0 - epoch / pw_anneal) if pw_anneal > 0 else 0.0

        t_loss, t_spec, t_param = train_epoch_B(
            model, synth, train_loader, optimizer, device,
            fft_sizes, hop_ratio, pw,
        )
        v_loss, v_spec, v_param = validate_B(
            model, synth, val_loader, device,
            fft_sizes, hop_ratio, pw,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        is_best = False
        if v_spec < best_val_spec:
            best_val_spec = v_spec
            best_epoch = epoch
            is_best = True
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  |  "
            f"Train: {t_loss:.4f} (spec={t_spec:.3f} param={t_param:.4f})  |  "
            f"Val: {v_loss:.4f} (spec={v_spec:.3f})  |  "
            f"pw={pw:.1f}  LR={lr:.2e}{' ✓' if is_best else ''}"
        )

        if run:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": t_loss,
                "train/spectral_loss": t_spec,
                "train/param_loss": t_param,
                "val/loss": v_loss,
                "val/spectral_loss": v_spec,
                "val/param_loss": v_param,
                "param_weight": pw,
                "lr": lr,
                "best_val_spectral": best_val_spec,
            })
            if is_best or (epoch + 1) % 5 == 0:
                log_audio_to_wandb(
                    model, synth, val_loader, device, epoch,
                )

    if run:
        wandb.summary["best_val_spectral"] = best_val_spec
        wandb.summary["best_epoch"] = best_epoch + 1
        art = wandb.Artifact(f"stage-b-best", type="model")
        art.add_file(output)
        wandb.log_artifact(art)
        wandb.finish()

    print(f"\nBest val spectral loss: {best_val_spec:.4f} (epoch {best_epoch+1})")
    print(f"Checkpoint: {output}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train parameter predictor (two-stage)",
    )
    parser.add_argument("--stage", type=str, choices=["A", "B"], required=True,
                        help="A = param regression, B = spectral fine-tune")
    parser.add_argument("--dataset", type=str, default="data/synthetic_20k.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3 for A, use ~1e-4 for B)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Checkpoint path (defaults: stage_a.pt / stage_b.pt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Stage A checkpoint to load for Stage B")
    parser.add_argument("--param_weight", type=float, default=1.0,
                        help="(Stage B) initial param-MSE weight")
    parser.add_argument("--param_anneal_epochs", type=int, default=20,
                        help="(Stage B) anneal param weight to 0 over N epochs")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", type=str, default="pocketchange",
                        help="W&B entity (your personal username or a shared team — "
                             "set this to stay off your company org)")
    parser.add_argument("--wandb_project", type=str, default="sound-2-synth",
                        help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="W&B run name (auto-generated if omitted)")

    args = parser.parse_args()

    if args.stage == "A":
        run_stage_a(args)
    else:
        run_stage_b(args)


if __name__ == "__main__":
    main()
