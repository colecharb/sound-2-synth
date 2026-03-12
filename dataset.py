"""
dataset.py — Generate synthetic training data for Stage 1 pre-training

Generates random FM synth parameters, renders audio using FMSynth,
extracts audio embeddings, and precomputes target spectrograms for
fast spectral-loss training.

Embedding types (--embedding flag):
  mel     : Mel spectrogram with mean+std pooling (256-dim, instant, no deps)
  panns   : PANNs CNN14 AudioSet features (2048-dim, requires panns_inference)
  openl3  : OpenL3 music embeddings (512-dim, requires openl3 + TensorFlow)

The dataset .pt file contains:
  - embeddings     (N, D)   audio embeddings (D depends on embedding type)
  - embedding_type str      which extractor produced the embeddings
  - embedding_dim  int      D
  - params         (N, 10)  ground-truth synth parameters
  - target_specs            dict of precomputed log-magnitude STFTs keyed by FFT size
  - param_names             list[str]

Parameter order (see PARAMETER_ORDER.md):
[carrier_freq, mod_ratio, mod_index, attack, decay, sustain, release,
 note_duration, lowpass_freq, highpass_freq]
"""

import argparse
import math
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from synth import FMSynth

PARAM_NAMES = [
    "carrier_freq",
    "mod_ratio",
    "mod_index",
    "attack",
    "decay",
    "sustain",
    "release",
    "note_duration",
    "lowpass_freq",
    "highpass_freq",
]

PARAM_RANGES = {
    "carrier_freq": (20.0, 2000.0),
    "mod_ratio": (0.5, 4.0),
    "mod_index": (0.0, 20.0),
    "attack": (0.001, 1.0),
    "decay": (0.001, 2.0),
    "sustain": (0.0, 1.0),
    "release": (0.001, 2.0),
    "note_duration": (0.05, 2.0),
    "lowpass_freq": (20.0, 20000.0),
    "highpass_freq": (20.0, 10000.0),
}

LOG_SCALE_PARAMS = {"carrier_freq", "lowpass_freq", "highpass_freq"}

SPEC_FFT_SIZES = [512, 1024, 2048]
SPEC_HOP_RATIO = 0.25
AUDIO_CLIP_SECONDS = 1.0

EMBEDDING_TYPES = ("mel", "panns", "openl3")


def sample_parameters_batch(n: int) -> torch.Tensor:
    """Sample n parameter vectors with perceptually-uniform distributions.

    - Frequencies (carrier, lowpass, highpass) are log-uniform.
    - Envelope times (attack, decay, release) are log-uniform so short
      transients are represented as well as long ones.
    - mod_ratio favours integer ratios (harmonic timbres) 60 % of the time.
    - Everything else is uniform within range.

    Returns shape (n, 10).
    """
    params = torch.zeros(n, len(PARAM_NAMES))

    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]

        if name in LOG_SCALE_PARAMS:
            log_lo, log_hi = math.log(lo), math.log(hi)
            params[:, i] = torch.empty(n).uniform_(log_lo, log_hi).exp()

        elif name in ("attack", "decay", "release"):
            log_lo, log_hi = math.log(lo), math.log(hi)
            params[:, i] = torch.empty(n).uniform_(log_lo, log_hi).exp()

        elif name == "mod_ratio":
            # 60 % integer ratios (1–4) with small jitter, 40 % continuous
            n_harmonic = int(n * 0.6)
            n_continuous = n - n_harmonic
            harmonic = torch.randint(1, 5, (n_harmonic,)).float()
            harmonic += torch.empty(n_harmonic).uniform_(-0.05, 0.05)
            continuous = torch.empty(n_continuous).uniform_(lo, hi)
            combined = torch.cat([harmonic, continuous])
            params[:, i] = combined[torch.randperm(n)].clamp(lo, hi)

        else:
            params[:, i] = torch.empty(n).uniform_(lo, hi)

    return params


def render_batch(
    synth: FMSynth,
    params: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Render audio for all parameter rows using vectorised forward_batch.

    Returns shape (N, clip_samples).
    """
    clip_samples = int(AUDIO_CLIP_SECONDS * synth.sample_rate)
    all_audio: list[torch.Tensor] = []
    n = params.shape[0]

    pbar = tqdm(total=n, desc="Rendering audio", unit="sample")
    for start in range(0, n, batch_size):
        chunk = params[start : start + batch_size]
        with torch.no_grad():
            audio = synth.forward_batch(chunk)
        audio = audio[:, :clip_samples]
        if audio.shape[1] < clip_samples:
            audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
        all_audio.append(audio.cpu())
        pbar.update(chunk.shape[0])
    pbar.close()

    return torch.cat(all_audio, dim=0)


def compute_target_specs(
    audio: torch.Tensor,
    fft_sizes: list[int] = None,
    hop_ratio: float = SPEC_HOP_RATIO,
    batch_size: int = 512,
) -> dict[int, torch.Tensor]:
    """Precompute log-magnitude STFTs for all target audio.

    Returns a dict mapping fft_size -> tensor of shape (N, n_freq, n_time).
    """
    if fft_sizes is None:
        fft_sizes = SPEC_FFT_SIZES

    specs: dict[int, list[torch.Tensor]] = {s: [] for s in fft_sizes}
    n = audio.shape[0]

    pbar = tqdm(total=n, desc="Computing spectrograms", unit="sample")
    for start in range(0, n, batch_size):
        chunk = audio[start : start + batch_size]
        for fft_size in fft_sizes:
            hop = int(fft_size * hop_ratio)
            window = torch.hann_window(fft_size, device=chunk.device)
            stft = torch.stft(
                chunk, n_fft=fft_size, hop_length=hop,
                window=window, return_complex=True,
            )
            eps_sq = 1e-14
            mag = (stft.real ** 2 + stft.imag ** 2 + eps_sq).sqrt()
            specs[fft_size].append(torch.log(mag))
        pbar.update(chunk.shape[0])
    pbar.close()

    return {s: torch.cat(v, dim=0) for s, v in specs.items()}


# ---------------------------------------------------------------------------
# Embedding extractors
# ---------------------------------------------------------------------------

def get_mel_embedding_batch(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Compute mel-spectrogram embeddings with mean+std temporal pooling.

    Fully batched PyTorch operations — no external model, runs on any device.

    Returns shape (N, n_mels * 2) = (N, 256) by default.
    """
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels,
        power=2.0,
    )

    n = audio_batch.shape[0]
    parts: list[torch.Tensor] = []

    pbar = tqdm(total=n, desc="Computing mel embeddings", unit="sample")
    for start in range(0, n, batch_size):
        chunk = audio_batch[start : start + batch_size]
        with torch.no_grad():
            mel = mel_transform(chunk)                    # (batch, n_mels, time)
            log_mel = torch.log(mel + 1e-8)
            mean = log_mel.mean(dim=-1)                   # (batch, n_mels)
            std = log_mel.std(dim=-1)                     # (batch, n_mels)
            emb = torch.cat([mean, std], dim=-1)          # (batch, n_mels*2)
        parts.append(emb)
        pbar.update(chunk.shape[0])
    pbar.close()

    return torch.cat(parts, dim=0)


def get_panns_embedding_batch(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract PANNs CNN14 embeddings (pre-trained on AudioSet).

    Returns shape (N, 2048).

    Requires ``pip install panns_inference``.
    Audio is resampled to 32 kHz internally (PANNs' expected rate).
    """
    try:
        from panns_inference import AudioTagging
    except ImportError:
        raise ImportError(
            "panns_inference is required for --embedding panns.\n"
            "Install it with:  pip install panns_inference"
        )
    import torchaudio

    target_sr = 32000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
    else:
        resampler = None

    at = AudioTagging(checkpoint_path=None, device="cpu")

    n = audio_batch.shape[0]
    parts: list[torch.Tensor] = []

    pbar = tqdm(total=n, desc="Extracting PANNs embeddings", unit="sample")
    for start in range(0, n, batch_size):
        chunk = audio_batch[start : start + batch_size]
        if resampler is not None:
            chunk = resampler(chunk)
        chunk_np = chunk.numpy().astype(np.float32)
        with torch.no_grad():
            _, embedding = at.inference(chunk_np)
        parts.append(torch.from_numpy(embedding).float())
        pbar.update(audio_batch[start : start + batch_size].shape[0])
    pbar.close()

    return torch.cat(parts, dim=0)


def _ensure_pkg_resources() -> None:
    """Shim pkg_resources for environments where setuptools is missing."""
    import sys
    if 'pkg_resources' not in sys.modules:
        try:
            import pkg_resources  # noqa: F401
        except ImportError:
            import importlib.util

            class PkgResources:
                @staticmethod
                def resource_filename(package_name, fname):
                    try:
                        spec = importlib.util.find_spec(package_name)
                        if spec and spec.origin:
                            import os
                            return os.path.join(os.path.dirname(spec.origin), fname)
                    except (ImportError, ValueError):
                        pass
                    return None

            sys.modules['pkg_resources'] = PkgResources()


def get_openl3_embedding_batch(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    batch_size: int = 128,
    start_from: int = 0,
    save_callback=None,
    save_interval: int = 500,
) -> torch.Tensor:
    """Extract OpenL3 embeddings for a batch of audio clips.

    Returns shape (N - start_from, 512) — only the newly-computed embeddings.

    Speedups over naive per-sample calls:
      - Model is loaded once and reused across all batches.
      - Each ``batch_size`` clips are passed as a list to openl3, which
        concatenates their frames and runs a single large TF inference pass
        instead of one pass per clip.

    Parameters
    ----------
    batch_size    : Number of clips per openl3 call (higher = faster, more RAM).
    start_from    : Skip the first N clips (already computed and saved in a checkpoint).
    save_callback : Called with the tensor of newly-computed embeddings so far,
                    roughly every ``save_interval`` samples.
    save_interval : How often (in samples) to invoke ``save_callback``.
    """
    _ensure_pkg_resources()
    import openl3

    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256", content_type="music", embedding_size=512,
    )

    total_n = audio_batch.shape[0]
    audio_to_process = audio_batch[start_from:]
    n_to_process = audio_to_process.shape[0]
    embeddings: list[torch.Tensor] = []
    last_saved = 0

    pbar = tqdm(total=total_n, initial=start_from, desc="Extracting OpenL3 embeddings", unit="sample")
    for start in range(0, n_to_process, batch_size):
        chunk = audio_to_process[start : start + batch_size]
        audio_list = [chunk[j].numpy().astype(np.float32) for j in range(chunk.shape[0])]
        sr_list = [sample_rate] * len(audio_list)

        emb_list, _ = openl3.get_audio_embedding(
            audio_list, sr_list, model=model,
            batch_size=len(audio_list) * 12,
            verbose=False,
        )

        for emb_np in emb_list:
            embeddings.append(torch.from_numpy(emb_np.mean(axis=0)).float())

        pbar.update(len(audio_list))

        if save_callback is not None and len(embeddings) - last_saved >= save_interval:
            save_callback(torch.stack(embeddings))
            last_saved = len(embeddings)
    pbar.close()

    return torch.stack(embeddings) if embeddings else torch.zeros(0, 512)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_samples: int = 20000,
    sample_rate: int = 44100,
    output_path: str = "data/synthetic_20k.pt",
    embedding_type: str = "mel",
    checkpoint_save_interval: int = 500,
) -> None:
    """Generate synthetic dataset with precomputed target spectrograms.

    Parameters
    ----------
    embedding_type : str
        One of ``"mel"``, ``"panns"``, or ``"openl3"``.

    For openl3 (which is slow), a ``<output_path>.ckpt`` sidecar is written
    after each major step so a crash can be resumed.  The sidecar is deleted
    once the final ``.pt`` is written successfully.
    """
    if embedding_type not in EMBEDDING_TYPES:
        raise ValueError(
            f"Unknown embedding type {embedding_type!r}. "
            f"Choose from {EMBEDDING_TYPES}"
        )

    ckpt_path = Path(output_path).with_suffix(".ckpt")

    print(f"\n{'=' * 60}")
    print(f"Generating {num_samples} synthetic training pairs")
    print(f"  Embedding type : {embedding_type}")
    print(f"  Audio clip     : {AUDIO_CLIP_SECONDS}s @ {sample_rate} Hz")
    if ckpt_path.exists():
        print(f"  Resuming from checkpoint: {ckpt_path}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    # Load existing checkpoint (if any)
    checkpoint: dict = {}
    if ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"  WARNING: could not load checkpoint ({e}), starting fresh.")
            checkpoint = {}

    def _save_ckpt() -> None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)

    # ------------------------------------------------------------------ #
    # 1) Sample parameters
    # ------------------------------------------------------------------ #
    if "params" in checkpoint:
        params = checkpoint["params"]
        print(f"Resumed params from checkpoint  ({params.shape})")
    else:
        print("Sampling parameters...")
        params = sample_parameters_batch(num_samples)
        print(f"  params shape: {params.shape}")
        checkpoint["params"] = params
        _save_ckpt()

    # ------------------------------------------------------------------ #
    # 2) Render audio
    # ------------------------------------------------------------------ #
    if "audio" in checkpoint:
        audio = checkpoint["audio"]
        print(f"Resumed audio from checkpoint  ({audio.shape})")
    else:
        synth = FMSynth(sample_rate=sample_rate)
        audio = render_batch(synth, params, batch_size=256)
        print(f"  audio shape: {audio.shape}  ({time.time() - t0:.1f}s elapsed)")
        checkpoint["audio"] = audio
        _save_ckpt()

    # ------------------------------------------------------------------ #
    # 3) Extract embeddings
    # ------------------------------------------------------------------ #
    if embedding_type == "mel":
        embeddings = get_mel_embedding_batch(audio, sample_rate)

    elif embedding_type == "panns":
        embeddings = get_panns_embedding_batch(audio, sample_rate)

    elif embedding_type == "openl3":
        existing_emb: torch.Tensor | None = checkpoint.get("embeddings", None)
        start_from = existing_emb.shape[0] if existing_emb is not None else 0

        if start_from >= num_samples:
            embeddings = existing_emb
            print(f"Resumed all {embeddings.shape[0]} embeddings from checkpoint.")
        else:
            if start_from > 0:
                print(f"Resuming OpenL3 from sample {start_from}/{num_samples}")

            def _save_embedding_progress(new_emb: torch.Tensor) -> None:
                combined = (
                    torch.cat([existing_emb, new_emb], dim=0)
                    if existing_emb is not None
                    else new_emb
                )
                checkpoint["embeddings"] = combined
                _save_ckpt()

            new_emb = get_openl3_embedding_batch(
                audio, sample_rate,
                start_from=start_from,
                save_callback=_save_embedding_progress,
                save_interval=checkpoint_save_interval,
            )
            embeddings = (
                torch.cat([existing_emb, new_emb], dim=0)
                if existing_emb is not None
                else new_emb
            )
            checkpoint["embeddings"] = embeddings
            _save_ckpt()

    print(
        f"  embeddings: {embeddings.shape}  "
        f"({embedding_type}, {time.time() - t0:.1f}s elapsed)"
    )

    # ------------------------------------------------------------------ #
    # 4) Write final dataset and remove checkpoint sidecar
    # ------------------------------------------------------------------ #
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "embedding_type": embedding_type,
        "embedding_dim": embeddings.shape[1],
        "params": params,
        "param_names": PARAM_NAMES,
        "sample_rate": sample_rate,
        "audio_clip_seconds": AUDIO_CLIP_SECONDS,
    }, output_path)

    if ckpt_path.exists():
        ckpt_path.unlink()

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\n{'=' * 60}")
    print(f"Saved to {output_path}  ({size_mb:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("--num_samples", type=int, default=20000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/synthetic_20k.pt", help="Output path")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate")
    parser.add_argument(
        "--embedding", type=str, default="mel", choices=EMBEDDING_TYPES,
        help="Embedding extractor: mel (256-d, instant), panns (2048-d, fast), "
             "openl3 (512-d, slow, needs TensorFlow)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500,
        help="Save a resumable checkpoint every N embeddings (openl3 only)",
    )
    args = parser.parse_args()

    generate_dataset(
        args.num_samples, args.sample_rate, args.output,
        embedding_type=args.embedding,
        checkpoint_save_interval=args.checkpoint_interval,
    )
