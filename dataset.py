"""
dataset.py — Generate synthetic training data for Stage 1 pre-training

Generates random FM synth parameters, renders audio using FMSynth,
extracts OpenL3 embeddings, and precomputes target spectrograms for
fast spectral-loss training.

The dataset .pt file contains:
  - embeddings  (N, 512)   OpenL3 audio embeddings
  - params      (N, 10)    ground-truth synth parameters
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
) -> list[torch.Tensor]:
    """Render audio for all parameter rows using vectorised forward_batch.

    Returns a list of 1-D audio tensors (variable length, but clipped to
    AUDIO_CLIP_SECONDS for the spectrogram cache).
    """
    clip_samples = int(AUDIO_CLIP_SECONDS * synth.sample_rate)
    all_audio: list[torch.Tensor] = []
    n = params.shape[0]

    for start in range(0, n, batch_size):
        chunk = params[start : start + batch_size]
        with torch.no_grad():
            audio = synth.forward_batch(chunk)
        # Clip to fixed length
        audio = audio[:, :clip_samples]
        # Pad short clips
        if audio.shape[1] < clip_samples:
            audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
        all_audio.append(audio.cpu())

    return torch.cat(all_audio, dim=0)  # (N, clip_samples)


def compute_target_specs(
    audio: torch.Tensor,
    fft_sizes: list[int] = None,
    hop_ratio: float = SPEC_HOP_RATIO,
    batch_size: int = 512,
) -> dict[int, torch.Tensor]:
    """Precompute log-magnitude STFTs for all target audio.

    Returns a dict mapping fft_size → tensor of shape (N, n_freq, n_time).
    These are stored in the dataset so Stage B never has to re-render or
    re-STFT the target audio.
    """
    if fft_sizes is None:
        fft_sizes = SPEC_FFT_SIZES

    specs: dict[int, list[torch.Tensor]] = {s: [] for s in fft_sizes}
    n = audio.shape[0]

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

    return {s: torch.cat(v, dim=0) for s, v in specs.items()}


def get_openl3_embedding_batch(
    audio_batch: torch.Tensor,
    sample_rate: int = 44100,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract OpenL3 embeddings for a batch of audio clips.

    Returns shape (N, 512).
    """
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

    import openl3

    n = audio_batch.shape[0]
    embeddings = []

    for start in range(0, n, batch_size):
        chunk = audio_batch[start : start + batch_size]
        for j in range(chunk.shape[0]):
            audio_np = chunk[j].numpy().astype(np.float32)
            emb_np, _ = openl3.get_audio_embedding(
                audio_np, sr=sample_rate,
                input_repr="mel256", content_type="music",
                embedding_size=512, center=True,
            )
            embeddings.append(torch.from_numpy(emb_np.mean(axis=0)).float())

        done = min(start + batch_size, n)
        print(f"  Embeddings: {done}/{n}", end="\r", flush=True)

    print()
    return torch.stack(embeddings)


def generate_dataset(
    num_samples: int = 20000,
    sample_rate: int = 44100,
    output_path: str = "data/synthetic_20k.pt",
) -> None:
    """Generate synthetic dataset with precomputed target spectrograms."""
    print(f"\n{'=' * 60}")
    print(f"Generating {num_samples} synthetic training pairs")
    print(f"  Audio clip length: {AUDIO_CLIP_SECONDS}s")
    print(f"  Precomputing target STFTs at sizes: {SPEC_FFT_SIZES}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    # 1) Sample parameters (instant)
    print("Sampling parameters...")
    params = sample_parameters_batch(num_samples)
    print(f"  params shape: {params.shape}")

    # 2) Render audio in vectorised batches
    print("Rendering audio...")
    synth = FMSynth(sample_rate=sample_rate)
    audio = render_batch(synth, params, batch_size=256)
    print(f"  audio shape: {audio.shape}  ({time.time() - t0:.1f}s elapsed)")

    # 3) Extract OpenL3 embeddings
    print("Extracting OpenL3 embeddings (this is the slow part)...")
    embeddings = get_openl3_embedding_batch(audio, sample_rate)
    print(f"  embeddings shape: {embeddings.shape}  ({time.time() - t0:.1f}s elapsed)")

    # 4) Precompute target spectrograms
    print("Precomputing target spectrograms...")
    target_specs = compute_target_specs(audio)
    for fft_size, spec in target_specs.items():
        print(f"  FFT {fft_size}: {spec.shape}")
    print(f"  ({time.time() - t0:.1f}s elapsed)")

    # 5) Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "params": params,
        "param_names": PARAM_NAMES,
        "target_specs": target_specs,
        "audio_clip_seconds": AUDIO_CLIP_SECONDS,
        "spec_fft_sizes": SPEC_FFT_SIZES,
        "spec_hop_ratio": SPEC_HOP_RATIO,
    }, output_path)

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
    args = parser.parse_args()

    generate_dataset(args.num_samples, args.sample_rate, args.output)
