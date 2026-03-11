"""
dataset.py — Generate synthetic training data for Stage 1 pre-training

Generates random FM synth parameters, renders audio using FMSynth.forward(),
extracts OpenL3 embeddings, and caches the results as a PyTorch dataset.

Parameter order (see PARAMETER_ORDER.md):
[carrier_freq, mod_ratio, mod_index, attack, decay, sustain, release, 
 note_duration, lowpass_freq, highpass_freq]
"""

import math
import torch
import numpy as np
from pathlib import Path

from synth import FMSynth

# Parameter ranges (from synth_ui.py PARAM_RANGES)
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

# Parameter order for the dataset
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


def sample_parameters() -> dict:
    """Sample random parameters within valid ranges.
    
    Log-scale parameters (lowpass_freq, highpass_freq) are sampled log-uniformly.
    """
    params = {}
    
    # Linear parameters
    for param in ["carrier_freq", "mod_ratio", "mod_index", "attack", "decay", "sustain", "release", "note_duration"]:
        min_val, max_val = PARAM_RANGES[param]
        params[param] = float(np.random.uniform(min_val, max_val))
    
    # Log-scale parameters
    for param in ["lowpass_freq", "highpass_freq"]:
        min_val, max_val = PARAM_RANGES[param]
        log_min, log_max = math.log(min_val), math.log(max_val)
        params[param] = float(math.exp(np.random.uniform(log_min, log_max)))
    
    return params


def render_audio(synth: FMSynth, params: dict, sample_rate: int = 44100) -> torch.Tensor:
    """Render audio from parameters using FMSynth.forward()."""
    with torch.no_grad():
        audio = synth.forward(
            carrier_freq=params["carrier_freq"],
            mod_ratio=params["mod_ratio"],
            mod_index=params["mod_index"],
            attack=params["attack"],
            decay=params["decay"],
            sustain=params["sustain"],
            release=params["release"],
            note_duration=params["note_duration"],
            lowpass_freq=params["lowpass_freq"],
            highpass_freq=params["highpass_freq"],
        )
    return audio


def get_openl3_embedding(audio: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
    """Extract OpenL3 embedding from audio.
    
    Audio should be a 1D torch tensor on CPU. Returns a 512-dimensional embedding.
    """
    try:
        # Workaround for pkg_resources import issue in resampy on Python 3.11+
        import sys
        if 'pkg_resources' not in sys.modules:
            try:
                import pkg_resources  # noqa: F401
            except ImportError:
                # Provide a minimal pkg_resources wrapper for resampy
                import importlib.resources as resources
                import importlib.util
                
                class PkgResources:
                    """Minimal wrapper to provide resource_filename for resampy."""
                    @staticmethod
                    def resource_filename(package_name, fname):
                        # For resampy, locate filter files in its data directory
                        try:
                            spec = importlib.util.find_spec(package_name)
                            if spec and spec.origin:
                                import os
                                pkg_dir = os.path.dirname(spec.origin)
                                return os.path.join(pkg_dir, fname)
                        except (ImportError, ValueError):
                            pass
                        return None
                
                sys.modules['pkg_resources'] = PkgResources()
        
        import openl3
    except ImportError as e:
        raise ImportError(
            "openl3 is required. Install it with:\n"
            "  pip install openl3 librosa\n"
            "  Note: openl3 may require additional system dependencies (libsndfile)\n"
            f"Import error: {e}"
        )
    
    # Convert tensor to numpy
    audio_np = audio.cpu().numpy().astype(np.float32)
    
    # Load OpenL3 model (TensorFlow/Keras, used in inference-only mode)
    # embedding_size: 512, input_repr: mel256, content_type: music
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=512
    )
    
    # Extract embedding using OpenL3's embedding function
    # This handles the audio preprocessing and model inference
    # Returns tuple of (embeddings, timestamps)
    embeddings_np, _ = openl3.get_audio_embedding(
        audio_np,
        sr=sample_rate,
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        center=True
    )
    
    # embeddings_np shape: (num_frames, 512)
    # Average across frames to get single 512-D embedding, convert to torch
    embedding = torch.from_numpy(embeddings_np.mean(axis=0)).float()  # (512,)
    
    return embedding


def generate_dataset(
    num_samples: int = 5000,
    sample_rate: int = 44100,
    output_path: str = "data/synthetic_5k.pt",
) -> None:
    """Generate synthetic dataset and save to disk.
    
    Parameters
    ----------
    num_samples : int
        Number of (audio, params) pairs to generate
    sample_rate : int
        Sample rate for audio rendering
    output_path : str
        Where to save the dataset
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} synthetic training pairs")
    print(f"{'='*60}\n")
    
    synth = FMSynth(sample_rate=sample_rate)
    
    embeddings_list = []
    params_list = []
    
    for i in range(num_samples):
        # Sample parameters
        params = sample_parameters()
        
        # Render audio
        audio = render_audio(synth, params, sample_rate)
        
        # Extract embedding
        embedding = get_openl3_embedding(audio, sample_rate)
        
        embeddings_list.append(embedding)
        
        # Store params in canonical order
        param_vector = torch.tensor([params[name] for name in PARAM_NAMES], dtype=torch.float32)
        params_list.append(param_vector)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    # Stack into tensors
    embeddings = torch.stack(embeddings_list)  # (num_samples, 512)
    params = torch.stack(params_list)          # (num_samples, 10)
    
    print(f"\nDataset shapes:")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  params: {params.shape}")
    
    # Save to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "params": params,
        "param_names": PARAM_NAMES,
    }, output_path)
    
    print(f"\n✓ Saved to {output_path}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/synthetic_5k.pt", help="Output path")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate")
    args = parser.parse_args()
    
    generate_dataset(args.num_samples, args.sample_rate, args.output)
