# Audio-to-Synth Parameter Mapping

Extract FM synthesizer parameters from arbitrary audio using machine learning.

## Quick Start

### Option 1: Use the Live Synthesizer
```bash
./run_ui.sh
```
Play notes with your keyboard (S-L keys), adjust parameters with arrow keys.

**Controls:**
- **SPACE**: Gate (hold to play)
- **1-5**: Load presets
- **← →**: Adjust parameter (fine control)
- **↑ ↓**: Switch between parameters
- **S-L, E-I, D-K, Y-U**: Play notes on keyboard
- **Q**: Quit

### Option 2: Train the Parameter Predictor (Stage 1)

```bash
# 1. Generate 5K synthetic training examples
python3 dataset.py --num_samples 5000 --output data/synthetic_5k.pt

# 2. Train the MLP
python3 train.py --dataset data/synthetic_5k.pt --epochs 50 --output checkpoints/best_model.pt
```

**Time estimates:**
- Dataset generation: ~30 min (CPU), ~5 min (GPU)
- Training: ~2-3 hours (GPU), longer on CPU

## Documentation

All documentation is in the [`docs/`](docs/) directory:

- **[SUMMARY.md](docs/SUMMARY.md)** — Project overview and technical architecture
- **[UI_GUIDE.md](docs/UI_GUIDE.md)** — Live synthesizer controls and features
- **[TRAINING.md](docs/TRAINING.md)** — Stage 1 pre-training workflow
- **[PARAMETER_ORDER.md](docs/PARAMETER_ORDER.md)** — Parameter vector specification
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** — Original system design

## What It Does

**Phase 1: Real-time FM Synthesizer** ✓
- 10-parameter FM synthesis with full ADSR envelope
- High-pass and low-pass filters with log-scale perceptual control
- Live parameter adjustment with intuitive slider interface
- 5 built-in presets (default, bell, bass, brass, noise)
- WAV export

**Phase 2: Training Pipeline** ✓
- Generate synthetic training data from random parameters
- Extract OpenL3 embeddings (512-dimensional audio features)
- Train MLP to map embeddings → synth parameters
- Multi-scale spectral loss (perceptually-motivated)
- GPU-ready, fully differentiable

## Architecture

```
synth.py          → FM synthesizer engine with filters
synth_ui.py       → Real-time terminal UI for live play
dataset.py        → Generate synthetic training data
model.py          → Parameter predictor MLP (512→256→128→10)
loss.py           → Multi-scale spectral loss function
train.py          → Stage 1 pre-training loop
play.py           → CLI tool for rendering single notes
```

## Requirements

### Core
- Python 3.8+
- PyTorch 2.0+
- NumPy

### Audio
- soundfile
- sounddevice
- librosa

### ML (for training)
- openl3
- (TensorFlow via openl3, frozen inference only)

### UI
- textual
- pynput

Install all dependencies:
```bash
pip install -r requirements.txt
pip install openl3 librosa  # For training
```

## Key Features

### Perceptual Parameter Control
- **Log-scale filter frequencies**: 1.2x multiplier per step (consistent perceptually)
- **Smooth envelope transitions**: No clicks when releasing during attack/decay
- **Filter state continuity**: No artifacts at chunk boundaries in real-time playback

### Training System
- **10-parameter prediction**: carrier_freq, mod_ratio, mod_index, attack, decay, sustain, release, note_duration, lowpass_freq, highpass_freq
- **OpenL3 embeddings**: 512-dimensional pre-trained audio features
- **Spectral loss**: STFT at 3 scales (512, 1024, 2048 FFT) with log magnitude

### GPU Support
- Full CUDA support for both synthesis and training
- Automatic device detection (falls back to CPU)
- Batch processing for training efficiency

## Next Steps

After Stage 1 pre-training:
1. **Evaluate** on held-out synthetic test set
2. **Stage 2 Fine-tuning**: Train on diverse real audio (no ground-truth needed)
3. **Integration**: Use model in real-time inference or plugin
4. **A/B Testing**: Perceptual evaluation of generated vs. target audio

## Performance

- **Real-time synthesis**: ~sub-millisecond latency per chunk
- **Dataset generation**: ~5 min for 5K samples (GPU), ~30 min (CPU)
- **Training**: ~2-3 hours for 50 epochs (GPU with batch size 32)

## License

MIT

## Questions?

See [`docs/README.md`](docs/README.md) for a complete documentation index and links to all guides.
