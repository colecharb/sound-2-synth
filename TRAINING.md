# Stage 1 Pre-training Pipeline

This guide walks through generating synthetic training data and training the parameter predictor.

## Overview

**Goal**: Train an MLP to predict FM synth parameters from audio embeddings.

**Architecture**:
- Input: OpenL3 audio embedding (512-dim)
- Model: MLP (512 → 256 → 128 → 10)
- Output: FM synth parameters (10-dim)
- Loss: Multi-scale spectral loss comparing rendered audio

**Parameter Order** (see `PARAMETER_ORDER.md`):
```
[carrier_freq, mod_ratio, mod_index, attack, decay, sustain, release, 
 note_duration, lowpass_freq, highpass_freq]
```

## Setup

### 1. Install dependencies

```bash
pip install openl3 librosa
```

**Note**: openl3 uses TensorFlow internally (frozen model) but we run it in PyTorch-compatible mode for inference only.

### 2. Verify installation

```bash
python3 -c "import openl3, librosa, torch; print('✓ Ready')"
```

## Workflow

### Step 1: Generate Synthetic Dataset

Generate 5,000 random FM synth parameters, render audio, extract embeddings, and cache.

```bash
python3 dataset.py --num_samples 5000 --output data/synthetic_5k.pt
```

**What happens**:
1. Sample 5K random parameter sets within valid ranges
2. Render each using `FMSynth.forward()` (includes filters!)
3. Extract 512-dim OpenL3 embedding per audio clip
4. Save embeddings + parameters to `data/synthetic_5k.pt`

**Time estimate**: ~30 min on CPU, ~5 min on GPU

**Output**: `data/synthetic_5k.pt` (~1.5 GB)

### Step 2: Train the Parameter Predictor

```bash
python3 train.py \
    --dataset data/synthetic_5k.pt \
    --epochs 50 \
    --lr 1e-3 \
    --output checkpoints/best_model.pt
```

**What happens**:
1. Load embeddings + parameters from cache
2. Split into 80% train / 20% validation
3. Train MLP for 50 epochs
4. Save best checkpoint (lowest val loss)

**Key design choices**:
- **Batch rendering**: Each training iteration renders both target and predicted audio in-batch
- **Spectral loss**: Compares STFT magnitudes at 3 scales (512, 1024, 2048 FFT)
- **Log-scale params**: `lowpass_freq` and `highpass_freq` use log-space scaling for perceptual uniformity
- **Dropout**: 0.1 to prevent overfitting on synthetic data

**Time estimate**: ~2-3 hours on GPU with batch size 32

**Output**: `checkpoints/best_model.pt` (~2 MB)

### Step 3: Evaluate

After training, you can load the model and test it:

```python
import torch
from model import ParameterPredictor

model = ParameterPredictor()
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Test with a random embedding
embedding = torch.randn(512)
predicted_params = model(embedding)
print(f"Predicted parameters: {predicted_params}")
```

## Troubleshooting

### "openl3" import error

openl3 may require system-level dependencies. On macOS/Linux:

```bash
pip install --upgrade openl3
```

If issues persist, you can skip data generation and I'll provide a pre-computed `synthetic_5k.pt`.

### GPU vs CPU

- GPU is ~5-10x faster for data generation and training
- Model still trains on CPU (slow but works)
- Specify device in `train.py` (detects automatically)

### Out of memory

Reduce batch size in `train.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # was 32
```

## Next Steps

After Stage 1 training succeeds:

1. **Evaluate** on held-out synthetic test set
2. **Stage 2**: Fine-tune on diverse real audio (no ground-truth params needed)
3. **Integration**: Use trained model in a real-time plugin or CLI tool

## File Structure

```
data/
  synthetic_5k.pt           # Cached embeddings + params
checkpoints/
  best_model.pt             # Best checkpoint from training
dataset.py                  # Generate synthetic data
model.py                    # ParameterPredictor MLP
loss.py                     # MultiScaleSpectralLoss
train.py                    # Training loop
PARAMETER_ORDER.md          # Parameter vector specification
TRAINING.md                 # This file
```

## References

- **Parameter Order**: `PARAMETER_ORDER.md`
- **OpenL3**: https://openl3.readthedocs.io/
- **Project Overview**: `PROJECT_OVERVIEW.md`
