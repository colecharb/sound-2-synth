# Audio-to-Synth Parameter Mapping Project

## Overview
Build a system that takes arbitrary audio input, extracts timbral features via OpenL3, and predicts FM synthesizer parameters that reproduce similar sound. The entire pipeline is fully differentiable.

## Architecture

### Synthesizer (torchsynth FM)
**Parameters to predict (8 total):**
- Carrier frequency (Hz)
- Modulator frequency (ratio to carrier, 0.5–4x range)
- Modulation index (FM depth, 0–20 range)
- ADSR envelope: attack, decay, sustain, release (all in seconds)

Render audio from these parameters using torchsynth's FM oscillator module.

### Embedding Model (OpenL3 - frozen)
Use pre-trained OpenL3 embedding model to extract 512-dimensional embeddings from audio. Aggregate across time (average frames) to get single embedding vector per audio clip.

### Parameter Predictor (MLP - trainable)
Small neural network: 512 (input) → 256 → 128 → 8 (output)
- Input: OpenL3 embedding (512,)
- Output: synth parameters (8,)
- Clamp output to valid parameter ranges using sigmoid

### Loss Function
Multi-scale spectral loss comparing STFT magnitudes at 3 scales (512, 1024, 2048 FFT sizes). Use log magnitude (perceptually closer to human hearing) and L1 distance.

## Training Pipeline

**Stage 1: Pre-training on Synthetic Data**
1. Generate 10K–50K random parameter sets
2. Render audio from each using torchsynth
3. Compute OpenL3 embeddings
4. Train MLP with direct parameter MSE or spectral loss
5. Run for a few epochs until validation plateaus

**Stage 2: Fine-tuning on Diverse Audio**
1. Collect arbitrary audio clips (1–3 seconds, diverse sources)
2. Compute embeddings (no ground-truth parameters)
3. Train MLP using spectral loss between input and rendered audio
4. Lower learning rate (1e-4), train until convergence

## Tech Stack
- PyTorch
- librosa (audio I/O)
- openl3 (embeddings)
- torchsynth (FM synthesis)
- soundfile (audio export)

## Key Implementation Notes
- All operations must be PyTorch tensors (differentiable)
- Cache pre-computed embeddings to avoid recomputing
- Start with small dataset (5K–10K synthetic pairs) to validate pipeline
- Use GPU if available; CPU is feasible with patience
- Spectral loss should normalize for audio length differences

## Success Criteria
- Pre-training: MLP recovers parameters reasonably well on held-out synthetic test set
- Fine-tuning: Rendered audio has similar spectral character to input (subjective listening test)
