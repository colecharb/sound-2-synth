# Documentation

This directory contains all project documentation for the Audio-to-Synth Parameter Mapping system.

## Quick Start

- **New to the project?** Start with [SUMMARY.md](SUMMARY.md)
- **Want to use the synthesizer?** See [UI_GUIDE.md](UI_GUIDE.md)
- **Training the parameter predictor?** Check [TRAINING.md](TRAINING.md)
- **Understanding the architecture?** Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

## Documentation Files

### [SUMMARY.md](SUMMARY.md)
High-level overview of the entire project including:
- What has been built (Phase 1 & 2)
- Key features and technical highlights
- File structure and technology stack
- Getting started guide

### [UI_GUIDE.md](UI_GUIDE.md)
Complete guide for the real-time FM synthesizer UI:
- Controls and keyboard shortcuts
- Parameter ranges and step sizes
- Preset configurations
- Tips for sound design

### [TRAINING.md](TRAINING.md)
Step-by-step guide for Stage 1 pre-training:
- Setup and dependency installation
- Dataset generation (synthetic data)
- Model training procedure
- Troubleshooting and hardware recommendations
- Time estimates and next steps

### [PARAMETER_ORDER.md](PARAMETER_ORDER.md)
Technical specification for the parameter vector:
- 10-dimensional parameter ordering (canonical)
- Parameter ranges and groupings
- Log-scale vs linear-scale parameters
- Usage notes for the training pipeline

### [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
Original project specification:
- System architecture and components
- Training pipeline design (Stage 1 & 2)
- Tech stack and dependencies
- Success criteria

## Architecture

```
Phase 1: Live Synthesizer UI
├── Real-time FM synthesis
├── ADSR envelope with filters
├── Perceptual parameter control
└── WAV export

Phase 2: Training Pipeline
├── Synthetic data generation
├── OpenL3 embedding extraction
├── Parameter predictor MLP
└── Multi-scale spectral loss
```

## Running the Project

### Start the Synthesizer
```bash
./run_ui.sh
```

### Generate Training Data
```bash
python3 dataset.py --num_samples 5000 --output data/synthetic_5k.pt
```

### Train the Model
```bash
python3 train.py --dataset data/synthetic_5k.pt --epochs 50 --output checkpoints/best_model.pt
```

## Key Concepts

### Parameter Vector (10 dimensions)
See [PARAMETER_ORDER.md](PARAMETER_ORDER.md) for the canonical ordering:
```
[carrier_freq, mod_ratio, mod_index, attack, decay, sustain, release, 
 note_duration, lowpass_freq, highpass_freq]
```

### Multi-scale Spectral Loss
Compares STFT magnitudes at 3 scales (512, 1024, 2048 FFT) using log magnitude and L1 distance, providing perceptually-motivated loss.

### Log-Scale Parameters
Filter frequencies use logarithmic scaling for perceptual uniformity across the frequency range, matching how human hearing works.

## Next Steps

After Stage 1 pre-training succeeds:
1. Evaluate on held-out synthetic test set
2. Stage 2: Fine-tune on diverse real audio
3. Integrate into real-time inference system
4. Perceptual evaluation and A/B testing

## Questions?

Refer to the specific guides above or check the inline documentation in the source files.
