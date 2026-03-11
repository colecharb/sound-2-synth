# FM Synth Parameter Order

This document defines the canonical ordering of synth parameters throughout the training pipeline.
All parameter vectors (predictions, targets, dataset entries) follow this order.

## Parameter Vector (10 dimensions)

```
Index  Parameter      Unit        Min      Max       Notes
-----  -----------    ----------  -------  --------  -----
0      carrier_freq   Hz          20.0     2000.0    Fundamental pitch
1      mod_ratio      ratio       0.5      4.0       Modulator freq = carrier * mod_ratio
2      mod_index      depth       0.0      20.0      FM modulation depth
3      attack         seconds     0.001    1.0       ADSR attack time
4      decay          seconds     0.001    2.0       ADSR decay time
5      sustain        level       0.0      1.0       ADSR sustain level
6      release        seconds     0.001    2.0       ADSR release time
7      note_duration  seconds     0.05     2.0       How long note is held at sustain
8      lowpass_freq   Hz          20.0     20000.0   Low-pass filter cutoff (log-scale)
9      highpass_freq  Hz          20.0     10000.0   High-pass filter cutoff (log-scale)
```

## Parameter Grouping

- **Oscillator** (0–2): carrier_freq, mod_ratio, mod_index
- **Envelope** (3–7): attack, decay, sustain, release, note_duration
- **Filters** (8–9): lowpass_freq, highpass_freq

## Usage Notes

- Log-scale parameters (lowpass_freq, highpass_freq): sample and scale using `log(x)` for perceptual uniformity
- All parameters are clamped to their valid ranges during prediction
- The MLP output layer produces 10 neurons matching this ordering
- OpenL3 embeddings are 512-dimensional and are input to the MLP
