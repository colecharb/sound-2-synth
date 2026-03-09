"""
synth.py — Differentiable FM Synthesizer Engine

Implements a single-operator FM synthesizer as a PyTorch module.
All operations are differentiable, making this suitable for gradient-based
parameter learning as described in PROJECT_OVERVIEW.md.

Signal model:
    mod(t)  = mod_index * sin(2π * mod_freq * t)
    out(t)  = env(t) * sin(2π * carrier_freq * t + mod(t))
    mod_freq = carrier_freq * mod_ratio

ADSR envelope phases:
    Attack  : 0 → 1 over `attack` seconds
    Decay   : 1 → sustain over `decay` seconds
    Sustain : held at `sustain` level for `note_duration` seconds
    Release : sustain → 0 over `release` seconds
"""

import torch
import torch.nn as nn


class FMSynth(nn.Module):
    """
    Single-operator FM synthesizer.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 44100).
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        carrier_freq: float | torch.Tensor = 440.0,
        mod_ratio: float | torch.Tensor = 2.0,
        mod_index: float | torch.Tensor = 5.0,
        attack: float | torch.Tensor = 0.01,
        decay: float | torch.Tensor = 0.1,
        sustain: float | torch.Tensor = 0.7,
        release: float | torch.Tensor = 0.3,
        note_duration: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """
        Render one note as a 1-D float32 tensor.

        Parameters
        ----------
        carrier_freq   : Carrier oscillator frequency in Hz.
        mod_ratio      : Modulator frequency = carrier_freq * mod_ratio.
        mod_index      : Modulation depth (FM index).
        attack         : Attack time in seconds.
        decay          : Decay time in seconds.
        sustain        : Sustain level (0–1).
        release        : Release time in seconds.
        note_duration  : Time in seconds the note is held at sustain level
                         (between end of decay and start of release).

        Returns
        -------
        torch.Tensor
            Shape (N,) float32 audio samples, normalised to [-1, 1].
        """
        # Convert scalars to tensors so all arithmetic is differentiable
        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(float(x))

        carrier_freq  = _t(carrier_freq)
        mod_ratio     = _t(mod_ratio)
        mod_index     = _t(mod_index)
        attack        = _t(attack).clamp(min=1e-4)
        decay         = _t(decay).clamp(min=1e-4)
        sustain       = _t(sustain).clamp(0.0, 1.0)
        release       = _t(release).clamp(min=1e-4)
        note_duration = _t(note_duration).clamp(min=0.0)

        sr = self.sample_rate
        total_duration = attack + decay + note_duration + release
        n_samples = int((total_duration * sr).item())

        # Time vector
        t = torch.linspace(0.0, total_duration.item(), n_samples)

        # --- ADSR envelope ---------------------------------------------------
        env = self._adsr(t, attack, decay, sustain, release, note_duration)

        # --- FM synthesis ----------------------------------------------------
        mod_freq = carrier_freq * mod_ratio
        modulator = mod_index * torch.sin(2.0 * torch.pi * mod_freq * t)
        carrier   = torch.sin(2.0 * torch.pi * carrier_freq * t + modulator)

        audio = env * carrier

        # Peak-normalise (avoid silence edge case)
        peak = audio.abs().max()
        if peak > 1e-8:
            audio = audio / peak

        return audio

    # ------------------------------------------------------------------
    def _adsr(
        self,
        t: torch.Tensor,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        note_duration: torch.Tensor,
    ) -> torch.Tensor:
        """Build an ADSR envelope over time vector `t`."""
        t_attack_end  = attack
        t_decay_end   = attack + decay
        t_sustain_end = attack + decay + note_duration

        # Attack ramp: 0 → 1
        attack_ramp = (t / attack).clamp(0.0, 1.0)

        # Decay ramp: 1 → sustain
        decay_progress = ((t - t_attack_end) / decay).clamp(0.0, 1.0)
        decay_ramp = 1.0 - (1.0 - sustain) * decay_progress

        # Sustain plateau
        sustain_level = sustain.expand_as(t)

        # Release ramp: sustain → 0
        release_progress = ((t - t_sustain_end) / release).clamp(0.0, 1.0)
        release_ramp = sustain * (1.0 - release_progress)

        # Blend phases with smooth masking
        in_attack  = (t <= t_attack_end).float()
        in_decay   = ((t > t_attack_end) & (t <= t_decay_end)).float()
        in_sustain = ((t > t_decay_end) & (t <= t_sustain_end)).float()
        in_release = (t > t_sustain_end).float()

        env = (
            in_attack  * attack_ramp
            + in_decay   * decay_ramp
            + in_sustain * sustain_level
            + in_release * release_ramp
        )
        return env
