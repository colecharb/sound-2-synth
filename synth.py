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

    def _adsr_gated(
        self,
        t: torch.Tensor,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        gate_release_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a gated ADSR envelope.
        
        In this mode, the sustain phase continues indefinitely while gate is held,
        and release begins at gate_release_time when gate is released.
        
        Parameters
        ----------
        t : torch.Tensor
            Time vector.
        attack : torch.Tensor
            Attack time in seconds.
        decay : torch.Tensor
            Decay time in seconds.
        sustain : torch.Tensor
            Sustain level (0-1).
        release : torch.Tensor
            Release time in seconds.
        gate_release_time : torch.Tensor
            Time at which gate was released (enters release phase).
        """
        t_attack_end = attack
        t_decay_end = attack + decay

        # Attack ramp: 0 → 1
        attack_ramp = (t / attack).clamp(0.0, 1.0)

        # Decay ramp: 1 → sustain
        decay_progress = ((t - t_attack_end) / decay).clamp(0.0, 1.0)
        decay_ramp = 1.0 - (1.0 - sustain) * decay_progress

        # Sustain plateau
        sustain_level = sustain.expand_as(t)

        # Release ramp: sustain → 0
        release_progress = ((t - gate_release_time) / release).clamp(0.0, 1.0)
        release_ramp = sustain * (1.0 - release_progress)

        # Determine if gate was ever opened
        # If gate_release_time is very early (< 0), gate was never opened, so skip to release immediately
        gate_was_opened = (gate_release_time > -100).float()  # Threshold to detect "gate never opened"
        
        # Blend phases with smooth masking
        in_attack = (t <= t_attack_end).float() * gate_was_opened
        in_decay = ((t > t_attack_end) & (t <= t_decay_end)).float() * gate_was_opened
        in_sustain = ((t > t_decay_end) & (t < gate_release_time)).float() * gate_was_opened
        in_release = (t >= gate_release_time).float()
        
        # Renormalize so exactly one phase is active
        total = in_attack + in_decay + in_sustain + in_release
        total = total.clamp(min=1e-8)  # Avoid division by zero
        in_attack = in_attack / total
        in_decay = in_decay / total
        in_sustain = in_sustain / total
        in_release = in_release / total

        env = (
            in_attack * attack_ramp
            + in_decay * decay_ramp
            + in_sustain * sustain_level
            + in_release * release_ramp
        )
        return env

    def generate_chunk(
        self,
        duration: float,
        time_offset: float,
        carrier_freq: float | torch.Tensor = 440.0,
        mod_ratio: float | torch.Tensor = 2.0,
        mod_index: float | torch.Tensor = 5.0,
        attack: float | torch.Tensor = 0.01,
        decay: float | torch.Tensor = 0.1,
        sustain: float | torch.Tensor = 0.7,
        release: float | torch.Tensor = 0.3,
        gate_open: bool = True,
        gate_release_time: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate a short chunk of audio for streaming/real-time use.

        Parameters
        ----------
        duration : float
            Duration of chunk to generate in seconds.
        time_offset : float
            Current time in the note (from when gate was pressed).
        carrier_freq : float | torch.Tensor
            Carrier oscillator frequency in Hz.
        mod_ratio : float | torch.Tensor
            Modulator frequency ratio.
        mod_index : float | torch.Tensor
            Modulation depth.
        attack : float | torch.Tensor
            Attack time in seconds.
        decay : float | torch.Tensor
            Decay time in seconds.
        sustain : float | torch.Tensor
            Sustain level (0-1).
        release : float | torch.Tensor
            Release time in seconds.
        gate_open : bool
            Whether the gate is currently open (True) or closed (False).
        gate_release_time : float | torch.Tensor, optional
            Time when gate was released. If None and gate_open=False, uses current time_offset.

        Returns
        -------
        torch.Tensor
            Audio samples, shape (N,) normalized to approximately [-1, 1].
        """
        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(float(x))

        carrier_freq = _t(carrier_freq)
        mod_ratio = _t(mod_ratio)
        mod_index = _t(mod_index)
        attack = _t(attack).clamp(min=1e-4)
        decay = _t(decay).clamp(min=1e-4)
        sustain = _t(sustain).clamp(0.0, 1.0)
        release = _t(release).clamp(min=1e-4)
        time_offset_scalar = float(time_offset)

        sr = self.sample_rate
        n_samples = int(duration * sr)

        # Time vector for this chunk
        t = torch.linspace(
            time_offset_scalar,
            time_offset_scalar + duration,
            n_samples,
        )

        # Determine gate release time
        if gate_release_time is None:
            # If gate is open, set release time far in future (sustain indefinitely)
            # If gate is closed, set release time far in past (fully released, silent)
            gate_release_time = _t(1e6 if gate_open else -1e6)
        else:
            gate_release_time = _t(gate_release_time)

        # Build gated ADSR envelope
        env = self._adsr_gated(t, attack, decay, sustain, release, gate_release_time)

        # FM synthesis
        mod_freq = carrier_freq * mod_ratio
        modulator = mod_index * torch.sin(2.0 * torch.pi * mod_freq * t)
        carrier = torch.sin(2.0 * torch.pi * carrier_freq * t + modulator)

        audio = env * carrier

        # Soft peak normalization (avoid clicks from sudden changes)
        # Only normalize if gate is open; if closed, let it stay quiet
        if gate_open:
            peak = audio.abs().max()
            if peak > 1e-8:
                audio = audio / peak

        return audio
