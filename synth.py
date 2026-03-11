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
import math


# Helper functions - not JIT due to tuple return complexity
# Instead, we'll inline these in the filter methods for better control
# and use state variables in the FMSynth class


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
        
        # IIR filter state for continuity across chunks
        self.lowpass_state = 0.0
        self.highpass_state = 0.0
        self.highpass_prev_sample = 0.0

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

    def _apply_lowpass_filter(self, audio: torch.Tensor, cutoff_freq: float) -> torch.Tensor:
        """
        Apply a first-order low-pass filter with state continuity.
        
        Parameters
        ----------
        audio : torch.Tensor
            Input audio samples.
        cutoff_freq : float
            Cutoff frequency in Hz.
            
        Returns
        -------
        torch.Tensor
            Filtered audio.
        """
        if cutoff_freq <= 0 or cutoff_freq >= self.sample_rate / 2:
            return audio
        
        wc = 2.0 * math.pi * cutoff_freq / self.sample_rate
        alpha = wc / (wc + 1.0)
        beta = 1.0 - alpha
        
        n = audio.shape[0]
        if n == 0:
            return audio
        
        # Simple loop-based IIR filter with state continuity
        # y[i] = alpha * x[i] + beta * y[i-1]
        filtered = torch.zeros_like(audio)
        state = self.lowpass_state
        
        for i in range(n):
            state = alpha * float(audio[i]) + beta * state
            filtered[i] = state
        
        # Save final state for next chunk
        self.lowpass_state = state
        
        return filtered
    
    def _apply_highpass_filter(self, audio: torch.Tensor, cutoff_freq: float) -> torch.Tensor:
        """
        Apply a first-order high-pass filter with state continuity.
        
        Parameters
        ----------
        audio : torch.Tensor
            Input audio samples.
        cutoff_freq : float
            Cutoff frequency in Hz.
            
        Returns
        -------
        torch.Tensor
            Filtered audio.
        """
        if cutoff_freq <= 0 or cutoff_freq >= self.sample_rate / 2:
            return audio
        
        # First-order high-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        wc = 2.0 * math.pi * cutoff_freq / self.sample_rate
        alpha = 1.0 / (wc + 1.0)
        
        n = audio.shape[0]
        if n == 0:
            return audio
        
        # Simple loop-based IIR filter with state continuity
        filtered = torch.zeros_like(audio)
        state = self.highpass_state
        last_sample = self.highpass_prev_sample
        
        for i in range(n):
            current_sample = float(audio[i])
            state = alpha * (state + current_sample - last_sample)
            filtered[i] = state
            last_sample = current_sample
        
        # Save state for next chunk
        self.highpass_state = state
        self.highpass_prev_sample = last_sample
        
        return filtered

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
        lowpass_freq: float = 20000.0,
        highpass_freq: float = 20.0,
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
        lowpass_freq : float
            Low-pass filter cutoff frequency in Hz (default 20000 = no filtering).
        highpass_freq : float
            High-pass filter cutoff frequency in Hz (default 20 = minimal filtering).

        Returns
        -------
        torch.Tensor
            Audio samples, shape (N,) normalized to approximately [-1, 1].
        """
        sr = self.sample_rate
        n_samples = int(duration * sr)
        
        # Fast path for scalars - avoid tensor conversion overhead
        cf = float(carrier_freq)
        mr = float(mod_ratio)
        mi = float(mod_index)
        att = max(float(attack), 1e-4)
        dec = max(float(decay), 1e-4)
        sus = min(max(float(sustain), 0.0), 1.0)
        rel = max(float(release), 1e-4)
        time_offset_scalar = float(time_offset)

        # Arange instead of linspace for better performance
        # t = time_offset + (arange / sr)
        t = torch.arange(n_samples, dtype=torch.float32) / sr + time_offset_scalar

        # Gate release time as scalar
        gate_rel_time = 1e6 if gate_open else -1e6
        if gate_release_time is not None:
            gate_rel_time = float(gate_release_time)

        # Build gated ADSR envelope (inlined for speed)
        t_attack_end = att
        t_decay_end = att + dec
        
        # Vectorized ADSR computation
        attack_ramp = (t / att).clamp(0.0, 1.0)
        decay_progress = ((t - t_attack_end) / dec).clamp(0.0, 1.0)
        decay_ramp = 1.0 - (1.0 - sus) * decay_progress
        
        release_progress = ((t - gate_rel_time) / rel).clamp(0.0, 1.0)
        release_ramp = sus * (1.0 - release_progress)
        
        in_attack  = (t <= t_attack_end).float()
        in_decay   = ((t > t_attack_end) & (t <= t_decay_end)).float()
        in_sustain = (t > t_decay_end) & (t <= gate_rel_time)
        in_release = (t > gate_rel_time).float()
        
        env = (
            in_attack  * attack_ramp
            + in_decay * decay_ramp
            + in_sustain.float() * sus
            + in_release * release_ramp
        )

        # FM synthesis - all scalar operations on vectorized time
        two_pi = 2.0 * math.pi
        mod_freq = cf * mr
        modulator = mi * torch.sin(two_pi * mod_freq * t)
        carrier = torch.sin(two_pi * cf * t + modulator)

        audio = env * carrier

        # Apply filters - skip if frequencies are neutral (common case)
        if highpass_freq > 1.0:  # Anything above 1 Hz is meaningful
            audio = self._apply_highpass_filter(audio, highpass_freq)
        if lowpass_freq < 20000.0:  # Anything below 20k is filtering
            audio = self._apply_lowpass_filter(audio, lowpass_freq)

        # Soft peak normalization (avoid clicks from sudden changes)
        # Only normalize if gate is open; if closed, let it stay quiet
        if gate_open:
            peak = audio.abs().max()
            if peak > 1e-8:
                audio = audio / peak

        return audio
