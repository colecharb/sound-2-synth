"""
play.py — FM Synth CLI test script

Renders a single FM note with the given parameters, saves it as a WAV file,
and plays it back through system audio.

Usage
-----
    python play.py [options]

Examples
--------
    # Default note (A4, mod_ratio=2, mod_index=5)
    python play.py

    # Bell-like tone
    python play.py --carrier 880 --mod_ratio 3.5 --mod_index 12 \
                   --attack 0.005 --decay 0.4 --sustain 0.0 --release 0.5

    # Warm bass
    python play.py --carrier 110 --mod_ratio 1.0 --mod_index 2 \
                   --attack 0.05 --decay 0.2 --sustain 0.8 --release 0.4 \
                   --note_duration 1.5

    # Save to a specific file without playing
    python play.py --out my_sound.wav --no_play
"""

import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from synth import FMSynth

# ---------------------------------------------------------------------------
# Preset patches — quick starting points for exploration
# ---------------------------------------------------------------------------
PRESETS = {
    "default": dict(
        carrier_freq=220.0,
        mod_ratio=2.0,
        mod_index=5.0,
        attack=0.01,
        decay=0.1,
        sustain=0.5,
        release=1.0,
        note_duration=0.25,
    ),
    "bell": dict(
        carrier_freq=880.0,
        mod_ratio=3.5,
        mod_index=12.0,
        attack=0.005,
        decay=0.4,
        sustain=0.0,
        release=0.5,
        note_duration=0.0,
    ),
    "bass": dict(
        carrier_freq=110.0,
        mod_ratio=1.0,
        mod_index=2.0,
        attack=0.05,
        decay=0.2,
        sustain=0.8,
        release=0.4,
        note_duration=1.5,
    ),
    "brass": dict(
        carrier_freq=440.0,
        mod_ratio=1.0,
        mod_index=8.0,
        attack=0.08,
        decay=0.15,
        sustain=0.9,
        release=0.25,
        note_duration=0.8,
    ),
    "noise": dict(
        carrier_freq=200.0,
        mod_ratio=4.0,
        mod_index=20.0,
        attack=0.001,
        decay=0.05,
        sustain=0.3,
        release=0.1,
        note_duration=0.5,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FM Synth — render and play a single note.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Preset shortcut
    p.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Load a preset patch (individual flags override preset values).",
    )

    # Synth parameters
    p.add_argument(
        "--carrier",
        type=float,
        default=None,
        metavar="Hz",
        help="Carrier oscillator frequency in Hz.",
    )
    p.add_argument(
        "--mod_ratio",
        type=float,
        default=None,
        help="Modulator freq = carrier × mod_ratio (0.5–4 range).",
    )
    p.add_argument(
        "--mod_index",
        type=float,
        default=None,
        help="FM modulation depth / index (0–20).",
    )
    p.add_argument(
        "--attack",
        type=float,
        default=None,
        metavar="s",
        help="ADSR attack time in seconds.",
    )
    p.add_argument(
        "--decay",
        type=float,
        default=None,
        metavar="s",
        help="ADSR decay time in seconds.",
    )
    p.add_argument(
        "--sustain", type=float, default=None, help="ADSR sustain level (0–1)."
    )
    p.add_argument(
        "--release",
        type=float,
        default=None,
        metavar="s",
        help="ADSR release time in seconds.",
    )
    p.add_argument(
        "--note_duration",
        type=float,
        default=None,
        metavar="s",
        help="How long the note is held at sustain before release.",
    )

    # Output
    p.add_argument(
        "--out", type=str, default="output.wav", help="Output WAV file path."
    )
    p.add_argument(
        "--no_play", action="store_true", help="Save the file but do not play it."
    )
    p.add_argument(
        "--sample_rate", type=int, default=44100, help="Audio sample rate in Hz."
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Start from preset or default
    base = PRESETS[args.preset].copy() if args.preset else PRESETS["default"].copy()

    # Override with any explicitly provided flags
    overrides = {
        "carrier_freq": args.carrier,
        "mod_ratio": args.mod_ratio,
        "mod_index": args.mod_index,
        "attack": args.attack,
        "decay": args.decay,
        "sustain": args.sustain,
        "release": args.release,
        "note_duration": args.note_duration,
    }
    params = {k: (v if v is not None else base[k]) for k, v in overrides.items()}

    # Print what we're rendering
    total = (
        params["attack"] + params["decay"] + params["note_duration"] + params["release"]
    )
    print("\nFM Synth Parameters")
    print("─" * 38)
    print(f"  carrier_freq   : {params['carrier_freq']:.2f} Hz")
    print(
        f"  mod_ratio      : {params['mod_ratio']:.3f}  →  mod_freq = {params['carrier_freq'] * params['mod_ratio']:.2f} Hz"
    )
    print(f"  mod_index      : {params['mod_index']:.3f}")
    print(f"  attack         : {params['attack']:.4f} s")
    print(f"  decay          : {params['decay']:.4f} s")
    print(f"  sustain        : {params['sustain']:.3f}")
    print(f"  release        : {params['release']:.4f} s")
    print(f"  note_duration  : {params['note_duration']:.3f} s")
    print(
        f"  total length   : {total:.3f} s  ({int(total * args.sample_rate)} samples)"
    )
    print("─" * 38)

    # Render
    synth = FMSynth(sample_rate=args.sample_rate)
    with torch.no_grad():
        audio: torch.Tensor = synth(**params)

    audio_np = audio.numpy().astype(np.float32)

    # Save WAV
    sf.write(args.out, audio_np, args.sample_rate, subtype="PCM_24")
    print(f"\nSaved  →  {args.out}")

    # Play
    if not args.no_play:
        print("Playing... (press Ctrl+C to stop early)")
        try:
            sd.play(audio_np, samplerate=args.sample_rate)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("\nStopped.")
        print("Done.")


if __name__ == "__main__":
    main()
