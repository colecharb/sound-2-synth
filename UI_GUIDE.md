# FM Synth Live Control UI

A real-time terminal UI for tweaking FM synthesizer parameters and playing live with minimal latency.

## Quick Start

```bash
# Activate virtual environment and run
source .venv/bin/activate
python3 -m synth_ui

# Or use the convenience script
./run_ui.sh
```

## Interface Overview

The UI is divided into three sections:

- **Left Panel**: 7 parameter sliders with real-time visual feedback
- **Right Panel**: Preset buttons, current values, and gate status
- **Header/Footer**: Title and keyboard shortcut reference

## Controls

### Initial State
- **App starts with gate CLOSED** (silent)
- No sound until you press Play or click the PLAY button
- All sliders respond to keyboard immediately when in focus

### Gate Control (Playback)
- **SPACE Key (Hold to Play)**:
  - **Press and hold SPACE**: Opens gate, synth enters attack/decay/sustain
  - **Release SPACE**: Closes gate, synth enters release phase and fades out
  - Works like a keyboard sustain pedal or trigger
  - 200ms timeout ensures release is detected even if key repeats are slow
  
- **▶ PLAY Button**: Alternative to spacebar, click to open gate
  - Synth enters Attack → Decay → Sustain phases
  - Visual indicator shows "PLAYING"
  
- **■ STOP Button**: Click to close gate and enter release phase
  - Synth fades out over the release time
  - Visual indicator shows "IDLE"

### Parameter Adjustment
- **↑ / ↓ (UP/DOWN arrows)**: Navigate between sliders (shows "◄ FOCUSED" indicator)
- **← / → (LEFT/RIGHT arrows)**: Adjust currently focused slider value
  - LEFT decreases by 5%
  - RIGHT increases by 5%
- Parameters update in real-time and immediately affect the synthesizer
- Slider shows visual bar and numeric value that updates live

### Buttons (Click Only)
- **Preset buttons (1-5)**: Click to load preset or press 1-5 key
- **Play/Stop buttons**: Click to control gate or use SPACE key
- Buttons cannot be focused via keyboard (sliders are keyboard-navigable only)

### Presets (5 built-in)
- **1**: Default - balanced FM sine tone
- **2**: Bell - bright, decaying bell-like timbre
- **3**: Bass - deep, warm bass tone
- **4**: Brass - sustained brass-like character
- **5**: Noise - high-modulation noise-like texture
- Click buttons or press number keys to load

### Exit
- **Q**: Quit the application

## Parameters

All 7 FM synth parameters are controllable:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **carrier_freq** | 20 - 2000 Hz | Main oscillator frequency (pitch) |
| **mod_ratio** | 0.5 - 4.0x | Modulator frequency as ratio of carrier |
| **mod_index** | 0 - 20 | FM depth/modulation intensity |
| **attack** | 0.001 - 1.0 s | Time to reach peak from silence |
| **decay** | 0.001 - 2.0 s | Time to reach sustain level |
| **sustain** | 0 - 1.0 | Level held during sustain phase (0-1) |
| **release** | 0.001 - 2.0 s | Time from sustain to silence when gate closes |

## Understanding the Envelope

The synth uses an ADSR (Attack, Decay, Sustain, Release) envelope:

```
      Peak
        |    \
        |     \___
    Sustain    |  \
        |      |   \___
        |______|_______|________
    Attack Decay Sustain Release
       ↑
    (gate opens)              ↑
                          (gate closes)
```

When you hold SPACE:
1. **Attack**: Quickly rises from 0 to 1
2. **Decay**: Falls from 1 to sustain level
3. **Sustain**: Held at specified level indefinitely (while SPACE held)

When you release SPACE:
4. **Release**: Falls from sustain level to 0

## Audio Streaming

- Uses low-latency audio streaming (~46ms buffer)
- Real-time parameter changes are smoothly interpolated
- Callback-based playback for minimal delay

## Tips for Exploration

1. **Start with Presets**: Load a preset and hold SPACE to hear the basic sound
2. **Adjust One Parameter**: Focus on carrier_freq to change pitch, then try mod_ratio for timbral changes
3. **Use Extremes**: Try min/max values to understand parameter effects
4. **Gate Practice**: 
   - Hold SPACE to hear attack/decay/sustain
   - Quickly release to hear the release envelope
   - Try tapping vs. holding for different effects
5. **Sustain Level**: Set to 0 for percussive sounds, 1 for sustained drones
6. **Spacebar Control**: Behaves like a keyboard sustain pedal - hold to play, release to stop

## Troubleshooting

**No Audio?**
- Check your system audio output is enabled
- Verify your default audio device in system settings
- Try a different preset

**Gate Toggles Unexpectedly?**
- This is the toggle behavior - SPACE toggles gate on/off
- For continuous control, hold SPACE and adjust parameters

**UI Not Rendering?**
- Ensure terminal width ≥ 100 characters
- Try resizing terminal window
- Textual requires modern terminal (iTerm2, Windows Terminal, etc.)

**Parameters Not Changing Sound?**
- Check that gate is ON (IDLE vs PLAYING indicator)
- Verify sustain level is not 0 (no sound will play)
- Try loading a preset to reset to known good state

## Technical Details

### Architecture
- **synth.py**: Core FM synthesizer with differentiable PyTorch operations
- **synth_ui.py**: Textual TUI app with audio streaming manager
- **AudioManager**: Handles real-time audio generation via `sounddevice.OutputStream`
- Streaming uses 2048-sample chunks for ~46ms latency at 44.1 kHz

### Audio Callback
Each audio chunk is generated on-demand by the audio callback thread, ensuring:
- Minimal buffering and latency
- Responsive parameter changes
- Smooth gate control

## Files

- `synth.py`: FM synthesizer engine (now with `generate_chunk()` for streaming)
- `synth_ui.py`: Textual UI application
- `run_ui.sh`: Convenience launch script
- `requirements.txt`: Dependencies (now includes `textual`)

## Keyboard Shortcut Reference

```
Slider Navigation  ↑ / ↓ (UP/DOWN arrows)
Slider Adjust      ← / → (LEFT/RIGHT arrows)
Gate Control       SPACE (hold to play, release to stop)
Play Button        Click ▶ PLAY button
Stop Button        Click ■ STOP button  
Presets 1-5        1 2 3 4 5 keys
Quit               Q
```
