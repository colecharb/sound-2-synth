"""
synth_ui.py — Real-time FM Synthesizer UI with Textual

Interactive terminal UI for tweaking FM synth parameters in real-time with
low-latency audio streaming. Press SPACE to gate the synth, adjust parameters
with arrow keys and sliders, and switch presets with number keys.

Controls:
    SPACE       : Gate (hold to play, release to enter release phase)
    Arrow Keys  : Navigate and adjust slider values
    Tab         : Switch between sliders
    1-5         : Load presets (default, bell, bass, brass, noise)
    Q           : Quit
"""

import threading
import queue
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import sounddevice as sd
from textual.app import ComposeResult, RenderableType
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Input, Label, Button
from textual.widgets._input import Input as InputWidget
from textual.reactive import reactive
from textual.binding import Binding
from textual import work
from textual.app import App
from textual.events import Key
import torch
import asyncio
import time
from pynput import keyboard
import math

from synth import FMSynth


class NonFocusableButton(Button):
    """A button that cannot be focused via Tab or arrow keys."""
    
    def __init__(self, label: str, **kwargs):
        super().__init__(label, **kwargs)
        self.can_focus = False


class KeyboardWidget(Static):
    """A visual piano keyboard widget with key press feedback."""
    
    # Key mapping: char -> (note_name, octave_offset)
    # S is C4 (middle C), L is C5 (next octave)
    KEY_MAP = {
        's': ('C', 0),
        'd': ('D', 0),
        'f': ('E', 0),
        'g': ('F', 0),
        'h': ('G', 0),
        'j': ('A', 0),
        'k': ('B', 0),
        'l': ('C', 1),  # C5 (next octave - +1)
        
        'e': ('C#', 0),
        'r': ('D#', 0),
        'y': ('F#', 0),
        'u': ('G#', 0),
        'i': ('A#', 0),
    }
    
    # Note to semitone offset from C
    NOTE_SEMITONES = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    def __init__(self, audio_manager, **kwargs):
        super().__init__(**kwargs)
        self.audio_manager = audio_manager
        self.octave = 4  # Middle octave
        self.pressed_keys = set()  # Track which keys are currently held
        self.can_focus = False
    
    def get_frequency(self, note_name: str, octave: int) -> float:
        """Calculate frequency for a given note and octave."""
        # A4 (octave 4) = 440 Hz
        semitones_from_a4 = self.NOTE_SEMITONES[note_name] - self.NOTE_SEMITONES['A']
        semitones_from_a4 += (octave - 4) * 12
        return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))
    
    def render(self) -> str:
        """Render the keyboard with visual feedback."""
        # Build keyboard display
        lines = []
        lines.append(f"[bold cyan]Keyboard (Octave {self.octave})[/bold cyan]")
        lines.append("")
        
        # Black keys row (C# D# _ F# G# A# _)
        black_row = ""
        black_positions = [
            ('e', 'C#'),
            ('r', 'D#'),
            None,
            ('y', 'F#'),
            ('u', 'G#'),
            ('i', 'A#'),
            None,
        ]
        for item in black_positions:
            if item:
                key_char, note_name = item
                is_pressed = key_char in self.pressed_keys
                if is_pressed:
                    black_row += "[bold white on black]█[/bold white on black] "
                else:
                    black_row += "[black on gray37]█[/black on gray37] "
            else:
                black_row += "  "
        lines.append(black_row)
        
        # White keys row (C D E F G A B | C)
        white_row = ""
        white_keys = [
            ('s', 'C'),
            ('d', 'D'),
            ('f', 'E'),
            ('g', 'F'),
            ('h', 'G'),
            ('j', 'A'),
            ('k', 'B'),
            ('l', 'C↑'),  # C in next octave
        ]
        for key_char, note_name in white_keys:
            is_pressed = key_char in self.pressed_keys
            if is_pressed:
                white_row += f"[bold white on blue]{note_name:^3}[/bold white on blue]"
            else:
                white_row += f"[white on dark_gray]{note_name:^3}[/white on dark_gray]"
        lines.append(white_row)
        
        # Controls info
        lines.append("")
        lines.append("[dim]Keys:[/dim] S D F G H J K L (white) | E R Y U I (black)")
        lines.append(f"[dim]Octave:[/dim] [cyan][{self.octave}][/cyan] [yellow]+ / - [/yellow] to change")
        
        return "\n".join(lines)
    
    def key_pressed(self, key_char: str):
        """Called when a key is pressed."""
        if key_char.lower() in self.KEY_MAP:
            self.pressed_keys.add(key_char.lower())
            note_name, octave_offset = self.KEY_MAP[key_char.lower()]
            freq = self.get_frequency(note_name, self.octave + octave_offset)
            self.audio_manager.set_carrier_freq(freq)
            self.audio_manager.set_gate(True)
            self.refresh()
    
    def key_released(self, key_char: str):
        """Called when a key is released."""
        if key_char.lower() in self.KEY_MAP:
            self.pressed_keys.discard(key_char.lower())
            if not self.pressed_keys:
                # All keys released
                self.audio_manager.set_gate(False)
            else:
                # Other keys still pressed, play the first one
                first_key = list(self.pressed_keys)[0]
                note_name, octave_offset = self.KEY_MAP[first_key]
                freq = self.get_frequency(note_name, self.octave + octave_offset)
                self.audio_manager.set_carrier_freq(freq)
            self.refresh()
    
    def change_octave(self, delta: int):
        """Change the octave."""
        self.octave = max(0, min(8, self.octave + delta))
        self.refresh()
        if self.pressed_keys:
            # Update frequency if keys are currently pressed
            first_key = list(self.pressed_keys)[0]
            note_name, octave_offset = self.KEY_MAP[first_key]
            freq = self.get_frequency(note_name, self.octave + octave_offset)
            self.audio_manager.set_carrier_freq(freq)


@dataclass
class SynthParameters:
    """Container for all synth parameters."""
    carrier_freq: float = 440.0
    mod_ratio: float = 2.0
    mod_index: float = 5.0
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.3

    def to_dict(self):
        return asdict(self)


# Preset definitions
PRESETS = {
    "default": SynthParameters(
        carrier_freq=440.0,
        mod_ratio=2.0,
        mod_index=5.0,
        attack=0.01,
        decay=0.1,
        sustain=0.7,
        release=0.3,
    ),
    "bell": SynthParameters(
        carrier_freq=880.0,
        mod_ratio=1.5,
        mod_index=8.0,
        attack=0.005,
        decay=0.3,
        sustain=0.5,
        release=0.5,
    ),
    "bass": SynthParameters(
        carrier_freq=55.0,
        mod_ratio=0.5,
        mod_index=3.0,
        attack=0.05,
        decay=0.1,
        sustain=0.8,
        release=0.2,
    ),
    "brass": SynthParameters(
        carrier_freq=220.0,
        mod_ratio=1.0,
        mod_index=4.0,
        attack=0.02,
        decay=0.15,
        sustain=0.9,
        release=0.4,
    ),
    "noise": SynthParameters(
        carrier_freq=200.0,
        mod_ratio=3.5,
        mod_index=15.0,
        attack=0.001,
        decay=0.05,
        sustain=0.6,
        release=0.2,
    ),
}

# Parameter ranges for validation
PARAM_RANGES = {
    "carrier_freq": (20.0, 2000.0),
    "mod_ratio": (0.5, 4.0),
    "mod_index": (0.0, 20.0),
    "attack": (0.001, 1.0),
    "decay": (0.001, 2.0),
    "sustain": (0.0, 1.0),
    "release": (0.001, 2.0),
}

# Step sizes for parameter adjustments (in actual parameter units)
PARAM_STEPS = {
    "carrier_freq": 10.0,   # Hz
    "mod_ratio": 0.1,       # Fine control for mod_ratio
    "mod_index": 0.5,       # 
    "attack": 0.01,         # seconds
    "decay": 0.01,          # seconds
    "sustain": 0.05,        # 0-1 range
    "release": 0.01,        # seconds
}


class AudioManager:
    """Manages real-time audio streaming and synth control."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 2048):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.synth = FMSynth(sample_rate=sample_rate)

        self.params = SynthParameters()
        self.params_lock = threading.Lock()

        # Gate control
        self.gate_open = False
        self.time_offset = 0.0
        self.gate_release_time: Optional[float] = -1e6  # Start fully released (silent)
        
        # Keyboard note control
        self.keyboard_freq: Optional[float] = None

        # Audio stream
        self.stream = None
        self.is_running = False
        self.output_queue = queue.Queue()

    def set_parameters(self, params: SynthParameters):
        """Thread-safely update synth parameters."""
        with self.params_lock:
            self.params = params

    def set_gate(self, open: bool):
        """Set gate state (True = gate open/playing, False = gate closed/release)."""
        if open and not self.gate_open:
            # Gate just opened
            self.gate_open = True
            self.gate_release_time = None
            self.time_offset = 0.0
        elif not open and self.gate_open:
            # Gate just closed - enter release phase
            self.gate_open = False
            self.gate_release_time = self.time_offset
    
    def set_carrier_freq(self, freq: float):
        """Set carrier frequency for keyboard notes."""
        with self.params_lock:
            self.keyboard_freq = freq
            self.params.carrier_freq = freq

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice streaming."""
        if status:
            print(f"Audio callback status: {status}")

        with self.params_lock:
            params = SynthParameters(**asdict(self.params))

        # Generate chunk
        duration = frames / self.sample_rate
        chunk = self.synth.generate_chunk(
            duration=duration,
            time_offset=self.time_offset,
            carrier_freq=params.carrier_freq,
            mod_ratio=params.mod_ratio,
            mod_index=params.mod_index,
            attack=params.attack,
            decay=params.decay,
            sustain=params.sustain,
            release=params.release,
            gate_open=self.gate_open,
            gate_release_time=self.gate_release_time,
        )

        # Convert to numpy
        chunk_np = chunk.numpy().astype(np.float32)

        # Handle size mismatch (shouldn't happen, but be safe)
        if len(chunk_np) < frames:
            chunk_np = np.pad(chunk_np, (0, frames - len(chunk_np)))
        elif len(chunk_np) > frames:
            chunk_np = chunk_np[:frames]

        outdata[:] = chunk_np.reshape(-1, 1)
        self.time_offset += duration

    def start(self):
        """Start audio streaming."""
        if not self.is_running:
            self.is_running = True
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                latency="low",
            )
            self.stream.start()

    def stop(self):
        """Stop audio streaming."""
        if self.is_running:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None


class ParameterSlider(Static):
    """A slider control for a single parameter."""

    value = reactive(0.5)

    def __init__(self, param_name: str, min_val: float, max_val: float, initial: float, slider_id: Optional[str] = None):
        super().__init__(id=slider_id)
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val
        self.value = (initial - min_val) / (max_val - min_val)
        self.can_focus = True  # Enable focus for this widget

    def watch_value(self, old_value: float, new_value: float) -> None:
        """Called when value changes (via reactive system)."""
        # Trigger a refresh to show the updated slider
        self.refresh()
        # Post a message up to parent to trigger parameter update
        try:
            if hasattr(self.app, 'on_slider_change'):
                self.app.on_slider_change()
        except Exception:
            # Ignore errors when app context is not available (e.g., during testing)
            pass

    def render(self) -> RenderableType:
        """Render the slider as a visual bar."""
        bar_width = 25
        filled = int(self.value * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Calculate actual value
        actual_value = self.min_val + self.value * (self.max_val - self.min_val)

        # Show if focused
        focused_indicator = " ◄ FOCUSED" if self.has_focus else ""
        return f"[bold cyan]{self.param_name:12}[/bold cyan] {bar} [yellow]{actual_value:8.4f}[/yellow]{focused_indicator}"

    def get_value(self) -> float:
        """Get the actual parameter value."""
        return self.min_val + self.value * (self.max_val - self.min_val)

    def set_value(self, actual_value: float):
        """Set the slider from an actual parameter value."""
        actual_value = max(self.min_val, min(self.max_val, actual_value))
        self.value = (actual_value - self.min_val) / (self.max_val - self.min_val)

    def increment(self, delta: Optional[float] = None):
        """Increment the slider value by the parameter's step size."""
        if delta is None:
            # Use parameter-specific step size
            step = PARAM_STEPS.get(self.param_name, 0.01)
            # Convert actual parameter step to normalized slider step
            delta = step / (self.max_val - self.min_val)
        self.value = max(0.0, min(1.0, self.value + delta))

    def decrement(self, delta: Optional[float] = None):
        """Decrement the slider value by the parameter's step size."""
        if delta is None:
            # Use parameter-specific step size
            step = PARAM_STEPS.get(self.param_name, 0.01)
            # Convert actual parameter step to normalized slider step
            delta = step / (self.max_val - self.min_val)
        self.value = max(0.0, min(1.0, self.value - delta))

    def on_key(self, event: Key) -> None:
        """Handle keyboard input when slider is focused."""
        if event.key == "right":
            event.prevent_default()
            self.increment()
        elif event.key == "left":
            event.prevent_default()
            self.decrement()
        elif event.key == "up":
            # UP navigates to previous slider
            event.prevent_default()
            try:
                self.app.action_focus_prev_slider()
            except Exception:
                pass
        elif event.key == "down":
            # DOWN navigates to next slider
            event.prevent_default()
            try:
                self.app.action_focus_next_slider()
            except Exception:
                pass


class SynthUI(App):
    """Main Textual application for synth control."""

    BINDINGS = [
        Binding("q", "quit()", "Quit", show=True),
        Binding("1", "load_preset('default')", "Preset 1", show=False),
        Binding("2", "load_preset('bell')", "Preset 2", show=False),
        Binding("3", "load_preset('bass')", "Preset 3", show=False),
        Binding("4", "load_preset('brass')", "Preset 4", show=False),
        Binding("5", "load_preset('noise')", "Preset 5", show=False),
    ]

    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }

    #header {
        height: 2;
        background: $boost;
        border: solid $primary;
        padding: 0 1;
    }

    #main-container {
        height: 1fr;
    }

    #sliders-container {
        width: 1fr;
        height: auto;
        border: solid $accent;
        padding: 1;
    }

    #info-panel {
        width: 50;
        border: solid $accent;
        padding: 1;
        overflow: auto;
    }

    #footer {
        height: 2;
        background: $boost;
        border: solid $primary;
        padding: 0 1;
    }

    ParameterSlider {
        height: 1;
        margin: 0;
        padding: 0 1;
        border: none;
    }

    .value-display {
        width: 1fr;
        height: auto;
    }

    .status-line {
        width: 1fr;
        height: auto;
        text-align: left;
    }

    Button {
        margin: 0 1;
        width: 1fr;
    }
    """

    def __init__(self):
        super().__init__()
        self.audio_manager = AudioManager()
        self.sliders: dict[str, ParameterSlider] = {}
        self.gate_is_open = False
        self.focused_slider_index = 0
        self.space_pressed_down = False  # True while space is physically held
        self._listener = None  # Keyboard listener
        self.keyboard_widget: Optional[KeyboardWidget] = None
        self.param_names = [
            "carrier_freq",
            "mod_ratio",
            "mod_index",
            "attack",
            "decay",
            "sustain",
            "release",
        ]

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("🎛️  FM Synth Live Control", id="header")

        with Horizontal(id="main-container"):
            # Left: Sliders container - vertical stack of all sliders
            with Vertical(id="sliders-container"):
                for param_name in self.param_names:
                    min_val, max_val = PARAM_RANGES[param_name]
                    initial = getattr(PRESETS["default"], param_name)
                    yield ParameterSlider(param_name, min_val, max_val, initial)

            # Right: Info panel and keyboard
            with Vertical(id="info-panel"):
                yield Label("[bold yellow]PRESETS[/bold yellow]")
                yield NonFocusableButton("1: Default", id="preset-default", variant="primary")
                yield NonFocusableButton("2: Bell", id="preset-bell")
                yield NonFocusableButton("3: Bass", id="preset-bass")
                yield NonFocusableButton("4: Brass", id="preset-brass")
                yield NonFocusableButton("5: Noise", id="preset-noise")
                yield Label("")
                yield Label("[bold cyan]GATE CONTROL[/bold cyan]")
                yield NonFocusableButton("▶ PLAY (Space)", id="gate-play", variant="success")
                yield NonFocusableButton("■ STOP", id="gate-stop", variant="error")
                yield Label("[red]IDLE[/red]", id="gate-status")
                yield Label("")
                yield Label("[bold green]CURRENT VALUES[/bold green]")
                for param_name in self.param_names:
                    yield Label(f"{param_name}: 0.00", id=f"val-{param_name}")
                yield Label("")
                yield KeyboardWidget(self.audio_manager, id="keyboard")

        yield Static(
            "[dim]SPACE[/dim] Toggle Gate  |  [dim]1-5[/dim] Presets  |  [dim]↑/↓[/dim] Adjust  |  [dim]Q[/dim] Quit  |  [dim]S-L[/dim] Play Keys",
            id="footer",
        )

    def on_mount(self) -> None:
        """Initialize after widgets are mounted."""
        # Store slider references by querying all ParameterSliders
        all_sliders = self.query("ParameterSlider")
        for i, slider in enumerate(all_sliders):
            if i < len(self.param_names):
                param_name = self.param_names[i]
                self.sliders[param_name] = slider

        # Get keyboard widget reference
        try:
            self.keyboard_widget = self.query_one("#keyboard", KeyboardWidget)
        except Exception:
            pass

        # Start audio
        self.audio_manager.start()

        # Focus first slider
        if self.sliders:
            first_slider = list(self.sliders.values())[0]
            first_slider.focus()

        # Start global key listener for spacebar and keyboard notes
        self._start_key_listener()

    def on_unmount(self) -> None:
        """Clean up on exit."""
        self.audio_manager.stop()
        self._stop_key_listener()

    def _start_key_listener(self) -> None:
        """Start global keyboard listener for spacebar and note keys."""
        def on_press(key):
            try:
                # Handle spacebar for gate
                if key == keyboard.Key.space:
                    if not self.space_pressed_down:
                        self.space_pressed_down = True
                        self.gate_is_open = True
                        self.audio_manager.set_gate(True)
                        self.update_gate_status()
                
                # Handle note keys and octave changes
                elif hasattr(key, 'char') and key.char is not None:
                    char = key.char
                    # Octave up/down
                    if char == '+' or char == '=':
                        if self.keyboard_widget:
                            self.keyboard_widget.change_octave(1)
                    elif char == '-' or char == '_':
                        if self.keyboard_widget:
                            self.keyboard_widget.change_octave(-1)
                    # Note keys
                    else:
                        if self.keyboard_widget:
                            self.keyboard_widget.key_pressed(char)
            except AttributeError:
                pass

        def on_release(key):
            try:
                # Handle spacebar for gate
                if key == keyboard.Key.space:
                    if self.space_pressed_down:
                        self.space_pressed_down = False
                        self.gate_is_open = False
                        self.audio_manager.set_gate(False)
                        self.update_gate_status()
                
                # Handle note keys
                elif hasattr(key, 'char') and key.char is not None:
                    char = key.char
                    if char not in ['+', '=', '-', '_']:
                        if self.keyboard_widget:
                            self.keyboard_widget.key_released(char)
            except AttributeError:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _stop_key_listener(self) -> None:
        """Stop global keyboard listener."""
        if hasattr(self, '_listener') and self._listener is not None:
            self._listener.stop()



    def action_adjust_slider(self, delta: float) -> None:
        """Adjust the currently focused slider."""
        # This is now handled by the slider's on_key method
        # But keeping this for backward compatibility
        focused = self.focused
        if isinstance(focused, ParameterSlider):
            if delta > 0:
                focused.increment(abs(delta))
            else:
                focused.decrement(abs(delta))

    def action_focus_next_slider(self) -> None:
        """Focus the next slider in the list."""
        if not self.sliders:
            return
        
        slider_list = list(self.sliders.values())
        current = self.focused
        
        # Find current slider index
        try:
            current_index = slider_list.index(current)
            next_index = (current_index + 1) % len(slider_list)
        except (ValueError, TypeError):
            # Current is not a slider, focus the first one
            next_index = 0
        
        slider_list[next_index].focus()

    def action_focus_prev_slider(self) -> None:
        """Focus the previous slider in the list."""
        if not self.sliders:
            return
        
        slider_list = list(self.sliders.values())
        current = self.focused
        
        # Find current slider index
        try:
            current_index = slider_list.index(current)
            prev_index = (current_index - 1) % len(slider_list)
        except (ValueError, TypeError):
            # Current is not a slider, focus the last one
            prev_index = len(slider_list) - 1
        
        slider_list[prev_index].focus()

    def on_button_pressed(self, event) -> None:
        """Handle button clicks for presets and gate control."""
        button_id = event.button.id
        if button_id and button_id.startswith("preset-"):
            preset_name = button_id.replace("preset-", "")
            self.load_preset_impl(preset_name)
        elif button_id == "gate-play":
            self.gate_is_open = True
            self.audio_manager.set_gate(True)
            self.update_gate_status()
        elif button_id == "gate-stop":
            self.gate_is_open = False
            self.audio_manager.set_gate(False)
            self.update_gate_status()

    def action_load_preset(self, preset_name: str) -> None:
        """Load a preset by name."""
        self.load_preset_impl(preset_name)

    def load_preset_impl(self, preset_name: str) -> None:
        """Implementation of preset loading."""
        if preset_name not in PRESETS:
            return

        preset = PRESETS[preset_name]
        for param_name, slider in self.sliders.items():
            value = getattr(preset, param_name)
            slider.set_value(value)

        self.on_slider_change()
        self.notify(f"Loaded preset: {preset_name}")

    def on_slider_change(self) -> None:
        """Called when any slider changes."""
        # Collect current parameter values
        params = SynthParameters()
        for param_name, slider in self.sliders.items():
            setattr(params, param_name, slider.get_value())

        # Update audio manager
        self.audio_manager.set_parameters(params)

        # Update display
        self.update_value_display()

    def update_value_display(self) -> None:
        """Update the value display in the info panel."""
        for param_name, slider in self.sliders.items():
            value = slider.get_value()
            value_label = self.query_one(f"#val-{param_name}", Label)
            value_label.update(f"[cyan]{param_name}[/cyan]: {value:.4f}")

    def update_gate_status(self) -> None:
        """Update gate status indicator."""
        status_label = self.query_one("#gate-status", Label)
        if self.gate_is_open:
            status_label.update("[green]PLAYING[/green]")
        else:
            status_label.update("[red]IDLE[/red]")


def main():
    """Run the synth UI."""
    app = SynthUI()
    app.run()


if __name__ == "__main__":
    main()
