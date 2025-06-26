import sys
import math
import wave
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
import math, tempfile, time, glob, subprocess
try:
    import mido
except ImportError:
    mido = None
try:
    import importlib
    try:
        fluidsynth = importlib.import_module("fluidsynth")
    except (ImportError, FileNotFoundError, OSError):
        fluidsynth = None
except Exception:
    fluidsynth = None

from PySide6.QtWidgets import (
    QComboBox,
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressDialog,
)

try:
    import pyaudio  # type: ignore
except ImportError as e:
    pyaudio = None  # handle later

# ----------------------- Utility Functions ----------------------- #

SCALES = {
    "Pentatónica": [0,2,4,7,9],
    "Mayor": [0,2,4,5,7,9,11],
    "Menor": [0,2,3,5,7,8,10],
    "Menor Armónica": [0,2,3,5,7,8,11],
    "Menor Melódica": [0,2,3,5,7,9,11],
    "Dórica": [0,2,3,5,7,9,10],
    "Mixolidia": [0,2,4,5,7,9,10],
    "Blues Mayor": [0,2,3,4,7,9],
    "Blues Menor": [0,3,5,6,7,10],
}

CURRENT_SCALE = SCALES["Pentatónica"]

def _nearest_in_scale(midi_note: int) -> int:
    base = midi_note % 12
    offset_options = sorted(CURRENT_SCALE, key=lambda x: abs(x - base))
    chosen_offset = offset_options[0]
    return midi_note - base + chosen_offset

# Backward compatibility
_nearest_pentatonic = _nearest_in_scale


def equation_to_frequencies(equation: str, note_limit: int = 180) -> List[float]:
    """Convert chars to midi, quantize to CURRENT_SCALE, remove dups, add cadences with improved harmony and rhythm."""
    notes: List[int] = []
    last_midi = None
    
    # Limit the input length to prevent excessive notes
    max_chars = min(len(equation), note_limit // 2)  # Reserve space for embellishments
    equation = equation[:max_chars]
    
    # Improved note generation with better harmony and rhythm
    for i, ch in enumerate(equation):
        # Base MIDI note from character
        base_midi = (ord(ch) % 48) + 36
        
        # Quantize to current scale
        midi = _nearest_in_scale(base_midi)
        
        # Create rhythmic and melodic variety
        rhythm_pattern = i % 8  # 8-beat pattern
        
        # Add variety in octave selection based on rhythm position
        if rhythm_pattern == 0:  # Downbeat - use lower register
            octave_shift = -12 if midi > 60 else 0
        elif rhythm_pattern == 4:  # Backbeat - use higher register
            octave_shift = 12 if midi < 72 else 0
        elif rhythm_pattern in [2, 6]:  # Offbeats - use middle register
            octave_shift = 0
        else:  # Other positions - vary between registers
            octave_shift = 12 if i % 2 == 0 and midi < 72 else -12 if i % 2 == 1 and midi > 60 else 0
        
        midi += octave_shift
        
        # Avoid immediate duplicates with better interval selection
        if midi == last_midi:
            # Choose different intervals for variety
            intervals = [7, 12, -7, -12, 5, -5]  # Perfect 5th, Octave, etc.
            interval = intervals[i % len(intervals)]
            midi += interval
        
        # Ensure notes are in a good range for the instrument (but allow more variety)
        while midi < 36:  # Below C2 - too low
            midi += 12
        while midi > 96:  # Above C7 - too high
            midi -= 12
        
        notes.append(midi)
        last_midi = midi

    # Enhanced chord progression and cadences with rhythm (limited)
    if len(notes) >= 8:
        # More sophisticated chord progression with rhythm
        progression = [0, -3, 5, 7, 0, -3, 5, 7]  # I-vi-IV-V progression
        
        for i in range(7, min(len(notes), note_limit - 20), 8):  # Limit chord additions
            # Get the chord root for this measure
            chord_root = 60 + progression[(i//8) % len(progression)]  # C major as tonic
            
            # Create a richer chord (root, third, fifth, seventh)
            chord_notes = [
                chord_root,
                chord_root + 4,  # Major third
                chord_root + 7,  # Perfect fifth
                chord_root + 11  # Major seventh
            ]
            
            # Quantize chord notes to current scale
            chord_notes = [_nearest_in_scale(note) for note in chord_notes]
            
            # Replace the last note of the measure with a chord note
            if i < len(notes):
                # Choose the chord note closest to the original note
                original_note = notes[i]
                closest_chord_note = min(chord_notes, key=lambda x: abs(x - original_note))
                notes[i] = closest_chord_note
    
    # Add rhythmic embellishments and passing tones (limited)
    if len(notes) > 4 and len(notes) < note_limit - 10:
        # Add rhythmic hits and passing tones
        enhanced_notes = []
        for i, note in enumerate(notes):
            enhanced_notes.append(note)
            
            # Add rhythmic hits on strong beats (limited)
            if i % 4 == 0 and i > 0 and len(enhanced_notes) < note_limit - 5:
                # Add a complementary note
                complement = note + 7 if note < 72 else note - 7  # Perfect 5th
                complement = _nearest_in_scale(complement)
                enhanced_notes.append(complement)
            
            # Add passing tones for smoother transitions (limited)
            if i < len(notes) - 1 and abs(notes[i] - notes[i+1]) > 7 and len(enhanced_notes) < note_limit - 5:
                # Add a passing tone
                passing_note = (notes[i] + notes[i+1]) // 2
                passing_note = _nearest_in_scale(passing_note)
                enhanced_notes.append(passing_note)
        
        notes = enhanced_notes
    
    # Add some high and low note flourishes for guzheng character (limited)
    if len(notes) > 8 and len(notes) < note_limit - 5:
        # Add high note flourishes every 8 notes (limited)
        for i in range(7, min(len(notes), note_limit - 10), 8):
            if i < len(notes):
                # Add a high flourish note
                high_note = notes[i] + 12  # One octave higher
                high_note = _nearest_in_scale(high_note)
                if high_note <= 96:  # Within reasonable range
                    notes.insert(i + 1, high_note)
        
        # Add low note accents every 12 notes (limited)
        for i in range(11, min(len(notes), note_limit - 10), 12):
            if i < len(notes):
                # Add a low accent note
                low_note = notes[i] - 12  # One octave lower
                low_note = _nearest_in_scale(low_note)
                if low_note >= 36:  # Within reasonable range
                    notes.insert(i + 1, low_note)
    
    # Final limit check
    notes = notes[:note_limit]
    
    freqs = [440.0 * (2 ** ((m - 69) / 12)) for m in notes]
    return freqs


def _square_wave(f: float, t: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(2 * np.pi * f * t))


def _kick(t: np.ndarray) -> np.ndarray:
    env = np.exp(-40 * t)
    return env * np.sin(2 * np.pi * 60 * t)


def _snare(t: np.ndarray) -> np.ndarray:
    env = np.exp(-20 * t)
    noise = np.random.uniform(-1, 1, t.shape)
    return env * noise


def _chord_for_freq(f: float, t: np.ndarray) -> np.ndarray:
    # Enhanced harmonic chord with better voice leading and natural sound
    # Use more natural intervals and better voice distribution
    
    # Create a richer chord with better harmonic content
    root = _square_wave(f, t)
    third = _square_wave(f * 1.25, t)  # Major third
    fifth = _square_wave(f * 1.5, t)   # Perfect fifth
    octave = _square_wave(f * 2.0, t)  # Octave
    
    # Add some harmonic overtones for more natural sound
    overtone1 = _square_wave(f * 1.125, t) * 0.3  # Minor ninth
    overtone2 = _square_wave(f * 1.75, t) * 0.2   # Minor seventh
    
    # Blend the voices with different weights for more natural sound
    chord = (root * 0.4 + third * 0.3 + fifth * 0.2 + octave * 0.1 + 
             overtone1 * 0.3 + overtone2 * 0.2)
    
    # Normalize to prevent clipping
    return chord / np.max(np.abs(chord)) if np.max(np.abs(chord)) > 0 else chord


def _freq_to_midi(f: float) -> int:
    return int(round(69 + 12*math.log2(f/440.0)))

def _write_midi(path: Path, freqs: List[float], bpm:int=120):
    if mido is None:
        return False
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack(); mid.tracks.append(track)
    step_ticks = 240  # eighth note
    for f in freqs:
        note = _freq_to_midi(f)
        track.append(mido.Message('note_on', note=note, velocity=100, time=0))
        track.append(mido.Message('note_off', note=note, velocity=0, time=step_ticks))
    mid.save(path)
    return True

def _render_sf2_to_wav_improved(mid_path: Path, wav_path: Path, sf2_path: Path, sample_rate:int=44100):
    """Improved SF2 rendering with better harmony, faster processing, and optimized note selection."""
    if fluidsynth is None:
        print("FluidSynth not available")
        return False
    
    if mido is None:
        print("Mido not available for MIDI processing")
        return False
    
    try:
        # Create synth with optimized settings
        fs = fluidsynth.Synth(samplerate=sample_rate)
        
        # Optimized settings for better performance and sound
        try:
            fs.setting('audio.driver', 'file')
            fs.setting('audio.file.name', str(wav_path))
            fs.setting('audio.file.type', 'wav')
            # Remove problematic settings that don't exist in this FluidSynth version
            # fs.setting('synth.audio-channels', '2')  # Not available
            # fs.setting('synth.audio-groups', '2')    # Not available
            # fs.setting('synth.sample-rate', str(self.sample_rate))  # Not available
            # fs.setting('synth.gain', '0.8')          # Not available
            # fs.setting('synth.polyphony', '256')     # Not available
            # fs.setting('synth.reverb.active', 'yes') # Not available
            # fs.setting('synth.reverb.room-size', '0.2') # Not available
            # fs.setting('synth.reverb.damping', '0.0')   # Not available
            # fs.setting('synth.reverb.level', '0.3')     # Not available
            # fs.setting('synth.chorus.active', 'yes')    # Not available
            # fs.setting('synth.chorus.number', '3')      # Not available
            # fs.setting('synth.chorus.level', '0.2')     # Not available
            # fs.setting('synth.chorus.speed', '0.3')     # Not available
            # fs.setting('synth.chorus.depth', '8.0')     # Not available
        except Exception as e:
            print(f"Warning: Could not set all audio settings: {e}")
        
        # Start synth
        try:
            fs.start()
        except Exception as e:
            print(f"Failed to start FluidSynth: {e}")
            fs.delete()
            return False
        
        # Load SoundFont
        try:
            sfid = fs.sfload(str(sf2_path))
            fs.program_select(0, sfid, 0, 0)  # Channel 0, SoundFont ID, Bank 0, Program 0
        except Exception as e:
            print(f"Failed to load SoundFont: {e}")
            fs.delete()
            return False
        
        # Enhanced MIDI playback with better harmony
        try:
            print("Playing MIDI with enhanced harmony...")
            mid = mido.MidiFile(mid_path)
            
            # Calculate total duration for progress tracking
            total_time = sum(msg.time for msg in mid.play())
            mid = mido.MidiFile(mid_path)  # Reset for actual playback
            
            # Enhanced playback with better note selection and timing
            active_notes = set()
            start_time = time.time()
            elapsed_time = 0
            
            for msg in mid.play():
                elapsed_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Improve note selection for better harmony
                    note = msg.note
                    velocity = msg.velocity
                    
                    # Adjust note range for better sound (avoid very low notes)
                    if note < 48:  # Below C3
                        note = note + 12  # Transpose up one octave
                    
                    # Ensure velocity is in good range
                    if velocity < 40:
                        velocity = 60  # Minimum velocity for clarity
                    elif velocity > 100:
                        velocity = 90  # Maximum velocity to avoid harshness
                    
                    # Add some velocity variation for more natural sound
                    velocity = int(velocity * (0.9 + 0.2 * (hash(str(note)) % 10) / 10))
                    
                    fs.noteon(0, note, velocity)
                    active_notes.add(note)
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note = msg.note
                    if note < 48:  # Adjust for transposed notes
                        note = note + 12
                    
                    fs.noteoff(0, note)
                    active_notes.discard(note)
                
                # Wait for the correct time
                if msg.time > 0:
                    time.sleep(msg.time)
                
                # Progress update every 2 seconds
                if int(elapsed_time) % 2 == 0 and elapsed_time > 0:
                    progress = min(100, (elapsed_time / total_time) * 100)
                    print(f"Progress: {progress:.1f}% ({elapsed_time:.1f}s / {total_time:.1f}s)")
            
            # Wait for all notes to finish naturally
            time.sleep(1.5)
            
            # Ensure all notes are stopped
            for note in active_notes:
                fs.noteoff(0, note)
            
            print("Enhanced MIDI playback completed successfully")
            
        except Exception as e:
            print(f"Enhanced MIDI playback failed: {e}")
            fs.delete()
            return False
        
        fs.delete()
        return True
        
    except Exception as e:
        print(f"FluidSynth error: {e}")
        return False


def _adsr(t: np.ndarray, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.ones_like(t)
    a_len = int(len(t)*attack)
    d_len = int(len(t)*decay)
    r_len = int(len(t)*release)
    s_len = len(t)-a_len-d_len-r_len
    if a_len>0:
        env[:a_len]=np.linspace(0,1,a_len)
    if d_len>0:
        env[a_len:a_len+d_len]=np.linspace(1,sustain,d_len)
    if s_len>0:
        env[a_len+d_len:a_len+d_len+s_len]=sustain
    if r_len>0:
        env[-r_len:]=np.linspace(sustain,0,r_len)
    return env


def _midi_to_freq(m:int)->float:
    return 440.0 * (2 ** ((m - 69) / 12))

PROGRESSION_DEGREES=[0,-3,5,7]  # I vi IV V in semitone offsets relative to tonic


def compose_audio(frequencies: List[float], bpm: int = 120, sample_rate: int = 44100) -> np.ndarray:
    """Compose a chiptune-style track with melody, bass, kick, snare, and enhanced harmony."""
    if not frequencies:
        return np.array([], dtype=np.int16)

    base_step = 60 / bpm / 2  # eighth note
    segments: List[np.ndarray] = []
    
    # Track chord progression for better harmony
    current_chord_root = 60  # Start with C
    chord_progression = [0, -3, 5, 7, 0, -3, 5, 7]  # I-vi-IV-V progression

    for i, root_freq in enumerate(frequencies):
        # Determine chord root for this measure (every 8 steps)
        chord_index = (i // 8) % len(chord_progression)
        degree = chord_progression[chord_index]
        chord_root_midi = _nearest_in_scale(60 + degree)  # C major as tonic
        chord_root_freq = _midi_to_freq(chord_root_midi)
        
        # Smooth chord transitions
        if i % 8 == 0:  # New measure
            current_chord_root = chord_root_freq

        # Enhanced melody with better voice leading
        melody_freq = root_freq
        
        # Add octave variety based on rhythm position
        rhythm_pos = i % 8
        if rhythm_pos == 0:  # Downbeat - emphasize
            melody_freq = root_freq
        elif rhythm_pos == 4:  # Backbeat - contrast
            melody_freq = root_freq * 2 if root_freq < 800 else root_freq / 2
        elif rhythm_pos in [2, 6]:  # Offbeats - middle
            melody_freq = root_freq * 1.5 if root_freq < 600 else root_freq / 1.5
        else:  # Other positions - vary
            melody_freq = root_freq * (1.25 if i % 2 == 0 else 0.75)
        
        # Better chord tone selection for smoother harmony
        chord_tones = [
            current_chord_root,           # Root
            current_chord_root * 1.25,    # Major third
            current_chord_root * 1.5,     # Perfect fifth
            current_chord_root * 1.875,   # Major seventh
            current_chord_root * 2.0      # Octave
        ]
        
        # Find the closest chord tone for better voice leading
        nearest = min(chord_tones, key=lambda x: abs(x - melody_freq))
        melody_freq = nearest
        
        # Consistent timing - no swing for better rhythm
        step_duration = base_step
        t_step = np.arange(int(sample_rate * step_duration)) / sample_rate
        
        # Enhanced chord with better voice distribution
        chord = _chord_for_freq(current_chord_root, t_step) * _adsr(t_step, 0.05, 0.1, 0.7, 0.15)
        
        # Melody overlay with better balance
        melody_volume = 0.5 if rhythm_pos == 0 else 0.3  # Consistent volume
        chord += melody_volume * _square_wave(melody_freq, t_step) * _adsr(t_step, 0.02, 0.05, 0.6, 0.2)
        
        # Bass with consistent rhythm
        if i % 4 == 0:  # Every downbeat
            bass_freq = current_chord_root / 2
            chord += 0.6 * _square_wave(bass_freq, t_step) * _adsr(t_step, 0.05, 0.2, 0.5, 0.2)
        
        # Simplified percussion pattern
        if i % 8 == 0:  # Kick on 1
            chord += 0.8 * _kick(t_step)
        elif i % 8 == 4:  # Snare on 3
            chord += 0.4 * _snare(t_step)
        
        # Hi-hat pattern
        if i % 2 == 0:  # Every even step
            hat = _snare(t_step) * 0.2
            chord += hat
        
        segments.append(chord)

    audio = np.concatenate(segments)
    
    # Better normalization to prevent clipping
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.8  # Leave some headroom
    
    audio_int16 = np.int16(audio * 32767)
    return audio_int16.astype(np.int16)


def _harmonic_chord(f: float, t: np.ndarray) -> np.ndarray:
    """Generate a more harmonic chord using sine waves and better voice leading."""
    # Use sine waves for more natural harmonic sound
    def sine_wave(freq: float, t_array: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * freq * t_array)
    
    # Create chord with natural harmonic series
    root = sine_wave(f, t)
    third = sine_wave(f * 1.25, t)      # Major third
    fifth = sine_wave(f * 1.5, t)       # Perfect fifth
    octave = sine_wave(f * 2.0, t)      # Octave
    
    # Add some natural overtones
    overtone1 = sine_wave(f * 1.125, t) * 0.2  # Minor ninth
    overtone2 = sine_wave(f * 1.75, t) * 0.15  # Minor seventh
    
    # Blend with natural harmonic weights
    chord = (root * 0.5 + third * 0.3 + fifth * 0.25 + octave * 0.15 + 
             overtone1 * 0.2 + overtone2 * 0.15)
    
    # Normalize
    max_val = np.max(np.abs(chord))
    if max_val > 0:
        chord = chord / max_val * 0.7  # Leave headroom
    
    return chord


def _smooth_chord_progression(frequencies: List[float], bpm: int = 120, sample_rate: int = 44100) -> np.ndarray:
    """Generate smooth chord progression with better voice leading."""
    if not frequencies:
        return np.array([], dtype=np.int16)

    base_step = 60 / bpm / 2  # eighth note
    segments: List[np.ndarray] = []
    
    # Chord progression with smooth transitions
    chord_progression = [0, -3, 5, 7, 0, -3, 5, 7]  # I-vi-IV-V
    current_chord_root = 60  # C major

    for i, root_freq in enumerate(frequencies):
        # Determine chord for this measure
        chord_index = (i // 8) % len(chord_progression)
        degree = chord_progression[chord_index]
        chord_root_midi = _nearest_in_scale(60 + degree)
        chord_root_freq = _midi_to_freq(chord_root_midi)
        
        # Smooth chord transition
        if i % 8 == 0:
            current_chord_root = chord_root_freq
        
        # Melody note selection with voice leading
        melody_freq = root_freq
        rhythm_pos = i % 8
        
        # Octave variety
        if rhythm_pos == 0:
            melody_freq = root_freq
        elif rhythm_pos == 4:
            melody_freq = root_freq * 2 if root_freq < 800 else root_freq / 2
        else:
            melody_freq = root_freq * 1.5 if root_freq < 600 else root_freq / 1.5
        
        # Voice leading - find closest chord tone
        chord_tones = [
            current_chord_root,
            current_chord_root * 1.25,
            current_chord_root * 1.5,
            current_chord_root * 1.875
        ]
        melody_freq = min(chord_tones, key=lambda x: abs(x - melody_freq))
        
        # Generate audio segment
        step_duration = base_step
        t_step = np.arange(int(sample_rate * step_duration)) / sample_rate
        
        # Harmonic chord as base
        chord = _harmonic_chord(current_chord_root, t_step) * _adsr(t_step, 0.05, 0.1, 0.7, 0.15)
        
        # Add melody
        melody_volume = 0.4 if rhythm_pos == 0 else 0.25
        chord += melody_volume * np.sin(2 * np.pi * melody_freq * t_step) * _adsr(t_step, 0.02, 0.05, 0.6, 0.2)
        
        # Add bass on downbeats
        if i % 4 == 0:
            bass_freq = current_chord_root / 2
            chord += 0.5 * np.sin(2 * np.pi * bass_freq * t_step) * _adsr(t_step, 0.05, 0.2, 0.5, 0.2)
        
        # Simple percussion
        if i % 8 == 0:
            chord += 0.6 * _kick(t_step)
        elif i % 8 == 4:
            chord += 0.3 * _snare(t_step)
        
        if i % 2 == 0:
            chord += 0.15 * _snare(t_step)
        
        segments.append(chord)

    audio = np.concatenate(segments)
    
    # Normalize
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.8
    
    audio_int16 = np.int16(audio * 32767)
    return audio_int16.astype(np.int16)


def image_to_frequencies(img_path: Path, note_limit: int = 180) -> List[float]:
    """Convert an image to a list of frequencies by mapping pixel brightness to pitch with improved harmony and rhythm."""
    try:
        img = Image.open(img_path).convert('L')  # grayscale
    except Exception:
        return []
    
    # Resize for lighter processing
    img = img.resize((128, 64))
    pixels = np.array(img).flatten()
    freqs: List[float] = []
    notes: List[int] = []
    
    # Calculate sampling rate to stay within note limit
    total_pixels = len(pixels)
    sample_rate = max(25, total_pixels // (note_limit // 2))  # Adjust sampling to stay within limit
    
    # Sample pixels with rhythm pattern
    for i, val in enumerate(pixels[::sample_rate]):
        if len(notes) >= note_limit:  # Stop when limit reached
            break
            
        # Map brightness to MIDI note with more variety
        midi = int(val / 255 * 60) + 36  # Wider range: 36-96 (C2-C7)
        midi = _nearest_in_scale(midi)
        
        # Add rhythmic variety based on position
        rhythm_pattern = i % 8
        
        # Vary octave based on rhythm position
        if rhythm_pattern == 0:  # Downbeat - emphasize
            octave_shift = 0  # Keep original
        elif rhythm_pattern == 4:  # Backbeat - contrast
            octave_shift = 12 if midi < 72 else -12
        elif rhythm_pattern in [2, 6]:  # Offbeats - middle
            octave_shift = 0
        else:  # Other positions - vary
            octave_shift = 12 if i % 2 == 0 and midi < 72 else -12 if i % 2 == 1 and midi > 60 else 0
        
        midi += octave_shift
        
        # Ensure good range
        while midi < 36:
            midi += 12
        while midi > 96:
            midi -= 12
        
        notes.append(midi)
    
    # Add rhythmic embellishments for images too (limited)
    if len(notes) > 4 and len(notes) < note_limit - 10:
        enhanced_notes = []
        for i, note in enumerate(notes):
            enhanced_notes.append(note)
            
            # Add rhythmic hits on strong beats (limited)
            if i % 4 == 0 and i > 0 and len(enhanced_notes) < note_limit - 5:
                complement = note + 7 if note < 72 else note - 7
                complement = _nearest_in_scale(complement)
                enhanced_notes.append(complement)
        
        notes = enhanced_notes
    
    # Add high and low flourishes for guzheng character (limited)
    if len(notes) > 8 and len(notes) < note_limit - 5:
        # High flourishes (limited)
        for i in range(7, min(len(notes), note_limit - 10), 8):
            if i < len(notes):
                high_note = notes[i] + 12
                high_note = _nearest_in_scale(high_note)
                if high_note <= 96:
                    notes.insert(i + 1, high_note)
        
        # Low accents (limited)
        for i in range(11, min(len(notes), note_limit - 10), 12):
            if i < len(notes):
                low_note = notes[i] - 12
                low_note = _nearest_in_scale(low_note)
                if low_note >= 36:
                    notes.insert(i + 1, low_note)
    
    # Final limit check
    notes = notes[:note_limit]
    
    freqs = [440.0 * (2 ** ((m - 69) / 12)) for m in notes]
    return freqs


def save_wav(path: Path, audio_data: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


# ----------------------- Worker Thread ----------------------- #

class PlayThread(QThread):
    finished = Signal()

    def __init__(self, audio_data: np.ndarray, sample_rate: int = 44100):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def run(self):
        if pyaudio is None:
            self.finished.emit()
            return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True)
        stream.write(self.audio_data.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.finished.emit()


class SF2Worker(QThread):
    finished = Signal(bool)  # Success status
    progress = Signal(int)   # Progress percentage
    status = Signal(str)     # Status message

    def __init__(self, mid_path: Path, wav_path: Path, sf2_path: Path, sample_rate: int = 44100):
        super().__init__()
        self.mid_path = mid_path
        self.wav_path = wav_path
        self.sf2_path = sf2_path
        self.sample_rate = sample_rate

    def run(self):
        if fluidsynth is None:
            self.status.emit("FluidSynth no disponible")
            self.finished.emit(False)
            return
        
        if mido is None:
            self.status.emit("Mido no disponible")
            self.finished.emit(False)
            return
        
        try:
            self.status.emit("Inicializando FluidSynth...")
            self.progress.emit(5)
            
            # Create synth with optimized settings
            fs = fluidsynth.Synth(samplerate=self.sample_rate)
            
            # Optimized settings for better performance and sound
            try:
                fs.setting('audio.driver', 'file')
                fs.setting('audio.file.name', str(self.wav_path))
                fs.setting('audio.file.type', 'wav')
                # Remove problematic settings that don't exist in this FluidSynth version
                # fs.setting('synth.audio-channels', '2')  # Not available
                # fs.setting('synth.audio-groups', '2')    # Not available
                # fs.setting('synth.sample-rate', str(self.sample_rate))  # Not available
                # fs.setting('synth.gain', '0.8')          # Not available
                # fs.setting('synth.polyphony', '256')     # Not available
                # fs.setting('synth.reverb.active', 'yes') # Not available
                # fs.setting('synth.reverb.room-size', '0.2') # Not available
                # fs.setting('synth.reverb.damping', '0.0')   # Not available
                # fs.setting('synth.reverb.level', '0.3')     # Not available
                # fs.setting('synth.chorus.active', 'yes')    # Not available
                # fs.setting('synth.chorus.number', '3')      # Not available
                # fs.setting('synth.chorus.level', '0.2')     # Not available
                # fs.setting('synth.chorus.speed', '0.3')     # Not available
                # fs.setting('synth.chorus.depth', '8.0')     # Not available
            except Exception as e:
                self.status.emit(f"Advertencia: No se pudieron configurar todos los ajustes: {e}")
            
            self.status.emit("Iniciando sintetizador...")
            self.progress.emit(15)
            
            # Start synth
            try:
                fs.start()
            except Exception as e:
                self.status.emit(f"Error al iniciar FluidSynth: {e}")
                fs.delete()
                self.finished.emit(False)
                return
            
            self.status.emit("Cargando SoundFont...")
            self.progress.emit(25)
            
            # Load SoundFont
            try:
                sfid = fs.sfload(str(self.sf2_path))
                fs.program_select(0, sfid, 0, 0)  # Channel 0, SoundFont ID, Bank 0, Program 0
            except Exception as e:
                self.status.emit(f"Error al cargar SoundFont: {e}")
                fs.delete()
                self.finished.emit(False)
                return
            
            self.status.emit("Reproduciendo MIDI con armonía mejorada...")
            self.progress.emit(35)
            
            # Enhanced MIDI playback with better harmony
            try:
                mid = mido.MidiFile(self.mid_path)
                
                # Calculate total duration for progress tracking
                total_time = sum(msg.time for msg in mid.play())
                mid = mido.MidiFile(self.mid_path)  # Reset for actual playback
                
                # Enhanced playback with better note selection and timing
                active_notes = set()
                elapsed_time = 0
                
                for msg in mid.play():
                    elapsed_time += msg.time
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Improve note selection for better harmony
                        note = msg.note
                        velocity = msg.velocity
                        
                        # Adjust note range for better sound (avoid very low notes)
                        if note < 48:  # Below C3
                            note = note + 12  # Transpose up one octave
                        
                        # Ensure velocity is in good range
                        if velocity < 40:
                            velocity = 60  # Minimum velocity for clarity
                        elif velocity > 100:
                            velocity = 90  # Maximum velocity to avoid harshness
                        
                        # Add some velocity variation for more natural sound
                        velocity = int(velocity * (0.9 + 0.2 * (hash(str(note)) % 10) / 10))
                        
                        fs.noteon(0, note, velocity)
                        active_notes.add(note)
                        
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        note = msg.note
                        if note < 48:  # Adjust for transposed notes
                            note = note + 12
                        
                        fs.noteoff(0, note)
                        active_notes.discard(note)
                    
                    # Wait for the correct time
                    if msg.time > 0:
                        time.sleep(msg.time)
                    
                    # Update progress
                    progress = 35 + int((elapsed_time / total_time) * 60)  # 35% to 95%
                    self.progress.emit(progress)
                    
                    # Update status every 2 seconds
                    if int(elapsed_time) % 2 == 0 and elapsed_time > 0:
                        self.status.emit(f"Progreso: {progress:.1f}% ({elapsed_time:.1f}s / {total_time:.1f}s)")
                
                self.status.emit("Finalizando reproducción...")
                self.progress.emit(95)
                
                # Wait for all notes to finish naturally
                time.sleep(1.5)
                
                # Ensure all notes are stopped
                for note in active_notes:
                    fs.noteoff(0, note)
                
                self.status.emit("Reproducción MIDI mejorada completada exitosamente")
                self.progress.emit(100)
                
            except Exception as e:
                self.status.emit(f"Error en reproducción MIDI mejorada: {e}")
                fs.delete()
                self.finished.emit(False)
                return
            
            fs.delete()
            self.finished.emit(True)
            
        except Exception as e:
            self.status.emit(f"Error de FluidSynth: {e}")
            self.finished.emit(False)


# ----------------------- Image Drop Widget ----------------------- #

class ImageDropLabel(QLabel):
    """Etiqueta que acepta arrastrar imágenes y emite la ruta."""
    image_dropped = Signal(Path)

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # acepta solo archivos con extensión de imagen
            valid = any(Path(u.toLocalFile()).suffix.lower() in {'.png', '.jpg', '.jpeg'} for u in event.mimeData().urls())
            if valid:
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                self.image_dropped.emit(path)
                break


# ----------------------- Main Window ----------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math ⬌ Music Composer")
        self.setMinimumSize(900, 600)
        self._setup_ui()
        self.audio_data: Optional[np.ndarray] = None
        self.image_path: Optional[Path] = None
        self.play_thread: Optional[PlayThread] = None

    def _setup_ui(self):
        # Central splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel – input
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.equation_edit = QTextEdit()
        self.equation_edit.setPlaceholderText("Escribe una ecuación o expresión matemática…")
        left_layout.addWidget(QLabel("Ecuación"))
        left_layout.addWidget(self.equation_edit)
        
        # Note limit control
        note_limit_layout = QVBoxLayout()
        note_limit_layout.addWidget(QLabel("Límite de notas"))
        self.note_limit_edit = QLineEdit("180")
        self.note_limit_edit.setPlaceholderText("Número máximo de notas (ej: 180)")
        note_limit_layout.addWidget(self.note_limit_edit)
        left_layout.addLayout(note_limit_layout)
        
        # Scale selector
        self.scale_box = QComboBox()
        for name in SCALES.keys():
            self.scale_box.addItem(name)
        self.scale_box.currentTextChanged.connect(self.change_scale)
        left_layout.addWidget(QLabel("Escala"))
        left_layout.addWidget(self.scale_box)
        
        # Style selector
        self.style_box = QComboBox()
        self.style_box.addItem("Chiptune (Ondas cuadradas)")
        self.style_box.addItem("Armónico (Ondas sinusoidales)")
        left_layout.addWidget(QLabel("Estilo de sonido"))
        left_layout.addWidget(self.style_box)

        self.image_label = ImageDropLabel("Arrastra una imagen PNG/JPG aquí")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #666; padding: 20px;")
        self.image_label.setAcceptDrops(True)
        left_layout.addWidget(self.image_label)
        self.image_label.image_dropped.connect(self.load_image)
        browse_btn = QPushButton("📂 Abrir imagen…")
        browse_btn.clicked.connect(self.browse_image)
        left_layout.addWidget(browse_btn)

        self.generate_btn = QPushButton("Generar Composición")
        self.generate_btn.clicked.connect(self.generate_music)
        left_layout.addWidget(self.generate_btn)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # Right panel – explanation + controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.explanation_view = QTextEdit()
        self.explanation_view.setReadOnly(True)
        right_layout.addWidget(QLabel("Explicación"))
        right_layout.addWidget(self.explanation_view)

        self.play_btn = QPushButton(" Reproducir")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        right_layout.addWidget(self.play_btn)

        self.export_btn = QPushButton("Exportar WAV")
        self.export_btn.clicked.connect(self.export_audio)
        self.export_btn.setEnabled(False)
        right_layout.addWidget(self.export_btn)
        # SoundFont play button (enabled dynamically)
        self.play_sf2_btn = QPushButton("Reproducir SF2")
        self.play_sf2_btn.setEnabled(False)
        self.play_sf2_btn.clicked.connect(self.play_sf2)
        right_layout.addWidget(self.play_sf2_btn)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        self.setCentralWidget(splitter)

        # Dark style
        self.setStyleSheet("""
            QWidget { background: #222; color: #ddd; font-family: Segoe UI, sans-serif; }
            QLineEdit, QTextEdit { background: #333; color: #eee; border: 1px solid #555; }
            QPushButton { background: #444; border: 1px solid #666; padding: 6px 12px; }
            QPushButton:disabled { background: #555; color: #888; }
            QLabel { font-size: 13px; }
        """)

        # Menu (optional future use)
        export_action = QAction("Exportar WAV", self)
        export_action.triggered.connect(self.export_audio)
        file_menu = self.menuBar().addMenu("Archivo")
        file_menu.addAction(export_action)

    # ----------------------- Slots ----------------------- #



    def play_audio(self):
        if self.audio_data is None:
            return
        self.play_btn.setEnabled(False)
        thread = PlayThread(self.audio_data)
        def _on_finished():
            self.play_btn.setEnabled(True)
            self.play_thread = None
        thread.finished.connect(_on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        # Mantener referencia viva hasta que termine
        self.play_thread = thread

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path):
        self.image_path = path
        pixmap = QPixmap(str(path)).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

    def change_scale(self, name: str):
        global CURRENT_SCALE
        CURRENT_SCALE = SCALES[name]

    def generate_music(self):
        # Get note limit from UI
        try:
            note_limit = int(self.note_limit_edit.text())
            if note_limit <= 0 or note_limit > 500:
                note_limit = 180  # Default if invalid
        except ValueError:
            note_limit = 180  # Default if not a number
        
        equation_text = self.equation_edit.toPlainText().strip()
        if equation_text:
            freqs = equation_to_frequencies(equation_text, note_limit)
            explanation = (f"Entrada: ecuación con {len(equation_text)} caracteres\n"
                           f"Notas generadas: {len(freqs)} (límite: {note_limit})\n"
                           f"Frecuencias: {', '.join(f'{round(f,1)} Hz' for f in freqs[:10])}{'…' if len(freqs)>10 else ''}")
        elif self.image_path:
            freqs = image_to_frequencies(self.image_path, note_limit)
            explanation = (f"Entrada: imagen {self.image_path.name}\n"
                           f"Notas generadas: {len(freqs)} (límite: {note_limit})")
        else:
            QMessageBox.warning(self, "Vacío", "Por favor escribe una ecuación o carga una imagen.")
            return

        if not freqs:
            QMessageBox.critical(self, "Error", "No se pudieron generar notas a partir de la entrada.")
            return

        self.current_freqs = freqs
        
        # Choose composition style based on user selection
        style = self.style_box.currentText()
        if "Armónico" in style:
            self.audio_data = _smooth_chord_progression(freqs)
        else:
            self.audio_data = compose_audio(freqs)
        
        # Verify audio data is not silent
        if self.audio_data is not None and len(self.audio_data) > 0:
            max_amplitude = np.max(np.abs(self.audio_data))
            if max_amplitude < 100:  # Very low amplitude
                QMessageBox.warning(self, "Audio débil", "El audio generado es muy débil. Intenta con más caracteres o una imagen diferente.")
        
        # Enable SF2 button if a soundfont is available and libs present
        sf2_files = list(Path('.').glob('*.sf2'))
        self.sf2_path = sf2_files[0] if sf2_files else None
        self.play_sf2_btn.setEnabled(bool(self.sf2_path and mido and fluidsynth))
        self.explanation_view.setPlainText(explanation)
        self.play_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def export_audio(self):
        if self.audio_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar WAV", "composition.wav", "WAV (*.wav)")
        if path:
            save_wav(Path(path), self.audio_data)
            QMessageBox.information(self, "Guardado", f"Archivo guardado en {path}")


# ----------------------- Entry Point ----------------------- #

    def play_sf2(self):
        if not (self.sf2_path and mido and fluidsynth):
            QMessageBox.warning(self, "Sin SoundFont", "No se encontró .sf2 o faltan librerías mido/pyfluidsynth.")
            return
        
        # Create temporary files
        tmpmid = Path(tempfile.gettempdir())/"temp_mid.mid"
        tmpwav = Path(tempfile.gettempdir())/"temp_sf2.wav"
        
        # Write MIDI file
        if not _write_midi(tmpmid, self.current_freqs):
            QMessageBox.critical(self,"Error","mido no disponible.")
            return
        
        # Create progress dialog
        progress_dialog = QProgressDialog("Generando audio SF2...", "Cancelar", 0, 100, self)
        progress_dialog.setWindowTitle("Procesando SoundFont")
        progress_dialog.setModal(True)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        
        # Create and configure SF2 worker
        self.sf2_worker = SF2Worker(tmpmid, tmpwav, self.sf2_path)
        
        # Connect signals
        def on_progress(value):
            progress_dialog.setValue(value)
        
        def on_status(message):
            progress_dialog.setLabelText(message)
        
        def on_finished(success):
            progress_dialog.close()
            if success:
                try:
                    # Load wav and play
                    with wave.open(str(tmpwav), 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    self.audio_data = audio
                    
                    # Verify the audio is not silent
                    if len(audio) > 0:
                        max_amplitude = np.max(np.abs(audio))
                        if max_amplitude > 100:  # Good audio
                            self.play_audio()
                            QMessageBox.information(self, "Éxito", f"Audio SF2 generado exitosamente!\nNotas: {len(self.current_freqs)}\nDuración: {len(audio)/44100:.1f}s")
                        else:
                            QMessageBox.warning(self, "Audio débil", "El audio SF2 generado es muy débil. Revisa la entrada.")
                    else:
                        QMessageBox.warning(self, "Audio vacío", "El audio SF2 generado está vacío.")
                        
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al cargar el audio generado: {e}")
            else:
                QMessageBox.critical(self, "Error", "Error al generar audio SF2. Revisa la consola para más detalles.")
            
            # Clean up worker
            if hasattr(self, 'sf2_worker') and self.sf2_worker:
                self.sf2_worker.deleteLater()
                self.sf2_worker = None
        
        self.sf2_worker.progress.connect(on_progress)
        self.sf2_worker.status.connect(on_status)
        self.sf2_worker.finished.connect(on_finished)
        
        # Start the worker
        self.sf2_worker.start()
        progress_dialog.exec()

    def closeEvent(self, event):
        if self.play_thread is not None and self.play_thread.isRunning():
            self.play_thread.wait()
        super().closeEvent(event)


# ----------------------- Entry Point ----------------------- #

def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
