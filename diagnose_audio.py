#!/usr/bin/env python3
"""
Diagnostic script for FluidSynth and audio issues
"""

import sys
import os
from pathlib import Path

def check_fluidsynth_installation():
    """Check FluidSynth installation and configuration"""
    print("=== FluidSynth Installation Check ===")
    
    try:
        import fluidsynth
        print("✓ FluidSynth imported successfully")
        
        # Check version
        try:
            version = fluidsynth.__version__
            print(f"✓ Version: {version}")
        except:
            print("? Version unknown")
        
        # Test basic synth creation
        try:
            fs = fluidsynth.Synth()
            print("✓ Basic Synth creation works")
            
            # Test settings
            try:
                fs.setting('audio.driver', 'file')
                print("✓ Settings configuration works")
            except Exception as e:
                print(f"✗ Settings failed: {e}")
            
            # Test start
            try:
                fs.start()
                print("✓ Synth start works")
                fs.delete()
            except Exception as e:
                print(f"✗ Synth start failed: {e}")
                fs.delete()
                
        except Exception as e:
            print(f"✗ Synth creation failed: {e}")
            
    except ImportError as e:
        print(f"✗ FluidSynth import failed: {e}")
        return False
    
    return True

def check_mido_installation():
    """Check Mido installation"""
    print("\n=== Mido Installation Check ===")
    
    try:
        import mido
        print("✓ Mido imported successfully")
        
        # Test MIDI file creation
        try:
            mid = mido.MidiFile()
            print("✓ MIDI file creation works")
        except Exception as e:
            print(f"✗ MIDI file creation failed: {e}")
            
    except ImportError as e:
        print(f"✗ Mido import failed: {e}")
        return False
    
    return True

def check_soundfont():
    """Check if SoundFont file exists"""
    print("\n=== SoundFont Check ===")
    
    sf2_files = list(Path('.').glob('*.sf2'))
    if sf2_files:
        print(f"✓ Found SoundFont files: {[f.name for f in sf2_files]}")
        return True
    else:
        print("✗ No SoundFont files found")
        print("  Place a .sf2 file in the current directory")
        return False

def check_audio_drivers():
    """Check available audio drivers"""
    print("\n=== Audio Driver Check ===")
    
    try:
        import fluidsynth
        fs = fluidsynth.Synth()
        
        # Try different drivers
        drivers = ['file', 'alsa', 'pulseaudio', 'jack', 'portaudio', 'sdl2']
        
        for driver in drivers:
            try:
                fs.setting('audio.driver', driver)
                print(f"✓ Driver '{driver}' available")
            except Exception as e:
                print(f"✗ Driver '{driver}' failed: {e}")
        
        fs.delete()
        
    except Exception as e:
        print(f"✗ Audio driver check failed: {e}")

def check_system_audio():
    """Check system audio capabilities"""
    print("\n=== System Audio Check ===")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Check default output device
        try:
            info = p.get_default_output_device_info()
            print(f"✓ Default output device: {info['name']}")
        except Exception as e:
            print(f"✗ No default output device: {e}")
        
        # List available devices
        print("\nAvailable audio devices:")
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    print(f"  {i}: {info['name']} (outputs: {info['maxOutputChannels']})")
            except:
                pass
        
        p.terminate()
        
    except ImportError:
        print("✗ PyAudio not available")
    except Exception as e:
        print(f"✗ Audio system check failed: {e}")

def provide_solutions():
    """Provide solutions for common issues"""
    print("\n=== Solutions ===")
    
    print("If you're having FluidSynth issues:")
    print("1. Install FluidSynth system libraries:")
    print("   Windows: Download from https://github.com/FluidSynth/fluidsynth/releases")
    print("   Linux: sudo apt-get install libfluidsynth-dev")
    print("   macOS: brew install fluid-synth")
    
    print("\n2. For SDL issues:")
    print("   - The SDL warning is usually harmless for file output")
    print("   - Try using 'file' driver instead of 'sdl2'")
    
    print("\n3. For MIDI device issues:")
    print("   - Set 'midi.driver' to 'none' to disable MIDI input")
    print("   - Use manual MIDI playback instead of sequencer")
    
    print("\n4. Alternative approach:")
    print("   - Use the simplified SF2 rendering function")
    print("   - It focuses on manual MIDI playback which is more reliable")

if __name__ == "__main__":
    print("Audio System Diagnostic Tool")
    print("=" * 40)
    
    fluidsynth_ok = check_fluidsynth_installation()
    mido_ok = check_mido_installation()
    sf2_ok = check_soundfont()
    
    check_audio_drivers()
    check_system_audio()
    
    provide_solutions()
    
    print("\n" + "=" * 40)
    if fluidsynth_ok and mido_ok and sf2_ok:
        print("✓ All basic requirements met!")
        print("  The application should work with the simplified SF2 rendering.")
    else:
        print("✗ Some requirements missing.")
        print("  Check the solutions above and try again.")
        sys.exit(1) 