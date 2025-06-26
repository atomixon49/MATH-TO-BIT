#!/usr/bin/env python3
"""
Test script to verify FluidSynth compatibility
"""

import sys
from pathlib import Path

def test_fluidsynth_import():
    """Test if FluidSynth can be imported and basic functionality works"""
    try:
        import fluidsynth
        print("✓ FluidSynth imported successfully")
        
        # Test basic Synth creation
        fs = fluidsynth.Synth()
        print("✓ Synth created successfully")
        
        # Test available attributes
        print(f"Available attributes: {[attr for attr in dir(fluidsynth) if not attr.startswith('_')]}")
        
        # Test if Sequencer is available
        if hasattr(fluidsynth, 'Sequencer'):
            print("✓ Sequencer is available (new API)")
            seq = fluidsynth.Sequencer(fs)
            print("✓ Sequencer created successfully")
        else:
            print("✗ Sequencer not available")
            
        # Test if Player is available
        if hasattr(fluidsynth, 'Player'):
            print("✓ Player is available (old API)")
        else:
            print("✗ Player not available (expected in new API)")
            
        fs.delete()
        return True
        
    except ImportError as e:
        print(f"✗ FluidSynth import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ FluidSynth test failed: {e}")
        return False

def test_mido_import():
    """Test if mido can be imported"""
    try:
        import mido
        print("✓ Mido imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Mido import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing FluidSynth compatibility...")
    print("=" * 40)
    
    fluidsynth_ok = test_fluidsynth_import()
    print()
    mido_ok = test_mido_import()
    
    print("=" * 40)
    if fluidsynth_ok and mido_ok:
        print("✓ All tests passed! The application should work correctly.")
    else:
        print("✗ Some tests failed. Please check the installation.")
        sys.exit(1) 