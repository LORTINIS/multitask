"""
Multitask SNN Training Dispatcher.

This script automatically selects and runs the appropriate training script
based on the chosen architecture.
"""

import os
import sys
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architectures import (
    get_available_architectures,
    load_architecture,
    print_available_architectures
)


def main():
    """Main dispatcher - selects architecture and runs appropriate training script."""
    print("="*80)
    print("MULTITASK SNN TRAINING DISPATCHER")
    print("="*80)
    
    # Show available architectures
    print_available_architectures()
    
    available_archs = get_available_architectures()
    
    if not available_archs:
        print("\nERROR: No architectures found in the architectures directory!")
        print("Please add architecture files to the 'architectures' folder.")
        sys.exit(1)
    
    # Let user select architecture
    arch_list = list(available_archs.keys())
    print("\nSelect an architecture to train:")
    for i, arch_name in enumerate(arch_list, 1):
        print(f"  {i}. {arch_name}")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(arch_list)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(arch_list):
                selected_arch_name = arch_list[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(arch_list)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nTraining cancelled by user.")
            sys.exit(0)
    
    print(f"\n✓ Selected architecture: {selected_arch_name}")
    
    # Load the architecture to get the training script
    try:
        arch_module = load_architecture(selected_arch_name)
        
        # Get training script from architecture
        if hasattr(arch_module.MultitaskSNN, 'TRAINING_SCRIPT'):
            training_script = arch_module.MultitaskSNN.TRAINING_SCRIPT
        else:
            print(f"\nWARNING: Architecture '{selected_arch_name}' does not specify a TRAINING_SCRIPT.")
            print("Using default: train_two_shared_layers.py")
            training_script = "train_two_shared_layers.py"
        
        training_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            training_script
        )
        
        if not os.path.exists(training_script_path):
            print(f"\nERROR: Training script not found: {training_script_path}")
            print(f"Architecture '{selected_arch_name}' requires: {training_script}")
            sys.exit(1)
        
        print(f"✓ Training script: {training_script}")
        print("\n" + "="*80)
        print(f"LAUNCHING TRAINING: {training_script}")
        print("="*80 + "\n")
        
        # Run the appropriate training script with the selected architecture
        result = subprocess.run(
            [sys.executable, training_script_path, selected_arch_name],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"\nERROR: Failed to load architecture or start training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
