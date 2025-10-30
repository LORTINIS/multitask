"""
Architecture Registry for Multitask SNNs.

This module provides utilities to discover and load available architectures.
"""

import os
import importlib.util
from pathlib import Path


def get_available_architectures():
    """
    Discover all available architecture modules in the architectures directory.
    
    Returns:
        dict: Dictionary mapping architecture names to module information
              Format: {architecture_name: {'module_name': str, 'description': str}}
    """
    architectures = {}
    arch_dir = Path(__file__).parent
    
    # Find all Python files except __init__.py
    for file_path in arch_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        
        module_name = file_path.stem
        
        # Try to load the module to get its description
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get architecture name from the module's MultitaskSNN class
            if hasattr(module, 'MultitaskSNN') and hasattr(module.MultitaskSNN, 'ARCHITECTURE_NAME'):
                arch_name = module.MultitaskSNN.ARCHITECTURE_NAME
            else:
                arch_name = module_name
            
            # Get description from docstring
            description = module.__doc__.split('\n')[0] if module.__doc__ else "No description"
            
            architectures[arch_name] = {
                'module_name': module_name,
                'description': description,
                'file_path': str(file_path)
            }
        except Exception as e:
            print(f"Warning: Could not load architecture from {file_path.name}: {e}")
    
    return architectures


def load_architecture(architecture_name):
    """
    Load a specific architecture module by name.
    
    Args:
        architecture_name: Name of the architecture (from ARCHITECTURE_NAME constant)
    
    Returns:
        module: The loaded architecture module containing MultitaskSNN class
    
    Raises:
        ValueError: If architecture not found
    """
    available = get_available_architectures()
    
    if architecture_name not in available:
        raise ValueError(
            f"Architecture '{architecture_name}' not found. "
            f"Available architectures: {list(available.keys())}"
        )
    
    module_name = available[architecture_name]['module_name']
    file_path = available[architecture_name]['file_path']
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def print_available_architectures():
    """Print a formatted list of all available architectures."""
    architectures = get_available_architectures()
    
    print("\n" + "="*80)
    print("AVAILABLE ARCHITECTURES")
    print("="*80)
    
    if not architectures:
        print("No architectures found in the architectures directory.")
        return
    
    for i, (arch_name, info) in enumerate(architectures.items(), 1):
        print(f"\n{i}. {arch_name}")
        print(f"   Module: {info['module_name']}.py")
        print(f"   Description: {info['description']}")
    
    print("\n" + "="*80)
