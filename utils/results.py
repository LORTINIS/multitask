import os
import json
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt


def save_history_json(history: Dict[str, List[Any]], filepath: str) -> None:
    """Save training history dictionary to a JSON file.

    Args:
        history: dictionary of lists (losses, metrics, scales, ...)
        filepath: target path for JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert any non-serializable items (e.g., numpy floats/arrays) to python types
    serializable = {}
    for k, v in history.items():
        try:
            # try to convert list-like
            serializable[k] = [float(x) if (hasattr(x, '__float__') and not isinstance(x, (str,))) else x for x in v]
        except Exception:
            serializable[k] = v

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_history_json(filepath: str) -> Dict[str, List[Any]]:
    """Load a history JSON saved by save_history_json."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_history(history: Dict[str, List[Any]], save_dir: str) -> List[str]:
    """Generate and save a set of diagnostic plots from training history.

    Returns list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    def _save_fig(fig, name):
        path = os.path.join(save_dir, name)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)

    # Total loss
    if 'total_loss' in history:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history['total_loss'], label='total_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.grid(True)
        _save_fig(fig, 'loss_total.png')

    # Class & Reg losses
    if 'class_loss' in history and 'reg_loss' in history:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history['class_loss'], label='class_loss')
        plt.plot(history['reg_loss'], label='reg_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Class and Reg Loss')
        plt.legend()
        plt.grid(True)
        _save_fig(fig, 'loss_class_reg.png')

    # Scaled losses
    if 'scaled_class_loss' in history and 'scaled_reg_loss' in history:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history['scaled_class_loss'], label='scaled_class_loss')
        plt.plot(history['scaled_reg_loss'], label='scaled_reg_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Scaled Loss')
        plt.title('Scaled Losses')
        plt.legend()
        plt.grid(True)
        _save_fig(fig, 'loss_scaled.png')

    # Scales over time
    if 'class_scale' in history and 'reg_scale' in history:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history['class_scale'], label='class_scale')
        plt.plot(history['reg_scale'], label='reg_scale')
        plt.xlabel('Epoch')
        plt.ylabel('Scale')
        plt.title('Loss Scales')
        plt.legend()
        plt.grid(True)
        _save_fig(fig, 'loss_scales.png')

    # Accuracy and R2
    if 'eval_accuracy' in history or 'eval_r2' in history:
        fig = plt.figure(figsize=(6, 4))
        if 'eval_accuracy' in history and len(history['eval_accuracy']) > 0:
            plt.plot(history['eval_accuracy'], label='accuracy')
        if 'eval_r2' in history and len(history['eval_r2']) > 0:
            plt.plot(history['eval_r2'], label='r2')
        plt.xlabel('Evaluation Step (every N epochs)')
        plt.ylabel('Metric')
        plt.title('Accuracy / R2 over time')
        plt.legend()
        plt.grid(True)
        _save_fig(fig, 'accuracy_r2.png')

    return saved_files
