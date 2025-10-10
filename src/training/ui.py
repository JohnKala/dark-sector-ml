"""
Training UI utilities for progress tracking and results display.
"""

from tqdm import tqdm
from typing import Dict, Any

# Conditional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create a dummy base class for when TF is not available
    class DummyCallback:
        def __init__(self):
            pass


class ProgressBarCallback(tf.keras.callbacks.Callback if TF_AVAILABLE else DummyCallback):
    """
    Callback that displays a clean, organized progress bar during training.
    Note: Requires TensorFlow to be installed.
    """
    def __init__(self, epochs, desc="Training"):
        if TF_AVAILABLE:
            super(ProgressBarCallback, self).__init__()
        else:
            raise ImportError("TensorFlow is required for ProgressBarCallback")
        self.epochs = epochs
        self.desc = desc
        self.progress_bar = None
        self.best_val_loss = float('inf')
        
    def on_train_begin(self, logs=None):
        # Print a header for the training session
        print(f"\n{'-'*20} {self.desc} {'-'*20}")
        self.progress_bar = tqdm(total=self.epochs, desc=f"Epoch 0/{self.epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        # Update progress bar with current epoch
        self.progress_bar.set_description(f"Epoch {epoch+1}/{self.epochs}")
        self.progress_bar.update(1)
        
        # Format metrics for display
        metrics_str = []
        for k, v in logs.items():
            metrics_str.append(f"{k}: {v:.4f}")
        
        # Track best validation loss
        if 'val_loss' in logs and logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            metrics_str.append("✓")  # Mark new best model
            
        self.progress_bar.set_postfix_str(" - ".join(metrics_str))
        
    def on_train_end(self, logs=None):
        self.progress_bar.close()
        # Print summary line with best performance
        print(f"Training complete - Best val_loss: {self.best_val_loss:.4f}")


def print_training_summary(results: Dict[str, Any], header: str = "TRAINING SUMMARY"):
    """
    Print a clear, organized summary of training results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of training results
    header : str
        Header text for the summary
    """
    # Calculate maximum model name length for alignment
    max_name_len = max([len(name) for name in results.keys()])
    
    # Print header
    print(f"\n{'='*20} {header} {'='*20}")
    print(f"Successfully trained {len(results)} models:\n")
    
    # Print table header
    print(f"{'Model Name':<{max_name_len+2}} {'Val Loss':<10} {'Val Acc':<10} {'Train Time':<12}")
    print(f"{'-'*(max_name_len+2)} {'-'*10} {'-'*10} {'-'*12}")
    
    # Print each model's results
    for name, res in results.items():
        val_loss = min(res['history']['val_loss'])
        val_acc = max(res['history']['val_accuracy']) if 'val_accuracy' in res['history'] else max(res['history'].get('val_acc', [0]))
        train_time = res['training_time']
        
        print(f"{name:<{max_name_len+2}} {val_loss:<10.4f} {val_acc:<10.4f} {train_time:<12.2f}s")
    
    print(f"\n{'='*(40 + len(header))}")
