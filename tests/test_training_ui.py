#!/usr/bin/env python3
"""
Test the training UI components.
"""
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def test_progress_bar_callback():
    """Test that ProgressBarCallback can be imported and initialized."""
    print("=== Testing ProgressBarCallback ===")
    
    # Import here to avoid TensorFlow requirement at module level
    try:
        from src.training.ui import ProgressBarCallback
        # Test initialization
        callback = ProgressBarCallback(epochs=5, desc="Test Training")
        assert callback.epochs == 5
        assert callback.desc == "Test Training"
        assert callback.best_val_loss == float('inf')
        print("✅ ProgressBarCallback initialization works!")
    except ImportError as e:
        print(f"⚠️  ProgressBarCallback test skipped (TensorFlow not available): {e}")
        return


def test_print_training_summary():
    """Test the training summary function with mock data."""
    print("\n=== Testing print_training_summary ===")
    
    try:
        from src.training.ui import print_training_summary
        
        # Create mock training results
        mock_results = {
            "dense_model": {
                "history": {
                    "val_loss": [0.8, 0.6, 0.5, 0.4],
                    "val_accuracy": [0.7, 0.8, 0.85, 0.9]
                },
                "training_time": 123.45
            },
            "deepsets_model": {
                "history": {
                    "val_loss": [0.9, 0.7, 0.55, 0.45],
                    "val_accuracy": [0.65, 0.75, 0.82, 0.88]
                },
                "training_time": 234.56
            }
        }
        
        # Test the function
        print_training_summary(mock_results, "TEST SUMMARY")
        
        print("✅ print_training_summary works!")
    except ImportError as e:
        print(f"⚠️  print_training_summary test skipped: {e}")


def test_import():
    """Test that we can import everything correctly."""
    print("\n=== Testing Imports ===")
    
    try:
        from src.training.ui import print_training_summary
        print("✅ Basic imports work correctly!")
        try:
            from src.training.ui import ProgressBarCallback
            print("✅ TensorFlow-dependent imports work too!")
        except ImportError:
            print("⚠️  TensorFlow-dependent imports skipped (TensorFlow not available)")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise


if __name__ == "__main__":
    print("🧪 Testing Training UI Components")
    print("=" * 50)
    
    try:
        test_import()
        test_progress_bar_callback()
        test_print_training_summary()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TRAINING UI TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
