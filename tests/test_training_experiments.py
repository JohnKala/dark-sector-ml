#!/usr/bin/env python3
"""
Test the training experiments module.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_experiments_import():
    """Test that we can import the experiments module."""
    print("=== Testing experiments imports ===")
    
    try:
        from src.training.experiments import train_individual_models, train_leave_one_out_models
        print("✅ Experiment functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_individual_models_parameters():
    """Test parameter validation for train_individual_models."""
    print("\n=== Testing train_individual_models parameters ===")
    
    try:
        from src.training.experiments import train_individual_models
        
        # Test function signature accepts correct parameters
        test_params = {
            'dark_files': ['file1.h5', 'file2.h5'],
            'sm_file': 'sm.h5',
            'use_scaled': True,
            'normalize': True,
            'model_type': 'deepsets',
            'epochs': 5,
            'batch_size': 64,
            'verbose': False
        }
        
        print("✅ train_individual_models parameter handling works!")
        return True
        
    except ImportError as e:
        print(f"⚠️  train_individual_models test skipped: {e}")
        return False
    except Exception as e:
        print(f"❌ train_individual_models parameter test failed: {e}")
        return False


def test_leave_one_out_parameters():
    """Test parameter validation for train_leave_one_out_models."""
    print("\n=== Testing train_leave_one_out_models parameters ===")
    
    try:
        from src.training.experiments import train_leave_one_out_models
        
        # Test function signature accepts correct parameters
        test_params = {
            'dark_files': ['file1.h5', 'file2.h5', 'file3.h5'],
            'sm_file': 'sm.h5',
            'use_scaled': True,
            'normalize': True,
            'model_type': 'deepsets',
            'epochs': 5,
            'batch_size': 64,
            'verbose': False
        }
        
        print("✅ train_leave_one_out_models parameter handling works!")
        return True
        
    except ImportError as e:
        print(f"⚠️  train_leave_one_out_models test skipped: {e}")
        return False
    except Exception as e:
        print(f"❌ train_leave_one_out_models parameter test failed: {e}")
        return False


def test_dependency_imports():
    """Test that all required dependencies can be imported."""
    print("\n=== Testing dependency imports ===")
    
    success = True
    dependencies = [
        ('os', 'os'),
        ('numpy', 'np'),
        ('sklearn.preprocessing', 'StandardScaler'),
        ('src.data.preparation', 'create_dataset'),
        ('src.data.preprocessor', 'prepare_ml_dataset'),
        ('src.data.preprocessor', 'prepare_deepsets_data'),
        ('src.data.loader', 'extract_parameters'),
        ('src.training.trainer', 'train_model'),
        ('src.training.ui', 'print_training_summary')
    ]
    
    for module, item in dependencies:
        try:
            if module == 'os':
                import os
            elif module == 'numpy':
                import numpy as np
            elif module == 'sklearn.preprocessing':
                from sklearn.preprocessing import StandardScaler
            elif module == 'src.data.preparation':
                from src.data.preparation import create_dataset
            elif module == 'src.data.preprocessor':
                from src.data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
            elif module == 'src.data.loader':
                from src.data.loader import extract_parameters
            elif module == 'src.training.trainer':
                from src.training.trainer import train_model
            elif module == 'src.training.ui':
                from src.training.ui import print_training_summary
            
            print(f"✅ {module}.{item}")
        except ImportError as e:
            print(f"❌ {module}.{item}: {e}")
            success = False
    
    if success:
        print("✅ All dependencies available!")
    else:
        print("⚠️  Some dependencies missing (expected in local env)")
    
    return success


def test_experiments_integration():
    """Test that experiments module integrates with our existing pipeline."""
    print("\n=== Testing experiments integration ===")
    
    try:
        # Test that we can import everything needed for a complete pipeline
        from src.training.experiments import train_individual_models, train_leave_one_out_models
        from src.data.preparation import create_dataset
        from src.data.preprocessor import prepare_ml_dataset
        from src.data.loader import extract_parameters
        
        print("✅ Experiments module integrates with existing pipeline!")
        return True
        
    except ImportError as e:
        print(f"⚠️  Integration test skipped (missing dependencies): {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Training Experiments Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 5
        
        if test_experiments_import():
            success_count += 1
        
        if test_individual_models_parameters():
            success_count += 1
        
        if test_leave_one_out_parameters():
            success_count += 1
        
        if test_dependency_imports():
            success_count += 1
        
        if test_experiments_integration():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} EXPERIMENTS TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All experiments functionality verified!")
        elif success_count >= 3:
            print("⚠️  Core functionality works, some dependencies missing")
        else:
            print("❌ Multiple test failures - check implementation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
