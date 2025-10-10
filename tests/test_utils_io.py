#!/usr/bin/env python3
"""
Test the utils.io module.
"""
import sys
from pathlib import Path
import numpy as np
import json
import pickle
import os
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_io_imports():
    """Test that we can import I/O functions."""
    print("=== Testing I/O imports ===")
    
    try:
        from src.utils.io import save_results
        print("✅ I/O functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_save_results_json_only():
    """Test save_results with JSON output only."""
    print("\n=== Testing save_results JSON only ===")
    
    try:
        from src.utils.io import save_results
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock evaluation results
            mock_results = {
                'model_A': {
                    'metrics': {
                        'roc_auc': 0.85,
                        'precision': 0.78,
                        'recall': 0.82,
                        'f1': 0.80
                    },
                    'roc_curve': {
                        'fpr': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                        'tpr': np.array([0.0, 0.3, 0.6, 0.7, 0.9, 1.0])
                    },
                    'predictions': {
                        'y_true': np.array([0, 1, 1, 0, 1]),
                        'y_pred_proba': np.array([0.1, 0.8, 0.9, 0.2, 0.7])
                    }
                },
                'model_B': {
                    'metrics': {
                        'roc_auc': 0.92,
                        'precision': 0.88,
                        'recall': 0.86,
                        'f1': 0.87
                    }
                }
            }
            
            # Test saving (JSON only)
            filename = os.path.join(temp_dir, 'test_results')
            save_results(mock_results, filename, save_models=False)
            
            # Check that JSON file was created
            json_file = f"{filename}.json"
            assert os.path.exists(json_file), "JSON file should exist"
            
            # Check that pickle file was NOT created
            pkl_file = f"{filename}.pkl"
            assert not os.path.exists(pkl_file), "Pickle file should not exist when save_models=False"
            
            # Load and verify JSON content
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
            
            # Check structure
            assert 'model_A' in loaded_data
            assert 'model_B' in loaded_data
            
            # Check that numpy arrays were converted to lists
            assert isinstance(loaded_data['model_A']['roc_curve']['fpr'], list)
            assert isinstance(loaded_data['model_A']['predictions']['y_true'], list)
            
            # Check values
            assert loaded_data['model_A']['metrics']['roc_auc'] == 0.85
            assert loaded_data['model_B']['metrics']['roc_auc'] == 0.92
            
        print("✅ JSON-only saving works!")
        return True
        
    except Exception as e:
        print(f"❌ save_results JSON test failed: {e}")
        return False


def test_save_results_with_models():
    """Test save_results with pickle output for models."""
    print("\n=== Testing save_results with models ===")
    
    try:
        from src.utils.io import save_results
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use simple objects instead of classes for pickle compatibility
            mock_model_1 = {
                'name': 'test_model_1',
                'weights': np.random.rand(10, 5),
                'type': 'mock_model'
            }
            
            mock_model_2 = {
                'name': 'test_model_2',
                'weights': np.random.rand(8, 3),
                'type': 'mock_model'
            }
            
            # Create mock results with model objects
            mock_results = {
                'experiment_1': {
                    'model': mock_model_1,
                    'metrics': {'roc_auc': 0.88},
                    'params': {'model_type': 'dense', 'epochs': 50}
                },
                'experiment_2': {
                    'model': mock_model_2,  
                    'metrics': {'roc_auc': 0.91},
                    'params': {'model_type': 'deepsets', 'epochs': 100}
                }
            }
            
            # Test saving with models
            filename = os.path.join(temp_dir, 'test_results_with_models')
            save_results(mock_results, filename, save_models=True)
            
            # Check that both files were created
            json_file = f"{filename}.json"
            pkl_file = f"{filename}.pkl"
            assert os.path.exists(json_file), "JSON file should exist"
            assert os.path.exists(pkl_file), "Pickle file should exist when save_models=True"
            
            # Load and verify JSON content (should exclude models)
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            # Check that model objects were excluded from JSON
            assert 'model' not in json_data['experiment_1']
            assert 'model' not in json_data['experiment_2']
            
            # But other data should be present
            assert json_data['experiment_1']['metrics']['roc_auc'] == 0.88
            assert json_data['experiment_2']['params']['model_type'] == 'deepsets'
            
            # Load and verify pickle content (should include models)
            with open(pkl_file, 'rb') as f:
                pkl_data = pickle.load(f)
            
            # Check that model objects are present in pickle
            assert 'model' in pkl_data['experiment_1']
            assert 'model' in pkl_data['experiment_2']
            assert isinstance(pkl_data['experiment_1']['model'], dict)
            assert pkl_data['experiment_1']['model']['name'] == 'test_model_1'
            assert pkl_data['experiment_2']['model']['name'] == 'test_model_2'
            
        print("✅ Saving with models works!")
        return True
        
    except Exception as e:
        print(f"❌ save_results with models test failed: {e}")
        return False


def test_save_results_directory_creation():
    """Test that save_results creates directories when needed."""
    print("\n=== Testing directory creation ===")
    
    try:
        from src.utils.io import save_results
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory path that doesn't exist
            nested_path = os.path.join(temp_dir, 'results', 'experiments', 'run_1')
            filename = os.path.join(nested_path, 'test_results')
            
            # Directory shouldn't exist initially
            assert not os.path.exists(nested_path)
            
            # Save results - should create directory
            mock_results = {'test': {'metric': 0.5}}
            save_results(mock_results, filename)
            
            # Check directory was created
            assert os.path.exists(nested_path), "Directory should be created"
            assert os.path.exists(f"{filename}.json"), "JSON file should be created"
            
        print("✅ Directory creation works!")
        return True
        
    except Exception as e:
        print(f"❌ Directory creation test failed: {e}")
        return False


def test_save_results_complex_objects():
    """Test save_results with complex objects and edge cases."""
    print("\n=== Testing complex object serialization ===")
    
    try:
        from src.utils.io import save_results
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock complex object
            class ComplexObject:
                def __init__(self):
                    self.value = 42
                    self.nested = {'array': np.array([1, 2, 3])}
                
                def __str__(self):
                    return f"ComplexObject(value={self.value})"
            
            # Create results with various object types
            mock_results = {
                'arrays': {
                    'numpy_1d': np.array([1, 2, 3, 4, 5]),
                    'numpy_2d': np.array([[1, 2], [3, 4]]),
                    'numpy_float': np.array([1.1, 2.2, 3.3])
                },
                'lists': {
                    'simple_list': [1, 2, 3],
                    'nested_list': [[1, 2], [3, 4]],
                    'mixed_list': [1, 'string', 3.14, np.array([1, 2])]
                },
                'objects': {
                    'complex_obj': ComplexObject()
                }
            }
            
            # Test saving
            filename = os.path.join(temp_dir, 'complex_results')
            save_results(mock_results, filename)
            
            # Load and verify JSON
            with open(f"{filename}.json", 'r') as f:
                json_data = json.load(f)
            
            # Check numpy arrays converted to lists
            assert isinstance(json_data['arrays']['numpy_1d'], list)
            assert json_data['arrays']['numpy_1d'] == [1, 2, 3, 4, 5]
            assert isinstance(json_data['arrays']['numpy_2d'], list)
            assert json_data['arrays']['numpy_2d'] == [[1, 2], [3, 4]]
            
            # Check nested structures preserved
            assert json_data['lists']['simple_list'] == [1, 2, 3]
            assert json_data['lists']['nested_list'] == [[1, 2], [3, 4]]
            
            # Check complex object handling
            assert '_object_type' in json_data['objects']['complex_obj']
            assert json_data['objects']['complex_obj']['_object_type'] == 'ComplexObject'
            
        print("✅ Complex object serialization works!")
        return True
        
    except Exception as e:
        print(f"❌ Complex object test failed: {e}")
        return False


def test_utils_integration():
    """Test integration with utils module."""
    print("\n=== Testing utils integration ===")
    
    try:
        # Test import from main utils module
        from src.utils import save_results
        from src.utils.io import save_results as direct_import
        
        assert save_results is direct_import
        
        print("✅ Utils module integration works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Utils I/O Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 6
        
        if test_io_imports():
            success_count += 1
        
        if test_save_results_json_only():
            success_count += 1
        
        if test_save_results_with_models():
            success_count += 1
        
        if test_save_results_directory_creation():
            success_count += 1
        
        if test_save_results_complex_objects():
            success_count += 1
        
        if test_utils_integration():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} UTILS I/O TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All I/O functionality verified!")
        else:
            print("❌ Some I/O tests failed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
