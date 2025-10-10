#!/usr/bin/env python3
"""
Test the evaluation metrics module.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_metrics_imports():
    """Test that we can import evaluation metrics functions."""
    print("=== Testing metrics imports ===")
    
    try:
        from src.evaluation.metrics import evaluate_model, evaluate_all_models
        print("✅ Evaluation metrics functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_evaluate_model_mock():
    """Test evaluate_model function with mock data."""
    print("\n=== Testing evaluate_model with mock data ===")
    
    try:
        from src.evaluation.metrics import evaluate_model
        
        # Create mock test data
        n_samples = 100
        n_features = 10
        
        mock_test_data = {
            'test': {
                'features': np.random.rand(n_samples, n_features),
                'labels': np.random.randint(0, 2, n_samples),
                'attention_mask': np.ones((n_samples,), dtype=bool)  # For DeepSets
            }
        }
        
        # Create mock model (function that returns random predictions)
        class MockModel:
            def predict(self, inputs, verbose=0):
                if isinstance(inputs, list):
                    # DeepSets style input
                    return np.random.rand(inputs[0].shape[0], 1)
                else:
                    # Dense style input
                    return np.random.rand(inputs.shape[0], 1)
        
        mock_model = MockModel()
        
        # Test with dense model
        results_dense = evaluate_model(
            model=mock_model,
            test_data=mock_test_data,
            model_type='dense',
            verbose=False
        )
        
        # Check results structure
        assert 'roc_curve' in results_dense
        assert 'pr_curve' in results_dense
        assert 'metrics' in results_dense
        assert 'confusion_matrix' in results_dense
        assert 'predictions' in results_dense
        
        # Check metrics
        assert 'roc_auc' in results_dense['metrics']
        assert 'precision' in results_dense['metrics']
        assert 'recall' in results_dense['metrics']
        assert 'f1' in results_dense['metrics']
        
        print("✅ Dense model evaluation works!")
        
        # Test with deepsets model
        results_deepsets = evaluate_model(
            model=mock_model,
            test_data=mock_test_data,
            model_type='deepsets',
            verbose=False
        )
        
        assert 'roc_curve' in results_deepsets
        assert 'metrics' in results_deepsets
        print("✅ DeepSets model evaluation works!")
        
        return True
        
    except Exception as e:
        print(f"❌ evaluate_model test failed: {e}")
        return False


def test_evaluate_all_models_mock():
    """Test evaluate_all_models function with mock data."""
    print("\n=== Testing evaluate_all_models with mock data ===")
    
    try:
        from src.evaluation.metrics import evaluate_all_models
        
        # Create mock trained models
        class MockModel:
            def predict(self, inputs, verbose=0):
                if isinstance(inputs, list):
                    return np.random.rand(inputs[0].shape[0], 1)
                else:
                    return np.random.rand(inputs.shape[0], 1)
        
        n_samples = 50
        n_features = 8
        
        mock_trained_models = {
            'model1': {
                'model': MockModel(),
                'params': {'model_type': 'dense'},
                'test': {
                    'features': np.random.rand(n_samples, n_features),
                    'labels': np.random.randint(0, 2, n_samples),
                    'attention_mask': np.ones((n_samples,), dtype=bool)
                }
            },
            'model2': {
                'model': MockModel(),
                'params': {'model_type': 'deepsets'},
                'left_out_data': {  # Leave-one-out style
                    'test': {
                        'features': np.random.rand(n_samples, n_features),
                        'labels': np.random.randint(0, 2, n_samples),
                        'attention_mask': np.ones((n_samples,), dtype=bool)
                    }
                }
            }
        }
        
        # Test evaluation
        results = evaluate_all_models(
            trained_models=mock_trained_models,
            verbose=False
        )
        
        # Check results structure
        assert 'model1' in results
        assert 'model2' in results
        
        # Check each model has evaluation results
        for model_name in ['model1', 'model2']:
            assert 'metrics' in results[model_name]
            assert 'roc_auc' in results[model_name]['metrics']
        
        print("✅ evaluate_all_models works!")
        
        return True
        
    except Exception as e:
        print(f"❌ evaluate_all_models test failed: {e}")
        return False


def test_metrics_parameter_validation():
    """Test parameter validation and error handling."""
    print("\n=== Testing parameter validation ===")
    
    try:
        from src.evaluation.metrics import evaluate_model
        
        # Test threshold parameter
        class MockModel:
            def predict(self, inputs, verbose=0):
                return np.array([[0.8], [0.2], [0.6], [0.3]])  # Fixed predictions
        
        mock_test_data = {
            'test': {
                'features': np.random.rand(4, 5),
                'labels': np.array([1, 0, 1, 0]),
                'attention_mask': np.ones((4,), dtype=bool)
            }
        }
        
        # Test different thresholds
        results_low = evaluate_model(
            MockModel(), mock_test_data, threshold=0.1, verbose=False
        )
        results_high = evaluate_model(
            MockModel(), mock_test_data, threshold=0.9, verbose=False
        )
        
        # Should get different predictions with different thresholds
        assert not np.array_equal(
            results_low['predictions']['y_pred'],
            results_high['predictions']['y_pred']
        )
        
        print("✅ Parameter validation works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation test failed: {e}")
        return False


def test_metrics_integration():
    """Test integration with existing pipeline structures."""
    print("\n=== Testing metrics integration ===")
    
    try:
        from src.evaluation.metrics import evaluate_model, evaluate_all_models
        
        # Test that functions can be imported from main evaluation module
        from src.evaluation import evaluate_model as eval_model_import
        from src.evaluation import evaluate_all_models as eval_all_import
        
        assert eval_model_import is evaluate_model
        assert eval_all_import is evaluate_all_models
        
        print("✅ Module integration works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Evaluation Metrics Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 5
        
        if test_metrics_imports():
            success_count += 1
        
        if test_evaluate_model_mock():
            success_count += 1
        
        if test_evaluate_all_models_mock():
            success_count += 1
        
        if test_metrics_parameter_validation():
            success_count += 1
        
        if test_metrics_integration():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} EVALUATION METRICS TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All evaluation metrics functionality verified!")
        else:
            print("❌ Some evaluation metrics tests failed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
