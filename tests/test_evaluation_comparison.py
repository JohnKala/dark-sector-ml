#!/usr/bin/env python3
"""
Test the evaluation comparison module.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_comparison_imports():
    """Test that we can import comparison functions."""
    print("=== Testing comparison imports ===")
    
    try:
        from src.evaluation.comparison import evaluate_cross_model_performance
        print("✅ Comparison functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_cross_model_evaluation_mock():
    """Test evaluate_cross_model_performance with mock data."""
    print("\n=== Testing cross-model evaluation with mock data ===")
    
    try:
        from src.evaluation.comparison import evaluate_cross_model_performance
        
        # Create mock models
        class MockModel:
            def __init__(self, base_score=0.7):
                self.base_score = base_score
            
            def predict(self, inputs, verbose=0):
                if isinstance(inputs, list):
                    # DeepSets style - return predictions based on first input
                    n_samples = inputs[0].shape[0]
                else:
                    # Dense style
                    n_samples = inputs.shape[0]
                # Return deterministic predictions based on base_score
                return np.full((n_samples, 1), self.base_score)
        
        # Create mock trained models
        all_models = {
            'model_mDark-1': {
                'model': MockModel(base_score=0.8),
                'params': {'model_type': 'dense'}
            },
            'model_mDark-5': {
                'model': MockModel(base_score=0.6),
                'params': {'model_type': 'deepsets'}
            }
        }
        
        # Create mock test datasets
        n_samples = 50
        n_features = 8
        
        all_test_datasets = {
            'dataset_mDark-1': {
                'test': {
                    'features': np.random.rand(n_samples, n_features),
                    'labels': np.random.randint(0, 2, n_samples),
                    'attention_mask': np.ones((n_samples,), dtype=bool)
                }
            },
            'dataset_mDark-5': {
                'test': {
                    'features': np.random.rand(n_samples, n_features),
                    'labels': np.random.randint(0, 2, n_samples),
                    'attention_mask': np.ones((n_samples,), dtype=bool)
                }
            }
        }
        
        # Test cross-model evaluation
        results = evaluate_cross_model_performance(
            all_models=all_models,
            all_test_datasets=all_test_datasets,
            verbose=False
        )
        
        # Check results structure
        assert 'model_mDark-1' in results
        assert 'model_mDark-5' in results
        
        # Check each model evaluated on each dataset
        for model_name in ['model_mDark-1', 'model_mDark-5']:
            assert 'dataset_mDark-1' in results[model_name]
            assert 'dataset_mDark-5' in results[model_name]
            
            # Check metrics for each model-dataset combination
            for ds_name in ['dataset_mDark-1', 'dataset_mDark-5']:
                metrics = results[model_name][ds_name]
                assert 'roc_auc' in metrics
                assert 'precision' in metrics
                assert 'recall' in metrics
                assert 'f1' in metrics
                assert 'y_pred_proba' in metrics
                assert 'y_true' in metrics
                
                # Check that metrics are reasonable values
                assert 0 <= metrics['roc_auc'] <= 1
                assert 0 <= metrics['precision'] <= 1
                assert 0 <= metrics['recall'] <= 1
                assert 0 <= metrics['f1'] <= 1
        
        print("✅ Cross-model evaluation structure works!")
        print(f"✅ Evaluated {len(all_models)} models on {len(all_test_datasets)} datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-model evaluation test failed: {e}")
        return False


def test_cross_model_different_model_types():
    """Test cross-model evaluation with different model types."""
    print("\n=== Testing cross-model evaluation with different model types ===")
    
    try:
        from src.evaluation.comparison import evaluate_cross_model_performance
        
        class MockModel:
            def predict(self, inputs, verbose=0):
                if isinstance(inputs, list):
                    return np.random.rand(inputs[0].shape[0], 1)
                else:
                    return np.random.rand(inputs.shape[0], 1)
        
        # Mix of dense and deepsets models
        all_models = {
            'dense_model': {
                'model': MockModel(),
                'params': {'model_type': 'dense'}
            },
            'deepsets_model': {
                'model': MockModel(),
                'params': {'model_type': 'deepsets'}
            }
        }
        
        # Single test dataset
        all_test_datasets = {
            'test_dataset': {
                'test': {
                    'features': np.random.rand(30, 5),
                    'labels': np.random.randint(0, 2, 30),
                    'attention_mask': np.ones((30,), dtype=bool)
                }
            }
        }
        
        # Test evaluation
        results = evaluate_cross_model_performance(
            all_models=all_models,
            all_test_datasets=all_test_datasets,
            verbose=False
        )
        
        # Check that both model types were evaluated
        assert 'dense_model' in results
        assert 'deepsets_model' in results
        
        # Both should have results for the test dataset
        assert 'test_dataset' in results['dense_model']
        assert 'test_dataset' in results['deepsets_model']
        
        print("✅ Mixed model types evaluation works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed model types test failed: {e}")
        return False


def test_cross_model_threshold_parameter():
    """Test that threshold parameter affects results."""
    print("\n=== Testing cross-model threshold parameter ===")
    
    try:
        from src.evaluation.comparison import evaluate_cross_model_performance
        
        class MockModel:
            def predict(self, inputs, verbose=0):
                # Return fixed probabilities for threshold testing
                if isinstance(inputs, list):
                    n_samples = inputs[0].shape[0]
                else:
                    n_samples = inputs.shape[0]
                return np.full((n_samples, 1), 0.6)  # Fixed probability
        
        all_models = {
            'test_model': {
                'model': MockModel(),
                'params': {'model_type': 'dense'}
            }
        }
        
        all_test_datasets = {
            'test_dataset': {
                'test': {
                    'features': np.random.rand(20, 3),
                    'labels': np.ones(20),  # All positive labels
                    'attention_mask': np.ones((20,), dtype=bool)
                }
            }
        }
        
        # Test with different thresholds
        results_low = evaluate_cross_model_performance(
            all_models, all_test_datasets, threshold=0.3, verbose=False
        )
        results_high = evaluate_cross_model_performance(
            all_models, all_test_datasets, threshold=0.9, verbose=False
        )
        
        # With fixed probability 0.6 and all positive labels:
        # Low threshold (0.3): should predict all positive -> good recall, precision
        # High threshold (0.9): should predict all negative -> poor recall
        
        recall_low = results_low['test_model']['test_dataset']['recall']
        recall_high = results_high['test_model']['test_dataset']['recall']
        
        # Low threshold should give higher recall than high threshold
        assert recall_low >= recall_high
        
        print("✅ Threshold parameter works correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Threshold parameter test failed: {e}")
        return False


def test_cross_model_integration():
    """Test integration with evaluation module."""
    print("\n=== Testing cross-model integration ===")
    
    try:
        # Test import from main evaluation module
        from src.evaluation import evaluate_cross_model_performance
        from src.evaluation.comparison import evaluate_cross_model_performance as direct_import
        
        assert evaluate_cross_model_performance is direct_import
        
        print("✅ Module integration works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Evaluation Comparison Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 5
        
        if test_comparison_imports():
            success_count += 1
        
        if test_cross_model_evaluation_mock():
            success_count += 1
        
        if test_cross_model_different_model_types():
            success_count += 1
        
        if test_cross_model_threshold_parameter():
            success_count += 1
        
        if test_cross_model_integration():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} COMPARISON TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All cross-model evaluation functionality verified!")
        else:
            print("❌ Some cross-model evaluation tests failed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
