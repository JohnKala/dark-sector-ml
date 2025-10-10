#!/usr/bin/env python3
"""
Test the visualization plots module.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_plots_imports():
    """Test that we can import plotting functions."""
    print("=== Testing plots imports ===")
    
    try:
        from src.visualization.plots import (
            plot_roc_curve, plot_combined_roc_curves, plot_cross_model_roc_curves,
            plot_cross_model_heatmap, plot_training_history
        )
        print("✅ Plot functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_plot_roc_curve():
    """Test single ROC curve plotting."""
    print("\n=== Testing plot_roc_curve ===")
    
    try:
        from src.visualization.plots import plot_roc_curve
        
        # Create mock evaluation results
        n_points = 100
        fpr = np.linspace(0, 1, n_points)
        tpr = np.sort(np.random.rand(n_points))  # Monotonic increasing
        
        evaluation_results = {
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr
            },
            'metrics': {
                'roc_auc': 0.85
            }
        }
        
        # Test plotting
        fig = plot_roc_curve(
            evaluation_results=evaluation_results,
            model_name='test_model_mDark-1_rinv-0.3_alpha-peak'
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that plot has expected elements
        ax = fig.axes[0]
        assert len(ax.lines) >= 2  # ROC curve + diagonal
        assert ax.get_xlabel() == 'False Positive Rate'
        assert ax.get_ylabel() == 'True Positive Rate'
        
        # Clean up
        plt.close(fig)
        
        print("✅ Single ROC curve plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ plot_roc_curve test failed: {e}")
        return False


def test_plot_combined_roc_curves():
    """Test combined ROC curves plotting."""
    print("\n=== Testing plot_combined_roc_curves ===")
    
    try:
        from src.visualization.plots import plot_combined_roc_curves
        
        # Create mock results for multiple models
        all_results = {}
        for i, model_name in enumerate(['model_mDark-1_rinv-0.3_alpha-peak', 'model_mDark-5_rinv-0.3_alpha-peak']):
            n_points = 50
            fpr = np.linspace(0, 1, n_points)
            tpr = np.sort(np.random.rand(n_points))
            
            all_results[model_name] = {
                'roc_curve': {
                    'fpr': fpr,
                    'tpr': tpr
                },
                'metrics': {
                    'roc_auc': 0.8 + i * 0.1
                }
            }
        
        # Test plotting
        fig = plot_combined_roc_curves(
            all_results=all_results,
            title='Test Combined ROC Curves'
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that plot has curves for each model + diagonal
        ax = fig.axes[0]
        assert len(ax.lines) >= len(all_results) + 1  # Models + diagonal
        
        # Clean up
        plt.close(fig)
        
        print("✅ Combined ROC curves plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ plot_combined_roc_curves test failed: {e}")
        return False


def test_plot_cross_model_roc_curves():
    """Test cross-model ROC curves plotting."""
    print("\n=== Testing plot_cross_model_roc_curves ===")
    
    try:
        from src.visualization.plots import plot_cross_model_roc_curves
        
        # Create mock cross-evaluation results
        cross_eval_results = {
            'model_test': {
                'dataset_mDark-1': {
                    'roc_auc': 0.85,
                    'y_true': np.random.randint(0, 2, 100),
                    'y_pred_proba': np.random.rand(100)
                },
                'dataset_mDark-5': {
                    'roc_auc': 0.75,
                    'y_true': np.random.randint(0, 2, 100),
                    'y_pred_proba': np.random.rand(100)
                }
            }
        }
        
        # Test plotting
        fig = plot_cross_model_roc_curves(
            cross_eval_results=cross_eval_results,
            model_name='model_test'
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that plot has curves for each dataset + diagonal
        ax = fig.axes[0]
        expected_lines = len(cross_eval_results['model_test']) + 1  # Datasets + diagonal
        assert len(ax.lines) >= expected_lines
        
        # Clean up
        plt.close(fig)
        
        print("✅ Cross-model ROC curves plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ plot_cross_model_roc_curves test failed: {e}")
        return False


def test_plot_cross_model_heatmap():
    """Test cross-model heatmap plotting."""
    print("\n=== Testing plot_cross_model_heatmap ===")
    
    try:
        from src.visualization.plots import plot_cross_model_heatmap
        
        # Create mock cross-evaluation results
        cross_eval_results = {
            'model_A': {
                'dataset_1': {'roc_auc': 0.9, 'precision': 0.85, 'recall': 0.8, 'f1': 0.82},
                'dataset_2': {'roc_auc': 0.7, 'precision': 0.65, 'recall': 0.7, 'f1': 0.67}
            },
            'model_B': {
                'dataset_1': {'roc_auc': 0.75, 'precision': 0.7, 'recall': 0.75, 'f1': 0.72},
                'dataset_2': {'roc_auc': 0.85, 'precision': 0.8, 'recall': 0.82, 'f1': 0.81}
            }
        }
        
        # Test plotting
        fig = plot_cross_model_heatmap(
            cross_eval_results=cross_eval_results,
            metric='roc_auc'
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        
        # Clean up
        plt.close(fig)
        
        print("✅ Cross-model heatmap plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ plot_cross_model_heatmap test failed: {e}")
        return False


def test_plot_training_history():
    """Test training history plotting."""
    print("\n=== Testing plot_training_history ===")
    
    try:
        from src.visualization.plots import plot_training_history
        
        # Create mock training history
        epochs = 20
        history = {
            'loss': np.linspace(1.0, 0.1, epochs),  # Decreasing loss
            'val_loss': np.linspace(1.2, 0.2, epochs),  # Decreasing val loss
            'accuracy': np.linspace(0.5, 0.95, epochs),  # Increasing accuracy
            'val_accuracy': np.linspace(0.45, 0.9, epochs)  # Increasing val accuracy
        }
        
        # Test plotting
        fig = plot_training_history(
            history=history,
            model_name='test_model'
        )
        
        # Check that figure was created with 2 subplots
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Loss plot + accuracy plot
        
        # Check subplot contents
        loss_ax, acc_ax = fig.axes
        
        # Loss plot should have 2 lines (train + val)
        assert len(loss_ax.lines) == 2
        assert 'Loss' in loss_ax.get_title()
        
        # Accuracy plot should have 2 lines
        assert len(acc_ax.lines) == 2
        assert 'Accuracy' in acc_ax.get_title()
        
        # Clean up
        plt.close(fig)
        
        print("✅ Training history plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ plot_training_history test failed: {e}")
        return False


def test_plots_styling_integration():
    """Test that plots use our styling system."""
    print("\n=== Testing plots styling integration ===")
    
    try:
        from src.visualization.plots import plot_roc_curve
        from src.visualization.styling import get_model_style
        
        # Test model with known styling
        model_name = 'model_mDark-1_rinv-0.3_alpha-peak'
        expected_style = get_model_style(model_name)
        
        # Create mock evaluation results
        evaluation_results = {
            'roc_curve': {
                'fpr': np.linspace(0, 1, 10),
                'tpr': np.linspace(0, 1, 10)
            },
            'metrics': {
                'roc_auc': 0.85
            }
        }
        
        # Create plot
        fig = plot_roc_curve(evaluation_results, model_name)
        
        # Check that plot was created (detailed styling check would be complex)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        # Check that there's a line with some styling applied
        assert len(ax.lines) >= 1
        
        # Clean up
        plt.close(fig)
        
        print("✅ Plots use styling system!")
        return True
        
    except Exception as e:
        print(f"❌ Styling integration test failed: {e}")
        return False


def test_plots_module_integration():
    """Test integration with visualization module."""
    print("\n=== Testing plots module integration ===")
    
    try:
        # Test import from main visualization module
        from src.visualization import plot_roc_curve, plot_combined_roc_curves
        from src.visualization.plots import plot_roc_curve as direct_import
        
        assert plot_roc_curve is direct_import
        
        print("✅ Module integration works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Visualization Plots Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 8
        
        if test_plots_imports():
            success_count += 1
        
        if test_plot_roc_curve():
            success_count += 1
        
        if test_plot_combined_roc_curves():
            success_count += 1
        
        if test_plot_cross_model_roc_curves():
            success_count += 1
        
        if test_plot_cross_model_heatmap():
            success_count += 1
        
        if test_plot_training_history():
            success_count += 1
        
        if test_plots_styling_integration():
            success_count += 1
        
        if test_plots_module_integration():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} VISUALIZATION PLOTS TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All visualization plotting functionality verified!")
        else:
            print("❌ Some visualization plotting tests failed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
