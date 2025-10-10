#!/usr/bin/env python3
"""
Test the visualization styling module.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_styling_imports():
    """Test that we can import styling functions."""
    print("=== Testing styling imports ===")
    
    try:
        from src.visualization.styling import get_model_style, format_model_name
        print("✅ Styling functions import works!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_get_model_style():
    """Test get_model_style function with various model names."""
    print("\n=== Testing get_model_style ===")
    
    try:
        from src.visualization.styling import get_model_style
        
        # Test exact match
        style1 = get_model_style("model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak")
        assert "color" in style1
        assert "linestyle" in style1
        assert "linewidth" in style1
        assert style1["color"] == "blue"
        print("✅ Exact match styling works!")
        
        # Test parameter-based match
        style2 = get_model_style("some_custom_mDark-5_rinv-0.3_alpha-peak_model")
        assert style2["color"] == "red"
        print("✅ Parameter-based styling works!")
        
        # Test hash-based fallback
        style3 = get_model_style("completely_unknown_model_name")
        assert "color" in style3
        assert style3["linewidth"] == 2
        print("✅ Hash-based fallback styling works!")
        
        return True
        
    except Exception as e:
        print(f"❌ get_model_style test failed: {e}")
        return False


def test_format_model_name():
    """Test format_model_name function with various inputs."""
    print("\n=== Testing format_model_name ===")
    
    try:
        from src.visualization.styling import format_model_name
        
        # Test full parameter extraction
        formatted1 = format_model_name("model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak")
        expected1 = "mDark-1.0_rinv-0.3_alpha-peak"
        assert formatted1 == expected1, f"Expected {expected1}, got {formatted1}"
        print("✅ Full parameter formatting works!")
        
        # Test with integer mDark
        formatted2 = format_model_name("loo_AutomatedCMS_mZprime-2000_mDark-5_rinv-0.2_alpha-low")
        expected2 = "mDark-5.0_rinv-0.2_alpha-low"
        assert formatted2 == expected2, f"Expected {expected2}, got {formatted2}"
        print("✅ Integer mDark formatting works!")
        
        # Test SM file
        formatted3 = format_model_name("path/to/NominalSM.h5")
        expected3 = "Standard Model"
        assert formatted3 == expected3, f"Expected {expected3}, got {formatted3}"
        print("✅ SM file formatting works!")
        
        # Test prefix removal
        formatted4 = format_model_name("model_some_name")
        expected4 = "some_name"
        assert formatted4 == expected4, f"Expected {expected4}, got {formatted4}"
        print("✅ Prefix removal works!")
        
        return True
        
    except Exception as e:
        print(f"❌ format_model_name test failed: {e}")
        return False


def test_styling_consistency():
    """Test that styling is consistent across calls."""
    print("\n=== Testing styling consistency ===")
    
    try:
        from src.visualization.styling import get_model_style, format_model_name
        
        # Same model name should always get same style
        model_name = "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak"
        style1 = get_model_style(model_name)
        style2 = get_model_style(model_name)
        
        assert style1 == style2, "Same model should get consistent styling"
        print("✅ Styling consistency works!")
        
        # Same formatting should be consistent
        formatted1 = format_model_name(model_name)
        formatted2 = format_model_name(model_name)
        
        assert formatted1 == formatted2, "Same formatting should be consistent"
        print("✅ Formatting consistency works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Styling consistency test failed: {e}")
        return False


def test_physics_parameter_recognition():
    """Test recognition of physics parameters."""
    print("\n=== Testing physics parameter recognition ===")
    
    try:
        from src.visualization.styling import get_model_style, format_model_name
        
        # Test different parameter combinations get different styles
        models = [
            "mDark-1_rinv-0.3_alpha-peak",
            "mDark-5_rinv-0.3_alpha-peak", 
            "mDark-1_rinv-0.2_alpha-peak",
            "mDark-1_rinv-0.3_alpha-low"
        ]
        
        styles = [get_model_style(model) for model in models]
        colors = [style["color"] for style in styles]
        
        # All should have different colors (based on hardcoded mapping)
        assert len(set(colors)) == len(colors), "Different parameter combinations should get different colors"
        print("✅ Physics parameter differentiation works!")
        
        # Test parameter extraction in formatting
        for model in models:
            formatted = format_model_name(model)
            assert "mDark-" in formatted, f"mDark parameter missing in {formatted}"
            assert "rinv-" in formatted, f"rinv parameter missing in {formatted}"
            assert "alpha-" in formatted, f"alpha parameter missing in {formatted}"
        
        print("✅ Physics parameter extraction works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics parameter recognition test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Visualization Styling Module")
    print("=" * 50)
    
    try:
        success_count = 0
        total_tests = 5
        
        if test_styling_imports():
            success_count += 1
        
        if test_get_model_style():
            success_count += 1
        
        if test_format_model_name():
            success_count += 1
        
        if test_styling_consistency():
            success_count += 1
        
        if test_physics_parameter_recognition():
            success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎉 {success_count}/{total_tests} STYLING TESTS PASSED!")
        
        if success_count == total_tests:
            print("✅ All styling functionality verified!")
        else:
            print("❌ Some styling tests failed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
