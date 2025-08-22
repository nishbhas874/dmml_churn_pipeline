# test_setup.py - Simple test for recipients
# Test if the pipeline setup works correctly

import os
import subprocess
import sys

def test_packages():
    """Test if required packages are installed"""
    print("Testing package installation...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'kaggle', 'datasets'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All packages installed!")
    return True

def test_kaggle():
    """Test Kaggle API setup"""
    print("\nTesting Kaggle API...")
    
    try:
        result = subprocess.run(['kaggle', 'datasets', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Kaggle API working!")
            return True
        else:
            print("✗ Kaggle API error")
            print("Make sure kaggle.json is in ~/.kaggle/ folder")
            return False
    except Exception as e:
        print(f"✗ Kaggle test failed: {e}")
        return False

def test_data_download():
    """Test data download"""
    print("\nTesting data download...")
    
    try:
        result = subprocess.run(['python', 'get_data.py'], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ Data download successful!")
            return True
        else:
            print("✗ Data download failed")
            print("Check your Kaggle API setup")
            return False
    except Exception as e:
        print(f"✗ Data download test failed: {e}")
        return False

def main():
    print("Pipeline Setup Test")
    print("=" * 30)
    
    tests = [
        ("Package Installation", test_packages),
        ("Kaggle API", test_kaggle), 
        ("Data Download", test_data_download)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 Setup successful! Pipeline ready to use.")
        print("Next: Run 'python pipeline.py' for full pipeline")
    else:
        print("\n⚠️ Setup incomplete. Check SETUP_INSTRUCTIONS.md")

if __name__ == "__main__":
    main()
