#!/usr/bin/env python3
"""
Install all required dependencies for the ML stock predictor
"""

import subprocess
import sys

def install_dependencies():
    """Install all required dependencies"""
    
    dependencies = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.0"
    ]
    
    print("🚀 Installing ML Stock Predictor Dependencies")
    print("=" * 50)
    
    for package in dependencies:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("\n🎉 All dependencies installed successfully!")
    print("\nYou can now run:")
    print("  python train_model.py AAPL")
    print("  python train_model_fixed.py AAPL")
    
    return True

if __name__ == "__main__":
    install_dependencies()
