#!/usr/bin/env python3
"""
Glaucoma Detection Web App - Python Startup Script
This script sets up the environment and starts the Flask web server.
"""

import os
import sys
import subprocess
import socket
from pathlib import Path

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_models():
    """Check if required models exist"""
    models_found = 0
    if Path('best_model.pth').exists():
        print("✓ Classification model found (best_model.pth)")
        models_found += 1
    else:
        print("⚠️  Classification model not found (best_model.pth)")
    
    if Path('seg_model_deeplabv3plus.pth').exists():
        print("✓ Segmentation model found (seg_model_deeplabv3plus.pth)")
        models_found += 1
    elif Path('seg_model.pth').exists():
        print("✓ Segmentation model found (seg_model.pth)")
        models_found += 1
    else:
        print("⚠️  Segmentation model not found")
    
    return models_found >= 1

def check_dependencies():
    """Check if required packages are installed"""
    required = ['flask', 'torch', 'torchvision', 'PIL', 'cv2']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt -r requirements_web.txt")
        return False
    
    print("✓ All required packages installed")
    return True

def is_port_available(port):
    """Check if port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            return False  # Port is in use
        return True
    except:
        return False

def main():
    print("\n" + "="*50)
    print("  Glaucoma Detection Web Application")
    print("="*50 + "\n")
    
    # Checks
    check_python_version()
    
    if not check_dependencies():
        print("\n❌ Cannot start app: missing dependencies")
        sys.exit(1)
    
    if not check_models():
        print("\n⚠️  Warning: Some models are missing")
        print("   The app may have limited functionality")
    
    # Check port
    port = 5000
    if not is_port_available(port):
        print(f"\n⚠️  Port {port} is already in use")
        print("   Close other applications or use a different port")
    
    print("\n" + "="*50)
    print("  Starting Flask Server...")
    print("="*50)
    print("\n📍 Server will start at: http://localhost:5000")
    print("   (Browser will open automatically when ready)\n")
    
    # Start Flask app
    try:
        import app as flask_app
        # This line won't be reached since app.py has run()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
