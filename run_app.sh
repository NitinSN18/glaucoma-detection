#!/bin/bash

# Glaucoma Detection Web App - Startup Script

echo "=========================================="
echo "  Glaucoma Detection Web Application"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null
pip install -q -r requirements_web.txt 2>/dev/null

echo "✓ Dependencies installed"
echo ""

# Check if models exist
echo "🔍 Checking for models..."
if [ ! -f "best_model.pth" ]; then
    echo "⚠️  Warning: best_model.pth not found"
fi
if [ ! -f "seg_model_deeplabv3plus.pth" ] && [ ! -f "seg_model.pth" ]; then
    echo "⚠️  Warning: Segmentation model not found"
fi

echo ""
echo "=========================================="
echo "  Starting Web Server..."
echo "=========================================="
echo ""

# Start the Flask app
python app.py
