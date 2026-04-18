#!/usr/bin/env python3
"""
Launcher script for T-GD Model Evaluation UI
"""
import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import torchvision
        import gradio
        import PIL
        import numpy
        import sklearn
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def install_requirements():
    """Install requirements if needed"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def main():
    print("🚀 T-GD: Model Evaluation UI")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("models") or not os.path.exists("weights"):
        print("❌ Error: Please run this script from the T-GD repository root directory")
        print("Make sure you have the 'models' and 'weights' folders present")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n📦 Installing missing requirements...")
        if not install_requirements():
            print("Please manually install requirements with: pip install -r requirements.txt")
            sys.exit(1)
    
    print("\n🌐 Starting Gradio Model Evaluation UI...")
    print("The interface will be available at: http://localhost:7861")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the UI
    try:
        from eval_ui import create_interface
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            debug=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
