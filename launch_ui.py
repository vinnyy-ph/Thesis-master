#!/usr/bin/env python3
"""
Simple launcher script for T-GD Gradio UI
"""
import os
import sys
import subprocess

# UTF-8 console so emoji prints don't crash a cp1252 (Windows) terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def check_requirements():
    """Check if required packages are installed"""
    try:
        # Availability probes — imported only to detect missing dependencies.
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import gradio  # noqa: F401
        import PIL  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import matplotlib  # noqa: F401
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
    print("🚀 T-GD: GAN-Generated Image Detection UI")
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
    
    # Check if weights exist
    if not os.path.exists("weights/pre-train") and not os.path.exists("weights/t-gd"):
        print("\n⚠️  Warning: No model weights found!")
        print("Please download weights using:")
        print("cd weights && bash download_weights.sh")
        print("\nOr manually place weight files in weights/pre-train/ and weights/t-gd/")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n🌐 Starting Gradio UI...")
    print("The interface will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the UI
    try:
        from gradio_ui import create_interface
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
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
