#!/usr/bin/env python3
"""
Test script to verify SynerMix implementation works correctly
"""
import torch
import numpy as np
import sys
import os

# Add the current directory to path so we can import modules
sys.path.append('/Ubuntu/home/vincent/Thesis-master')

from synermix_pretrain import SynerMixEfficientNet, get_classes_with_sufficient_samples, intra_class_mixup, inter_class_mixup

def test_synermix_components():
    """Test key SynerMix components"""
    print("Testing SynerMix components...")

    # Test 1: Model wrapper
    print("1. Testing model wrapper...")
    try:
        # Create a simple mock model for testing
        from models import EfficientNet
        base_model = EfficientNet.from_name('efficientnet-b0', num_classes=2, override_params={'dropout_rate': 0.2, 'drop_connect_rate': 0.2})
        model = SynerMixEfficientNet(base_model)

        # Test feature extraction
        x = torch.randn(4, 3, 224, 224)
        features = model.extract_features(x)
        print(f"   Feature extraction successful. Output shape: {features.shape}")

        # Test forward pass
        output = model(x)
        print(f"   Forward pass successful. Output shape: {output.shape}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    # Test 2: Class filtering
    print("2. Testing class filtering...")
    try:
        targets = torch.tensor([0, 0, 1, 1, 1])  # 2 samples of class 0, 3 samples of class 1
        sufficient_classes = get_classes_with_sufficient_samples(targets, min_samples=2)
        print(f"   Classes with >= 2 samples: {sufficient_classes}")

        targets2 = torch.tensor([0, 1])  # Only 1 sample each
        sufficient_classes2 = get_classes_with_sufficient_samples(targets2, min_samples=2)
        print(f"   Classes with >= 2 samples: {sufficient_classes2}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    # Test 3: Intra-class mixing
    print("3. Testing intra-class mixing...")
    try:
        features_by_class = {
            0: torch.randn(3, 1280),  # 3 samples of class 0
            1: torch.randn(2, 1280),  # 2 samples of class 1
        }

        mixed_features, mixed_targets = intra_class_mixup(features_by_class, mix_ratio=0.5)
        print(f"   Intra-class mixing successful. Mixed features: {len(mixed_features)}, Mixed targets: {len(mixed_targets)}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    # Test 4: Inter-class mixing
    print("4. Testing inter-class mixing...")
    try:
        inputs = torch.randn(4, 3, 224, 224)
        targets = torch.tensor([0, 0, 1, 1])

        mixed_inputs, targets_a, targets_b, lam = inter_class_mixup(inputs, targets, alpha=1.0)
        print(f"   Inter-class mixing successful. Lambda: {lam:.4f}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_synermix_components()
    if success:
        print("\nSynerMix implementation appears to be working correctly!")
    else:
        print("\nSome issues were found in the SynerMix implementation.")
        sys.exit(1)
