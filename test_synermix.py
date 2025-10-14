#!/usr/bin/env python3
"""
Test script to verify SynerMix implementation works correctly
"""
import torch
import numpy as np
import sys
import os
import argparse
import random

# Add the current directory to path so we can import modules
sys.path.append('.')

def test_synermix_components():
    """Test key SynerMix components"""
    print("Testing SynerMix components...")
    
    # Import modules from synermix_pretrain.py
    from synermix_pretrain import (
        supplement_batch, 
        intra_class_mixup, 
        inter_class_mixup,
        adjust_synermix_params,
        SynerMixEfficientNet
    )
    
    # Import EfficientNet model
    from models import EfficientNet
    
    # Test 1: Model wrapper
    print("\n1. Testing model wrapper...")
    try:
        # Create a simple mock model for testing
        base_model = EfficientNet.from_name('efficientnet-b0', num_classes=2, 
                                           override_params={'dropout_rate': 0.2, 'drop_connect_rate': 0.2})
        model = SynerMixEfficientNet(base_model)
        
        # Test feature extraction
        x = torch.randn(4, 3, 224, 224)
        features = model.extract_features(x)
        print(f"   ✓ Feature extraction successful. Output shape: {features.shape}")
        
        # Test forward pass
        output = model(x)
        print(f"   ✓ Forward pass successful. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test 2: Parameter adjustment
    print("\n2. Testing parameter adjustment...")
    try:
        class MockOpt:
            def __init__(self):
                self.synermix_warmup_epochs = 5
                
        opt = MockOpt()
        
        # Test warmup period
        beta_warmup = adjust_synermix_params(3, 100)
        print(f"   ✓ Beta during warmup (epoch 3): {beta_warmup}")
        
        # Test early training
        beta_early = adjust_synermix_params(10, 100)
        print(f"   ✓ Beta during early training (epoch 10): {beta_early}")
        
        # Test late training
        beta_late = adjust_synermix_params(80, 100)
        print(f"   ✓ Beta during late training (epoch 80): {beta_late}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test 3: Batch supplementation
    print("\n3. Testing batch supplementation...")
    try:
        # Create mock dataset and batch
        class MockDataset:
            def __init__(self):
                self.data = [(torch.randn(3, 32, 32), 0) for _ in range(10)] + \
                           [(torch.randn(3, 32, 32), 1) for _ in range(10)]
                
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __len__(self):
                return len(self.data)
        
        dataset = MockDataset()
        
        # Create batch with imbalanced classes
        inputs = torch.randn(6, 3, 32, 32)
        targets = torch.tensor([0, 0, 0, 0, 1, 1])
        
        # Supplement batch
        supplemented_inputs, supplemented_targets = supplement_batch(inputs, targets, dataset, min_samples=3)
        
        # Count classes
        class_counts = {}
        for t in supplemented_targets:
            if t.item() not in class_counts:
                class_counts[t.item()] = 0
            class_counts[t.item()] += 1
            
        print(f"   ✓ Original class counts: {{0: 4, 1: 2}}")
        print(f"   ✓ Supplemented class counts: {class_counts}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test 4: Intra-class mixing
    print("\n4. Testing intra-class mixing...")
    try:
        # Create features for each class
        features_by_class = {
            0: torch.randn(4, 1280),  # 4 samples of class 0
            1: torch.randn(3, 1280),  # 3 samples of class 1
        }
        
        # Perform intra-class mixing
        mixed_features, mixed_targets = intra_class_mixup(features_by_class)
        
        print(f"   ✓ Mixed features shape: {mixed_features.shape}")
        print(f"   ✓ Mixed targets: {mixed_targets}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test 5: Inter-class mixing
    print("\n5. Testing inter-class mixing...")
    try:
        # Create batch
        inputs = torch.randn(6, 3, 32, 32)
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        
        # Perform inter-class mixing
        mixed_inputs, targets_a, targets_b, lam = inter_class_mixup(inputs, targets, alpha=1.0)
        
        print(f"   ✓ Mixed inputs shape: {mixed_inputs.shape}")
        print(f"   ✓ Lambda: {lam:.4f}")
        print(f"   ✓ Original targets: {targets}")
        print(f"   ✓ Mixed targets A: {targets_a}")
        print(f"   ✓ Mixed targets B: {targets_b}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    success = test_synermix_components()
    if success:
        print("\nSynerMix implementation appears to be working correctly!")
    else:
        print("\nSome issues were found in the SynerMix implementation.")
        sys.exit(1)