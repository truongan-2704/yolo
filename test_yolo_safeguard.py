"""
Test script for YOLO-SafeGuard PPE detection architecture.
Validates that the model builds correctly and prints architecture summary.
"""
import torch
import sys

def test_safeguard_blocks():
    """Test individual SafeGuard blocks."""
    from ultralytics.nn.modules.safeguard_blocks import (
        LightCoordAtt, SafeGuardPConv, SafeGuardBottleneck,
        C3k2_SafeGuard, SafeGuardCSP, BodyContextModule,
    )
    
    x = torch.randn(1, 64, 40, 40)
    
    # Test LightCoordAtt
    ca = LightCoordAtt(64)
    out = ca(x)
    assert out.shape == x.shape, f"LightCoordAtt shape mismatch: {out.shape}"
    print(f"✓ LightCoordAtt: {x.shape} → {out.shape}")
    
    # Test SafeGuardPConv
    pconv = SafeGuardPConv(64)
    out = pconv(x)
    assert out.shape == x.shape, f"SafeGuardPConv shape mismatch: {out.shape}"
    print(f"✓ SafeGuardPConv: {x.shape} → {out.shape}")
    
    # Test SafeGuardBottleneck
    bn = SafeGuardBottleneck(64, 64)
    out = bn(x)
    assert out.shape == x.shape, f"SafeGuardBottleneck shape mismatch: {out.shape}"
    print(f"✓ SafeGuardBottleneck: {x.shape} → {out.shape}")
    
    # Test C3k2_SafeGuard
    c3k2 = C3k2_SafeGuard(64, 128, n=2, c3k=False)
    out = c3k2(x)
    assert out.shape == (1, 128, 40, 40), f"C3k2_SafeGuard shape mismatch: {out.shape}"
    print(f"✓ C3k2_SafeGuard: {x.shape} → {out.shape}")
    
    # Test SafeGuardCSP
    csp = SafeGuardCSP(64, 128, n=2)
    out = csp(x)
    assert out.shape == (1, 128, 40, 40), f"SafeGuardCSP shape mismatch: {out.shape}"
    print(f"✓ SafeGuardCSP: {x.shape} → {out.shape}")
    
    # Test BodyContextModule
    bcm = BodyContextModule(64)
    out = bcm(x)
    assert out.shape == x.shape, f"BodyContextModule shape mismatch: {out.shape}"
    print(f"✓ BodyContextModule: {x.shape} → {out.shape}")
    
    print("\n✅ All SafeGuard blocks passed!\n")


def test_safeguard_model():
    """Test full YOLO-SafeGuard model build."""
    from ultralytics import YOLO
    
    yaml_path = r'ultralytics/cfg/models/11/yolo11-SafeGuard/yolo11-SafeGuard.yaml'
    
    print(f"Loading model from: {yaml_path}")
    model = YOLO(yaml_path)
    
    # Print model info
    model.info(verbose=True)
    
    print("\n✅ YOLO-SafeGuard model built successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")


if __name__ == '__main__':
    print("=" * 60)
    print("YOLO-SafeGuard PPE Detection — Test Suite")
    print("=" * 60)
    
    print("\n--- Test 1: Individual Blocks ---")
    test_safeguard_blocks()
    
    print("\n--- Test 2: Full Model Build ---")
    test_safeguard_model()
