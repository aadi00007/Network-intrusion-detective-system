#!/usr/bin/env python
"""Diagnostic script to identify HDC training issues"""
import sys
import traceback

print("="*60)
print("HDC Training Diagnostic")
print("="*60)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ numpy")
except Exception as e:
    print(f"   ✗ numpy: {e}")
    sys.exit(1)

try:
    from scipy import sparse
    print("   ✓ scipy")
except Exception as e:
    print(f"   ✗ scipy: {e}")
    sys.exit(1)

try:
    import sklearn
    print("   ✓ sklearn")
except Exception as e:
    print(f"   ✗ sklearn: {e}")
    sys.exit(1)

try:
    from hdc_model import HDClassifier
    print("   ✓ hdc_model")
except Exception as e:
    print(f"   ✗ hdc_model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: File existence
print("\n2. Checking data files...")
import os
files = ['KDDTrain+.txt', 'KDDTest+.txt']
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"   ✓ {f} ({size:.1f} MB)")
    else:
        print(f"   ✗ {f} NOT FOUND")

# Test 3: Model instantiation
print("\n3. Testing HDClassifier instantiation...")
try:
    clf = HDClassifier(dim=1000, n_bins=5, n_iter=2, random_state=42)
    print("   ✓ HDClassifier created")
except Exception as e:
    print(f"   ✗ HDClassifier creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Small fit test
print("\n4. Testing fit() with dummy data...")
try:
    import pandas as pd
    X_dummy = np.random.randn(10, 5)
    y_dummy = ['normal'] * 5 + ['dos'] * 5
    clf.fit(X_dummy, y_dummy)
    print("   ✓ fit() works")
except Exception as e:
    print(f"   ✗ fit() failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check nsl_kdd_analysis import
print("\n5. Testing nsl_kdd_analysis import...")
try:
    import nsl_kdd_analysis
    print("   ✓ nsl_kdd_analysis imported")
except Exception as e:
    print(f"   ✗ nsl_kdd_analysis import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
print("\nIf training still fails, please share the exact error message.")
print("You can run: python diagnose_hdc.py")

