#!/usr/bin/env python
"""Test HDC training to capture errors"""
import sys
import traceback

try:
    print("Testing HDC import...")
    from hdc_model import HDClassifier
    print("✓ HDC import successful")
    
    print("\nTesting nsl_kdd_analysis import...")
    import nsl_kdd_analysis
    print("✓ nsl_kdd_analysis import successful")
    
    print("\nAttempting to run training...")
    sys.argv = [
        'nsl_kdd_analysis.py',
        'train',
        '--train_path', 'KDDTrain+.txt',
        '--test_path', 'KDDTest+.txt',
        '--model_type', 'hdc',
        '--model_out', 'models/nsl_kdd_hdc.joblib',
        '--label_map_out', 'models/label_map.joblib',
        '--report_out', 'reports/metrics_hdc.json'
    ]
    
    nsl_kdd_analysis.main()
    
except Exception as e:
    print("\n" + "="*60)
    print("ERROR OCCURRED:")
    print("="*60)
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

