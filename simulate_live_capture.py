#!/usr/bin/env python
"""Simulate live network capture for 20 seconds and run predictions.

This script generates sample network traffic data that mimics what would be
captured from a live network, then runs the HDC model to classify it.
"""

import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Sample services and protocols
SERVICES = ["http", "https", "ssh", "ftp", "smtp", "dns", "dhcp", "other"]
PROTOCOLS = ["tcp", "udp"]
FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "OTH"]

def generate_sample_flow(flow_id: int) -> dict:
    """Generate a sample network flow with NSL-KDD features."""
    service = random.choice(SERVICES)
    protocol = random.choice(PROTOCOLS)
    flag = random.choice(FLAGS)
    
    # Generate realistic network flow data
    duration = random.uniform(0.0, 100.0)
    src_bytes = random.randint(0, 1000000)
    dst_bytes = random.randint(0, 1000000)
    
    # Create a feature row matching NSL-KDD format (41 features)
    row = {
        "duration": duration,
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0,
        "wrong_fragment": random.randint(0, 5),
        "urgent": random.randint(0, 10),
        "hot": random.randint(0, 5),
        "num_failed_logins": random.randint(0, 5),
        "logged_in": random.randint(0, 1),
        "num_compromised": random.randint(0, 10),
        "root_shell": random.randint(0, 1),
        "su_attempted": random.randint(0, 1),
        "num_root": random.randint(0, 5),
        "num_file_creations": random.randint(0, 10),
        "num_shells": random.randint(0, 5),
        "num_access_files": random.randint(0, 10),
        "num_outbound_cmds": 0,
        "is_host_login": random.randint(0, 1),
        "is_guest_login": random.randint(0, 1),
        "count": random.randint(1, 100),
        "srv_count": random.randint(1, 50),
        "serror_rate": random.uniform(0.0, 1.0),
        "srv_serror_rate": random.uniform(0.0, 1.0),
        "rerror_rate": random.uniform(0.0, 1.0),
        "srv_rerror_rate": random.uniform(0.0, 1.0),
        "same_srv_rate": random.uniform(0.0, 1.0),
        "diff_srv_rate": random.uniform(0.0, 1.0),
        "srv_diff_host_rate": random.uniform(0.0, 1.0),
        "dst_host_count": random.randint(1, 200),
        "dst_host_srv_count": random.randint(1, 100),
        "dst_host_same_srv_rate": random.uniform(0.0, 1.0),
        "dst_host_diff_srv_rate": random.uniform(0.0, 1.0),
        "dst_host_same_src_port_rate": random.uniform(0.0, 1.0),
        "dst_host_srv_diff_host_rate": random.uniform(0.0, 1.0),
        "dst_host_serror_rate": random.uniform(0.0, 1.0),
        "dst_host_srv_serror_rate": random.uniform(0.0, 1.0),
        "dst_host_rerror_rate": random.uniform(0.0, 1.0),
        "dst_host_srv_rerror_rate": random.uniform(0.0, 1.0),
    }
    return row

def simulate_capture(duration_seconds: int = 20):
    """Simulate network capture for specified duration."""
    flows = []
    start_time = time.time()
    flow_id = 0
    
    logging.info(f"Starting simulated capture for {duration_seconds} seconds...")
    
    while time.time() - start_time < duration_seconds:
        # Generate flows at random intervals (simulating network activity)
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate 1-5 flows per interval
        num_flows = random.randint(1, 5)
        for _ in range(num_flows):
            flow = generate_sample_flow(flow_id)
            flows.append(flow)
            flow_id += 1
        
        elapsed = time.time() - start_time
        if flow_id % 10 == 0:
            logging.info(f"Captured {flow_id} flows in {elapsed:.1f} seconds...")
    
    logging.info(f"Capture complete: {len(flows)} total flows captured")
    return pd.DataFrame(flows)

def classify_flows(df: pd.DataFrame, model_path: Path, label_map_path: Path):
    """Classify flows using the trained model."""
    if df.empty:
        logging.warning("No flows to classify")
        return pd.DataFrame()
    
    logging.info(f"Loading model from {model_path}")
    pipeline = load(model_path)
    label_info = load(label_map_path)
    classes = label_info["classes_"]
    
    # Extract features (first 41 columns)
    features = df.iloc[:, :41]
    
    # Predict
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        confidence = probs.max(axis=1)
    else:
        preds = pipeline.predict(features)
        confidence = np.ones(len(preds))
    
    labels = classes[preds]
    
    result = df.copy()
    result["predicted_label"] = labels
    result["confidence"] = confidence
    
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simulate live network capture")
    parser.add_argument("--timeout", type=int, default=20, help="Capture duration in seconds")
    parser.add_argument("--model_path", type=Path, default="models/nsl_kdd_hdc.joblib")
    parser.add_argument("--label_map_path", type=Path, default="models/label_map.joblib")
    parser.add_argument("--output_csv", type=Path, default="tmp_live_predictions.csv")
    parser.add_argument("--predict", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Simulate capture
    df = simulate_capture(args.timeout)
    
    # Save raw features
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        logging.info(f"Saved {len(df)} flows to {args.output_csv}")
    
    # Classify if requested
    if args.predict and args.model_path.exists() and args.label_map_path.exists():
        predictions = classify_flows(df, args.model_path, args.label_map_path)
        
        if not predictions.empty:
            logging.info("\n=== Classification Results ===")
            for idx, row in predictions.iterrows():
                logging.info(
                    f"Flow {idx} → label={row['predicted_label']} "
                    f"confidence={row['confidence']:.3f} "
                    f"service={row['service']} protocol={row['protocol_type']}"
                )
            
            # Save predictions
            if args.output_csv:
                predictions.to_csv(args.output_csv, index=False)
                logging.info(f"Saved predictions to {args.output_csv}")
            
            # Summary statistics
            label_counts = predictions["predicted_label"].value_counts()
            logging.info("\n=== Summary ===")
            logging.info(f"Total flows: {len(predictions)}")
            logging.info(f"Label distribution:\n{label_counts}")
            avg_confidence = predictions["confidence"].mean()
            logging.info(f"Average confidence: {avg_confidence:.3f}")
        else:
            logging.warning("No predictions generated")
    elif args.predict:
        logging.warning("Model files not found. Skipping classification.")

if __name__ == "__main__":
    main()

