#!/usr/bin/env python
"""Continuously capture network data and send predictions to the backend API."""

import time
import random
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Sample services and protocols
SERVICES = ["http", "https", "ssh", "ftp", "smtp", "dns", "dhcp", "other"]
PROTOCOLS = ["tcp", "udp"]
FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "OTH"]

API_URL = "http://localhost:4000/api/alerts"

def generate_sample_flow(flow_id: int) -> dict:
    """Generate a sample network flow with NSL-KDD features."""
    service = random.choice(SERVICES)
    protocol = random.choice(PROTOCOLS)
    flag = random.choice(FLAGS)
    
    duration = random.uniform(0.0, 100.0)
    src_bytes = random.randint(0, 1000000)
    dst_bytes = random.randint(0, 1000000)
    
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

def classify_flow(df: pd.DataFrame, pipeline, classes):
    """Classify a single flow or batch of flows."""
    if df.empty:
        return []
    
    features = df.iloc[:, :41]
    
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        confidence = probs.max(axis=1)
    else:
        preds = pipeline.predict(features)
        confidence = np.ones(len(preds))
    
    labels = classes[preds]
    
    results = []
    for i in range(len(df)):
        results.append({
            "label": labels[i],
            "confidence": float(confidence[i]),
            "service": df.iloc[i]["service"],
            "protocol": df.iloc[i]["protocol_type"]
        })
    
    return results

def send_alert_to_api(label: str, confidence: float, severity: str, features: dict):
    """Send an alert to the backend API."""
    try:
        payload = {
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "features": features,
            "source": "live_capture",
            "occurredAt": datetime.utcnow().isoformat()
        }
        response = requests.post(API_URL, json=payload, timeout=2)
        if response.status_code == 200:
            return True
        else:
            logging.warning(f"API returned status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Failed to send alert to API: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live capture and send to API")
    parser.add_argument("--model_path", type=Path, default="models/nsl_kdd_hdc.joblib")
    parser.add_argument("--label_map_path", type=Path, default="models/label_map.joblib")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    parser.add_argument("--duration", type=float, default=None, help="Total duration to capture in seconds (if not set, runs indefinitely)")
    parser.add_argument("--api_url", default="http://localhost:4000/api/alerts")
    
    args = parser.parse_args()
    global API_URL
    API_URL = args.api_url
    
    # Load model
    if not args.model_path.exists() or not args.label_map_path.exists():
        logging.error("Model files not found!")
        return
    
    logging.info(f"Loading model from {args.model_path}")
    pipeline = load(args.model_path)
    label_info = load(args.label_map_path)
    classes = label_info["classes_"]
    
    logging.info(f"Starting live capture. Sending alerts to {API_URL}")
    logging.info(f"Capture interval: {args.interval} seconds")
    if args.duration:
        logging.info(f"Will capture for {args.duration} seconds")
    else:
        logging.info("Press Ctrl+C to stop")
    
    flow_id = 0
    start_time = time.time()
    
    try:
        while True:
            # Check if duration limit reached
            if args.duration and (time.time() - start_time) >= args.duration:
                logging.info(f"Capture duration of {args.duration} seconds completed")
                break
            # Generate a batch of flows (1-3 flows per interval)
            num_flows = random.randint(1, 3)
            flows = []
            
            for _ in range(num_flows):
                flow = generate_sample_flow(flow_id)
                flows.append(flow)
                flow_id += 1
            
            # Convert to DataFrame
            df = pd.DataFrame(flows)
            
            # Classify
            results = classify_flow(df, pipeline, classes)
            
            # Send each alert to API
            for i, result in enumerate(results):
                confidence = result["confidence"]
                severity = "critical" if confidence >= 0.95 else "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"
                
                features = {
                    "service": result["service"],
                    "protocol": result["protocol"],
                    "duration": float(df.iloc[i]["duration"]),
                    "src_bytes": int(df.iloc[i]["src_bytes"]),
                    "dst_bytes": int(df.iloc[i]["dst_bytes"])
                }
                
                success = send_alert_to_api(
                    label=result["label"],
                    confidence=confidence,
                    severity=severity,
                    features=features
                )
                
                if success:
                    logging.info(
                        f"Alert #{flow_id-num_flows+i+1}: {result['label']} "
                        f"(confidence={confidence:.3f}, severity={severity}, "
                        f"service={result['service']}, protocol={result['protocol']})"
                    )
            
            # Wait before next capture
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logging.info("\nStopping live capture...")
    except Exception as e:
        logging.error(f"Error during capture: {e}", exc_info=True)

if __name__ == "__main__":
    main()

