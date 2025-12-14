# send_predictions_auto.py - Automatically login and send predictions
import os, csv, time, requests
from datetime import datetime

BACKEND_URL = "http://localhost:4000"
CSV_FILE = os.path.join(os.getcwd(), "tmp_live_predictions.csv")

# Default admin credentials
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"

def get_token():
    """Login and get JWT token"""
    try:
        r = requests.post(f"{BACKEND_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        }, timeout=5)
        if r.status_code == 200:
            return r.json().get("token")
        else:
            print(f"Login failed: {r.status_code} {r.text}")
            return None
    except Exception as e:
        print(f"Login error: {e}")
        return None

def row_to_payload(row):
    label = row.get("predicted_label") or row.get("label") or row.get("prediction") or "unknown"
    confidence = float(row.get("confidence") or row.get("score") or 0)
    
    # Map attack types to severity
    severity_map = {
        "normal": "low",
        "dos": "high",
        "probe": "medium",
        "r2l": "critical",
        "u2r": "critical",
        "mscan": "high",
        "saint": "high",
        "apache2": "high",
        "back": "high",
        "land": "high",
        "mailbomb": "high",
        "neptune": "high",
        "pod": "high",
        "smurf": "high",
        "teardrop": "high",
        "udpstorm": "high",
    }
    severity = severity_map.get(label.lower(), "medium")
    
    return {
        "label": label,
        "confidence": confidence,
        "severity": severity,
        "features": dict(row),
        "raw": dict(row),
        "source": "hdc_model",
        "occurredAt": datetime.utcnow().isoformat()
    }

def main():
    # Get token
    print("Logging in...")
    token = get_token()
    if not token:
        print("ERROR: Could not get authentication token")
        return
    
    print(f"Got token, sending predictions from {CSV_FILE}")
    
    if not os.path.exists(CSV_FILE):
        print(f"CSV not found: {CSV_FILE}")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    sent = 0
    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            payload = row_to_payload(row)
            try:
                r = requests.post(f"{BACKEND_URL}/api/alerts", json=payload, headers=headers, timeout=5)
                if r.status_code in [200, 201]:
                    sent += 1
                    if sent % 10 == 0:
                        print(f"Sent {sent} predictions...")
                else:
                    print(f"Error on row {i}: {r.status_code} {r.text[:100]}")
            except Exception as e:
                print(f"Error posting row {i}: {e}")
            time.sleep(0.05)  # Small delay to avoid overwhelming backend
    
    print(f"\nDone! Sent {sent} predictions to backend.")
    print(f"Check http://localhost:5173 to see them in the dashboard!")

if __name__ == "__main__":
    main()

