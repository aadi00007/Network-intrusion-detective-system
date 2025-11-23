# send_predictions_with_env_token.py
import os, csv, time, requests
from datetime import datetime

BACKEND_URL = "http://localhost:4000/api/alerts"
CSV_FILE = os.path.join(os.getcwd(), "tmp_live_predictions.csv")
TOKEN = os.environ.get("IDS_TOKEN")

if not TOKEN:
    print("ERROR: IDS_TOKEN not set. Run: export IDS_TOKEN=\"<token>\"")
    raise SystemExit(1)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def row_to_payload(row):
    label = row.get("predicted_label") or row.get("label") or row.get("prediction") or "unknown"
    confidence = float(row.get("confidence") or row.get("score") or 0)
    return {
        "label": label,
        "confidence": confidence,
        "severity": row.get("severity") or "medium",
        "features": row,
        "raw": row,
        "source": "live_capture",
        "occurredAt": datetime.utcnow().isoformat()
    }

def main():
    if not os.path.exists(CSV_FILE):
        print("CSV not found:", CSV_FILE)
        return
    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            payload = row_to_payload(row)
            try:
                r = requests.post(BACKEND_URL, json=payload, headers=HEADERS, timeout=5)
                print(i, r.status_code, r.text)
            except Exception as e:
                print("Error posting:", e)
            time.sleep(0.05)

if __name__ == "__main__":
    main()

