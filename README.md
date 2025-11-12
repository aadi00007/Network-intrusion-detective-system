## NSL-KDD Intrusion Detection System (IDS)

This project builds a scalable, multi-model IDS on the NSL-KDD dataset. It supports Random Forest, SVM, MLP, and an ensemble, with proper preprocessing, metrics, feature importance, and a basic real-time alert watcher.

### Environment
```bash
python -V   # use the venv included: /venv
source ./venv/bin/activate
```

### Train and Evaluate
```bash
# Random Forest (default), saves model + report
python nsl_kdd_analysis.py train \
  --train_path KDDTrain+.txt \
  --test_path KDDTest+.txt \
  --model_type rf \
  --class_weight balanced \
  --model_out models/nsl_kdd_model.joblib \
  --label_map_out models/label_map.joblib \
  --report_out reports/metrics.json

# SVM (rbf)
python nsl_kdd_analysis.py train --model_type svm --svm_kernel rbf

# MLP
python nsl_kdd_analysis.py train --model_type mlp

# Ensemble (soft voting over RF, SVM, MLP)
python nsl_kdd_analysis.py train --model_type ensemble
```

Outputs:
- `models/nsl_kdd_model.joblib`: full sklearn Pipeline (preprocess + model)
- `models/label_map.joblib`: target classes
- `reports/metrics.json`: accuracy, macro/weighted precision/recall/F1

### Predict on a File
```bash
python nsl_kdd_analysis.py predict \
  --model_path models/nsl_kdd_model.joblib \
  --label_map_path models/label_map.joblib \
  --input_path KDDTest+.txt \
  --output_path predictions.csv
```

The predictions file appends two columns: `predicted_label`, `confidence`.

### Real-time Watcher (basic)
Watches a growing CSV (rows appended) and prints alerts.
```bash
python nsl_kdd_analysis.py watch \
  --model_path models/nsl_kdd_model.joblib \
  --label_map_path models/label_map.joblib \
  --input_path live.csv \
  --alert_label normal \
  --min_confidence 0.5
```

### Live Capture with Scapy
Capture live TCP/UDP traffic, approximate NSL-KDD features, and optionally run predictions.

> Requires root privileges and [Scapy](https://scapy.net) installed in your environment.

```bash
sudo ./venv/bin/python live_capture.py sniff \
  --iface en0 \
  --timeout 15 \
  --model_path models/nsl_kdd_model.joblib \
  --label_map_path models/label_map.joblib \
  --predict \
  --output_csv tmp_live_predictions.csv
```

- `--iface`: network interface to monitor (omit to let Scapy choose).
- `--count` / `--timeout`: stop sniffing after N packets or seconds.
- `--predict`: classify captured flows if model artifacts are supplied.
- `--output_csv`: persist features/predictions for later analysis.

**Limitations**
- Only IPv4 TCP/UDP packets are processed.
- Many statistical NSL-KDD features are estimated using capture-local heuristics.
- Service identification is best-effort based on destination port.

### Data & Preprocessing
- Uses official NSL-KDD `KDDTrain+.txt` (train) and `KDDTest+.txt` (test)
- Features: columns 0-40; label at 41; difficulty at 42 (ignored)
- Preprocessing via ColumnTransformer:
  - OneHotEncoder for `protocol_type`, `service`, `flag` (cols 1,2,3)
  - StandardScaler for numeric columns

### Handling Class Imbalance
- `--class_weight balanced` for RF and SVM; stratified splits
- Consider SMOTE or focal loss for further improvement if needed

### Metrics and Feature Importance
- Console prints full per-class report; JSON summary saved
- For RF and ensemble with RF: prints top features using transformed feature names

### Notes on Accuracy Targets
- Achieving 99%+ 23-class accuracy on `KDDTest+.txt` is challenging; macro metrics (macro-F1) are more representative due to imbalance. Try the ensemble and tune hyperparameters for best results.

### Next Steps
- Add advanced resampling (SMOTE/ADASYN) and threshold tuning
- Hyperparameter search (GridSearchCV) with stratified folds
- Full-stack dashboard for monitoring, alerting, and forensics (ask for scaffold prompts)

## Full-Stack IDS Web App (Local)

### Prerequisites
- Node.js 20+
- MongoDB Community Server (or use Docker Compose)

### Backend (API)
```bash
cd backend
npm install
# create .env (see keys used in code):
# PORT=4000
# MONGO_URI=mongodb://localhost:27017/ids
# JWT_SECRET=devsecret
# ALLOWED_ORIGIN=http://localhost:5173
npm run dev
```
- Endpoints:
  - POST `/api/auth/login` { email, password }
  - GET `/api/alerts` query filters: label, minConfidence, severity, from, to, page, pageSize
  - GET `/api/alerts/stats/by-label` and `/api/alerts/stats/by-time?interval=hour|day`
  - GET `/api/alerts/export/csv`
  - GET `/api/events/stream` (SSE)
  - POST `/api/ml/batch` { inputPath } → runs Python `predict`, stores alerts, broadcasts SSE

### Frontend (React)
```bash
cd web
npm install
npm run dev
```
- Visit http://localhost:5173

### Optional: Docker Compose
```bash
docker compose up -d
```
- Spins up MongoDB and the backend on port 4000.

### Connect ML pipeline
- Train your model first (see earlier section), ensuring artifacts in `models/`.
- Use `/api/ml/batch` with `inputPath` (absolute or repo-relative) to ingest and broadcast predictions.

### Notes
- Authentication: simple JWT. Seed an admin user directly in Mongo or temporarily bypass `register` protection by creating the initial user via Mongo shell.
- SSE: front-end subscribes to `/api/events/stream` and updates the dashboard in real time.
- Performance: indexes applied on `label`, `occurredAt`, `confidence`; use filters and pagination for large datasets.


