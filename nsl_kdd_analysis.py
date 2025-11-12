import argparse
import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from joblib import dump, load


def load_nsl_kdd(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load NSL-KDD train and test files.

    Files are expected in CSV-like text with no headers. NSL-KDD has 42 feature columns,
    followed by a label column (index 41), and a difficulty score at the end (index 42).
    """
    column_count = 43  # 41 features (0-40), label (41), difficulty (42)
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    if train_df.shape[1] != column_count or test_df.shape[1] != column_count:
        raise ValueError(
            f"Unexpected column count. Expected {column_count}, got {train_df.shape[1]} (train) and {test_df.shape[1]} (test)."
        )

    X_train = train_df.iloc[:, :-2]
    y_train = train_df.iloc[:, 41]
    X_test = test_df.iloc[:, :-2]
    y_test = test_df.iloc[:, 41]
    return X_train, y_train, X_test, y_test


def build_pipeline(model_type: str = "rf", class_weight: Optional[str] = "balanced", svm_kernel: str = "rbf") -> Pipeline:
    """Create a preprocessing + model pipeline.

    - OneHot encode categorical columns (1,2,3): protocol_type, service, flag
    - Standard scale numeric features
    - RandomForest classifier
    """
    categorical_columns = [1, 2, 3]
    numeric_columns = [i for i in range(0, 41) if i not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("num", StandardScaler(with_mean=False), numeric_columns),
        ]
    )

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight=class_weight,
        )
    elif model_type == "svm":
        model = SVC(kernel=svm_kernel, probability=True, class_weight=class_weight, random_state=42)
    elif model_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=50, random_state=42)
    elif model_type == "ensemble":
        rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42, class_weight=class_weight)
        sv = SVC(kernel=svm_kernel, probability=True, class_weight=class_weight, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=50, random_state=42)
        model = VotingClassifier(estimators=[("rf", rf), ("svm", sv), ("mlp", mlp)], voting="soft")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def train_and_evaluate(train_path: str, test_path: str, model_out: str, label_map_out: str, model_type: str, class_weight: Optional[str], svm_kernel: str, report_out: Optional[str]) -> None:
    X_train, y_train_raw, X_test, y_test_raw = load_nsl_kdd(train_path, test_path)

    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Label distribution (train):")
    print(y_train_raw.value_counts())

    # Encode target labels (fit on union of train+test to handle rare/unseen labels in test)
    target_encoder = LabelEncoder()
    target_encoder.fit(pd.concat([y_train_raw, y_test_raw], axis=0))
    y_train = target_encoder.transform(y_train_raw)
    y_test = target_encoder.transform(y_test_raw)

    pipeline = build_pipeline(model_type=model_type, class_weight=class_weight, svm_kernel=svm_kernel)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(target_encoder.classes_))),
            target_names=target_encoder.classes_,
            zero_division=0,
        )
    )

    # Macro/weighted metrics summary
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    summary = {
        "model_type": model_type,
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "classes": [str(c) for c in target_encoder.classes_],
    }

    # Save model and label encoder mapping
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    dump(pipeline, model_out)
    dump({"classes_": target_encoder.classes_}, label_map_out)
    print(f"Saved model to {model_out}")
    print(f"Saved label map to {label_map_out}")
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        pd.Series(summary).to_json(report_out, indent=2)
        print(f"Saved report to {report_out}")

    # Feature importance for RF-based models
    try:
        preprocess = pipeline.named_steps["preprocess"]
        feature_names = preprocess.get_feature_names_out()
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-20:][::-1]
            print("\nTop 20 features:")
            for idx in top_idx:
                print(f"{feature_names[idx]}: {importances[idx]:.5f}")
    except Exception as e:
        print(f"Feature importance unavailable: {e}")


def predict_file(model_path: str, label_map_path: str, input_path: str, output_path: Optional[str] = None, alert_threshold: Optional[float] = None) -> pd.DataFrame:
    pipeline: Pipeline = load(model_path)
    label_info = load(label_map_path)
    classes: np.ndarray = label_info["classes_"]

    df = pd.read_csv(input_path, header=None)
    # Handle case where input has a header row (e.g., from live_capture.py)
    try:
        if df.shape[0] > 0 and str(df.iloc[0, 0]).strip().lower() == "duration":
            df = df.iloc[1:, :].reset_index(drop=True)
    except Exception:
        pass
    if df.shape[1] < 43:
        # If only features present, assume up to difficulty missing and pad
        # Expecting 41 features + label + difficulty. We'll slice features correctly.
        pass

    X = df.iloc[:, :41] if df.shape[1] >= 41 else df
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        max_prob = probs.max(axis=1)
    else:
        preds = pipeline.predict(X)
        max_prob = np.ones(len(preds))
    pred_labels = pd.Series(classes[preds], name="predicted_label")
    conf = pd.Series(max_prob, name="confidence")

    result = pd.concat([df.reset_index(drop=True), pred_labels, conf], axis=1)
    if output_path:
        result.to_csv(output_path, index=False, header=False)
        print(f"Wrote predictions to {output_path}")
    return result


def watch_file(model_path: str, label_map_path: str, input_path: str, poll_interval: float = 1.0, alert_label: Optional[str] = None, min_confidence: float = 0.5) -> None:
    pipeline: Pipeline = load(model_path)
    label_info = load(label_map_path)
    classes: np.ndarray = label_info["classes_"]
    last_rows = 0
    print(f"Watching {input_path} for new rows...")
    while True:
        try:
            df = pd.read_csv(input_path, header=None)
        except Exception:
            df = pd.DataFrame()
        if df.shape[0] > last_rows:
            new = df.iloc[last_rows:]
            X = new.iloc[:, :41]
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                probs = pipeline.predict_proba(X)
                preds = np.argmax(probs, axis=1)
                max_prob = probs.max(axis=1)
            else:
                preds = pipeline.predict(X)
                max_prob = np.ones(len(preds))
            labels = classes[preds]
            for i, (lbl, conf) in enumerate(zip(labels, max_prob)):
                is_alert = (alert_label is None or lbl != alert_label) and conf >= min_confidence
                tag = "ALERT" if is_alert else "INFO"
                print(f"{tag} row={last_rows + i} label={lbl} confidence={conf:.3f}")
            last_rows = df.shape[0]
        import time
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="NSL-KDD Intrusion Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train model on NSL-KDD")
    train_p.add_argument("--train_path", default="KDDTrain+.txt")
    train_p.add_argument("--test_path", default="KDDTest+.txt")
    train_p.add_argument("--model_out", default="models/nsl_kdd_model.joblib")
    train_p.add_argument("--label_map_out", default="models/label_map.joblib")
    train_p.add_argument("--model_type", choices=["rf", "svm", "mlp", "ensemble"], default="rf")
    train_p.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    train_p.add_argument("--svm_kernel", choices=["rbf", "linear"], default="rbf")
    train_p.add_argument("--report_out", default="reports/metrics.json")

    pred_p = subparsers.add_parser("predict", help="Predict using a saved model")
    pred_p.add_argument("--model_path", default="models/nsl_kdd_model.joblib")
    pred_p.add_argument("--label_map_path", default="models/label_map.joblib")
    pred_p.add_argument("--input_path", required=True)
    pred_p.add_argument("--output_path", default=None)
    pred_p.add_argument("--alert_threshold", type=float, default=None)

    watch_p = subparsers.add_parser("watch", help="Watch a CSV appending file and emit alerts")
    watch_p.add_argument("--model_path", default="models/nsl_kdd_model.joblib")
    watch_p.add_argument("--label_map_path", default="models/label_map.joblib")
    watch_p.add_argument("--input_path", required=True)
    watch_p.add_argument("--poll_interval", type=float, default=1.0)
    watch_p.add_argument("--alert_label", default="normal")
    watch_p.add_argument("--min_confidence", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "train":
        cw = None if args.class_weight == "none" else "balanced"
        train_and_evaluate(args.train_path, args.test_path, args.model_out, args.label_map_out, args.model_type, cw, args.svm_kernel, args.report_out)
    elif args.command == "predict":
        predict_file(args.model_path, args.label_map_path, args.input_path, args.output_path, args.alert_threshold)
    elif args.command == "watch":
        watch_file(args.model_path, args.label_map_path, args.input_path, args.poll_interval, args.alert_label, args.min_confidence)


if __name__ == "__main__":
    main()
