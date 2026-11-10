import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
import os
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime
from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CoverFlow",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CoverFlow")
st.subheader("Uncertainty-Aware Financial Fraud Triage")

st.caption(
    "XGBoost + Segment-Conditional Conformal Prediction "
    "for auditable and workload-aware financial fraud monitoring"
)


# =========================================================
# SQLITE DATABASE — PERSISTENT ANALYST FEEDBACK & AUDIT LOG
# =========================================================
# By default the database is created beside app.py. Override with
# COVERFLOW_DB_PATH when deploying to a persistent volume.
DB_PATH = os.environ.get("COVERFLOW_DB_PATH", "coverflow_audit.db")
_db_parent = os.path.dirname(os.path.abspath(DB_PATH))
os.makedirs(_db_parent, exist_ok=True)


def get_db_connection():
    """Create a SQLite connection with safe defaults."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def db_session():
    """Yield a connection and always close it after the operation."""
    connection = get_db_connection()
    try:
        yield connection
        if connection.in_transaction:
            connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    """Create the batch, latest-feedback and append-only audit tables."""
    with db_session() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_key TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                transaction_count INTEGER NOT NULL DEFAULT 0,
                calibration_mode TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS batch_transactions (
                batch_key TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                amount REAL,
                amount_segment TEXT,
                device_type TEXT,
                fraud_probability REAL NOT NULL,
                non_fraud_probability REAL NOT NULL,
                prediction_set TEXT NOT NULL,
                model_decision TEXT NOT NULL,
                actual_label INTEGER,
                calibration_mode TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (batch_key, source_row),
                FOREIGN KEY (batch_key) REFERENCES batches(batch_key)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_transactions_txid
            ON batch_transactions(transaction_id)
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyst_feedback (
                batch_key TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                model_decision TEXT NOT NULL,
                prediction_set TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                analyst_decision TEXT NOT NULL,
                investigation_notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (batch_key, source_row),
                FOREIGN KEY (batch_key) REFERENCES batches(batch_key)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_key TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                model_decision TEXT NOT NULL,
                prediction_set TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                analyst_decision TEXT NOT NULL,
                investigation_notes TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL DEFAULT 'ASSESSMENT_SAVED',
                event_timestamp TEXT NOT NULL,
                FOREIGN KEY (batch_key) REFERENCES batches(batch_key)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_batch_row
            ON audit_events(batch_key, source_row, event_id)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_batch
            ON analyst_feedback(batch_key)
        """)


def register_batch(batch_key, file_name, file_size, transaction_count, calibration_mode):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with db_session() as connection:
        connection.execute("""
            INSERT INTO batches (
                batch_key, file_name, file_size, transaction_count,
                calibration_mode, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_key) DO UPDATE SET
                transaction_count = excluded.transaction_count,
                calibration_mode = excluded.calibration_mode,
                updated_at = excluded.updated_at
        """, (
            batch_key, file_name, int(file_size), int(transaction_count),
            calibration_mode, now, now
        ))


def save_analyst_assessment(feedback_record, batch_key, source_row):
    """Append an audit event and atomically update the latest assessment."""
    with db_session() as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute("""
            SELECT analyst_decision
            FROM analyst_feedback
            WHERE batch_key = ? AND source_row = ?
        """, (batch_key, int(source_row))).fetchone()

        event_type = (
            "ASSESSMENT_UPDATED" if previous is not None
            else "ASSESSMENT_CREATED"
        )

        values = (
            batch_key,
            int(source_row),
            str(feedback_record["Transaction ID"]),
            str(feedback_record["Model Decision"]),
            str(feedback_record["Prediction Set"]),
            float(feedback_record["Fraud Probability (%)"]),
            str(feedback_record["Analyst Decision"]),
            str(feedback_record["Investigation Notes"]),
            str(feedback_record["Timestamp"])
        )

        connection.execute("""
            INSERT INTO audit_events (
                batch_key, source_row, transaction_id, model_decision,
                prediction_set, fraud_probability, analyst_decision,
                investigation_notes, event_type, event_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values[:8] + (event_type, values[8]))

        connection.execute("""
            INSERT INTO analyst_feedback (
                batch_key, source_row, transaction_id, model_decision,
                prediction_set, fraud_probability, analyst_decision,
                investigation_notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_key, source_row) DO UPDATE SET
                transaction_id = excluded.transaction_id,
                model_decision = excluded.model_decision,
                prediction_set = excluded.prediction_set,
                fraud_probability = excluded.fraud_probability,
                analyst_decision = excluded.analyst_decision,
                investigation_notes = excluded.investigation_notes,
                updated_at = excluded.updated_at
        """, values)
        connection.commit()


def persist_batch_predictions(batch_key, batch_results, calibration_mode):
    """Persist model outputs separately from analyst decisions."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for result in batch_results:
        prediction_set = result.get("Prediction Set", [])
        rows.append((
            batch_key,
            int(result.get("_Source Row", 0)),
            str(result.get("Transaction ID", "")),
            float(result["Amount"]) if result.get("Amount") is not None else None,
            str(result.get("Amount Segment", "")),
            str(result.get("Device Type", "")),
            float(result["Fraud Probability"]),
            float(result["Non-Fraud Probability"]),
            json.dumps(list(prediction_set)),
            str(result.get("Decision", "")),
            int(result["Actual Label"]) if result.get("Actual Label") is not None else None,
            str(calibration_mode),
            now
        ))

    with db_session() as connection:
        connection.executemany("""
            INSERT INTO batch_transactions (
                batch_key, source_row, transaction_id, amount, amount_segment,
                device_type, fraud_probability, non_fraud_probability,
                prediction_set, model_decision, actual_label,
                calibration_mode, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_key, source_row) DO UPDATE SET
                transaction_id = excluded.transaction_id,
                amount = excluded.amount,
                amount_segment = excluded.amount_segment,
                device_type = excluded.device_type,
                fraud_probability = excluded.fraud_probability,
                non_fraud_probability = excluded.non_fraud_probability,
                prediction_set = excluded.prediction_set,
                model_decision = excluded.model_decision,
                actual_label = excluded.actual_label,
                calibration_mode = excluded.calibration_mode,
                created_at = excluded.created_at
        """, rows)


def search_persisted_transactions(search_term, limit=100):
    """Search historical predictions by transaction ID or batch filename."""
    term = f"%{search_term.strip()}%"
    with db_session() as connection:
        rows = connection.execute("""
            SELECT
                t.transaction_id AS "Transaction ID",
                b.file_name AS "Batch File",
                t.amount AS "Amount",
                t.fraud_probability * 100 AS "Fraud Probability (%)",
                t.amount_segment AS "Amount Segment",
                t.device_type AS "Device Type",
                t.prediction_set AS "Prediction Set",
                t.model_decision AS "Model Decision",
                f.analyst_decision AS "Latest Analyst Decision",
                f.updated_at AS "Analyst Decision Time",
                t.calibration_mode AS "Calibration Mode",
                t.created_at AS "Prediction Stored At"
            FROM batch_transactions t
            JOIN batches b ON b.batch_key = t.batch_key
            LEFT JOIN analyst_feedback f
                ON f.batch_key = t.batch_key
                AND f.source_row = t.source_row
            WHERE t.transaction_id LIKE ? OR b.file_name LIKE ?
            ORDER BY t.created_at DESC
            LIMIT ?
        """, (term, term, int(limit))).fetchall()
    return [dict(row) for row in rows]


def _feedback_row_to_dict(row):
    return {
        "Batch File": row["file_name"] if "file_name" in row.keys() else "",
        "Transaction ID": row["transaction_id"],
        "Source Row": row["source_row"],
        "Model Decision": row["model_decision"],
        "Prediction Set": row["prediction_set"],
        "Fraud Probability (%)": row["fraud_probability"],
        "Analyst Decision": row["analyst_decision"],
        "Investigation Notes": row["investigation_notes"],
        "Timestamp": row["updated_at"]
    }


def get_latest_feedback(batch_key, source_row):
    with db_session() as connection:
        row = connection.execute("""
            SELECT f.*, b.file_name
            FROM analyst_feedback f
            JOIN batches b ON b.batch_key = f.batch_key
            WHERE f.batch_key = ? AND f.source_row = ?
        """, (batch_key, int(source_row))).fetchone()
    return _feedback_row_to_dict(row) if row else None


def get_batch_feedback(batch_key):
    with db_session() as connection:
        rows = connection.execute("""
            SELECT f.*, b.file_name
            FROM analyst_feedback f
            JOIN batches b ON b.batch_key = f.batch_key
            WHERE f.batch_key = ?
            ORDER BY f.source_row
        """, (batch_key,)).fetchall()
    return [_feedback_row_to_dict(row) for row in rows]


def get_transaction_audit_history(batch_key, source_row):
    with db_session() as connection:
        rows = connection.execute("""
            SELECT transaction_id AS "Transaction ID",
                   source_row AS "Source Row",
                   model_decision AS "Model Decision",
                   prediction_set AS "Prediction Set",
                   fraud_probability AS "Fraud Probability (%)",
                   analyst_decision AS "Analyst Decision",
                   investigation_notes AS "Investigation Notes",
                   event_type AS "Event Type",
                   event_timestamp AS "Timestamp"
            FROM audit_events
            WHERE batch_key = ? AND source_row = ?
            ORDER BY event_id
        """, (batch_key, int(source_row))).fetchall()
    return [dict(row) for row in rows]


def get_batch_audit_history(batch_key):
    with db_session() as connection:
        rows = connection.execute("""
            SELECT transaction_id AS "Transaction ID",
                   source_row AS "Source Row",
                   model_decision AS "Model Decision",
                   prediction_set AS "Prediction Set",
                   fraud_probability AS "Fraud Probability (%)",
                   analyst_decision AS "Analyst Decision",
                   investigation_notes AS "Investigation Notes",
                   event_type AS "Event Type",
                   event_timestamp AS "Timestamp"
            FROM audit_events
            WHERE batch_key = ?
            ORDER BY event_id
        """, (batch_key,)).fetchall()
    return [dict(row) for row in rows]


def get_database_overview():
    with db_session() as connection:
        batches = connection.execute(
            "SELECT COUNT(*) FROM batches"
        ).fetchone()[0]
        feedback = connection.execute(
            "SELECT COUNT(*) FROM analyst_feedback"
        ).fetchone()[0]
        events = connection.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
        predictions = connection.execute(
            "SELECT COUNT(*) FROM batch_transactions"
        ).fetchone()[0]
    return batches, predictions, feedback, events


initialize_database()


# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():

    model = XGBClassifier()
    model.load_model("artifacts/xgboost_model.json")

    encoder = joblib.load("artifacts/encoder.pkl")
    imputer = joblib.load("artifacts/imputer.pkl")

    feature_columns = joblib.load(
        "artifacts/feature_columns.pkl"
    )

    categorical_cols = joblib.load(
        "artifacts/categorical_cols.pkl"
    )

    numeric_cols = joblib.load(
        "artifacts/numeric_cols.pkl"
    )

    qhat_class = joblib.load(
        "artifacts/qhat_class.pkl"
    )

    amount_qhat = joblib.load(
        "artifacts/amount_segment_qhat.pkl"
    )

    amount_config = joblib.load(
        "artifacts/amount_config.pkl"
    )

    device_qhat = joblib.load(
        "artifacts/device_segment_qhat.pkl"
    )

    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    return (
        model,
        encoder,
        imputer,
        feature_columns,
        categorical_cols,
        numeric_cols,
        qhat_class,
        amount_qhat,
        amount_config,
        device_qhat,
        metrics
    )


(
    model,
    encoder,
    imputer,
    feature_columns,
    categorical_cols,
    numeric_cols,
    qhat_class,
    amount_qhat,
    amount_config,
    device_qhat,
    metrics
) = load_artifacts()


# =========================================================
# LOAD DEMO TRANSACTIONS
# =========================================================
@st.cache_data
def load_demo_data():
    return pd.read_csv("demo_transactions.csv")


demo_df = load_demo_data()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_calibration_threshold(mode, amount_segment, device, class_id):

    if mode == "Class-Conditional":
        qhat = qhat_class[class_id]

    elif mode == "Amount Segment":
        qhat = amount_qhat[(amount_segment, class_id)]

    else:
        if device not in ["Unknown", "desktop", "mobile"]:
            device = "Unknown"

        key = (device, class_id)

        if key not in device_qhat:
            key = ("Unknown", class_id)

        if key not in device_qhat:
            raise ValueError(
                f"No device calibration threshold available for {device}, "
                f"class {class_id}"
            )

        qhat = device_qhat[key]

    return float(qhat)


def explain_ambiguous_transaction(
    probabilities,
    mode,
    amount_segment,
    device
):

    class_details = []

    class_names = {
        0: "Non-Fraud",
        1: "Fraud"
    }

    for class_id in [0, 1]:

        probability = float(probabilities[class_id])

        qhat = get_calibration_threshold(
            mode,
            amount_segment,
            device,
            class_id
        )

        threshold = 1 - qhat

        included = probability >= threshold

        margin = probability - threshold

        class_details.append({

            "Class": class_names[class_id],

            "Probability (%)":
                round(probability * 100, 2),

            "Required Threshold (%)":
                round(threshold * 100, 2),

            "Threshold Difference (%)":
                round(margin * 100, 2),

            "Included in Prediction Set":
                "Yes" if included else "No"

        })

    included_classes = [
        item["Class"]
        for item in class_details
        if item["Included in Prediction Set"] == "Yes"
    ]

    if len(included_classes) == 2:

        explanation = (
            "Both Fraud and Non-Fraud satisfy their respective "
            "conformal inclusion thresholds. Therefore, the system "
            "cannot exclude either class and routes this transaction "
            "for analyst review."
        )

    elif len(included_classes) == 0:

        explanation = (
            "Neither class satisfies its conformal inclusion threshold. "
            "The prediction set is empty, so the transaction requires "
            "manual investigation."
        )

    else:

        explanation = (
            f"Only {included_classes[0]} satisfies its conformal "
            "inclusion threshold."
        )

    return {
        "details": class_details,
        "included_classes": included_classes,
        "explanation": explanation
    }


def generate_investigation_recommendations(row, probability):

    recommendations = []

    amount = row.get("TransactionAmt", None)
    device = row.get("DeviceType", "Unknown")
    product = row.get("ProductCD", "Unknown")

    # 1. Transaction amount analysis
    if pd.notna(amount):

        if amount > amount_config["q2"]:

            recommendations.append(
                "HIGH AMOUNT: Examine whether this transaction "
                "amount is consistent with the customer's "
                "usual spending pattern."
            )

        elif amount <= amount_config["q1"]:

            recommendations.append(
                "LOW AMOUNT: Check whether this transaction "
                "is part of repeated small transactions "
                "or an unusual purchase pattern."
            )

        else:

            recommendations.append(
                "AMOUNT REVIEW: Compare this transaction "
                "with previous transactions to identify "
                "unusual spending behaviour."
            )

    # 2. Device analysis
    if pd.isna(device) or str(device).lower() == "unknown":

        recommendations.append(
            "UNKNOWN DEVICE: Verify the available device "
            "and identity information before approving "
            "the transaction."
        )

    elif str(device).lower() == "mobile":

        recommendations.append(
            "MOBILE DEVICE: Examine available device and "
            "transaction consistency information."
        )

    else:

        recommendations.append(
            "DESKTOP DEVICE: Verify that the transaction "
            "is consistent with the customer's known "
            "device activity."
        )

    # 3. Product analysis
    if pd.notna(product):

        recommendations.append(
            f"PRODUCT REVIEW: Examine transaction details "
            f"associated with product category {product}."
        )

    # 4. Fraud probability analysis
    if probability >= 0.70:

        recommendations.append(
            "ELEVATED FRAUD SCORE: Prioritize verification "
            "of the transaction and available supporting evidence."
        )

    else:

        recommendations.append(
            "UNCERTAIN FRAUD SCORE: Avoid making a decision "
            "based on probability alone. Review both possible "
            "classes and supporting transaction information."
        )

    return recommendations

def get_amount_segment(amount):

    if amount <= amount_config["q1"]:
        return "Low"

    elif amount <= amount_config["q2"]:
        return "Medium"

    return "High"


def preprocess_data(raw_df):

    X = raw_df.copy()

    if "TransactionID" in X.columns:
        X = X.drop(columns=["TransactionID"])

    if "isFraud" in X.columns:
        X = X.drop(columns=["isFraud"])

    # Add any missing features
    for col in feature_columns:

        if col not in X.columns:

            if col in categorical_cols:
                X[col] = "Unknown"
            else:
                X[col] = np.nan

    # Correct feature order
    X = X[feature_columns]

    # Categorical preprocessing
    X[categorical_cols] = (
        X[categorical_cols]
        .fillna("Unknown")
    )

    X[categorical_cols] = encoder.transform(
        X[categorical_cols]
    )

    # Numerical preprocessing
    X[numeric_cols] = imputer.transform(
        X[numeric_cols]
    )

    return X



def get_prediction_set(
    probabilities,
    mode,
    amount_segment,
    device
):

    prediction_set = []

    for c in [0, 1]:

        qhat = get_calibration_threshold(
            mode,
            amount_segment,
            device,
            c
        )

        threshold = 1 - qhat

        if probabilities[c] >= threshold:

            if c == 0:
                prediction_set.append("Non-Fraud")
            else:
                prediction_set.append("Fraud")

    return prediction_set


def get_decision(prediction_set):

    if len(prediction_set) == 2:
        return "Analyst Review"

    if prediction_set == ["Fraud"]:
        return "Fraud Alert"

    if prediction_set == ["Non-Fraud"]:
        return "Auto Resolve"

    # Safety fallback
    return "Analyst Review"


def analyze_transactions(raw_df, mode):

    X = preprocess_data(raw_df)

    probabilities = model.predict_proba(X)

    results = []

    for i in range(len(raw_df)):

        row = raw_df.iloc[i]

        # Amount
        if "TransactionAmt" in raw_df.columns:
            amount = row["TransactionAmt"]
        else:
            amount = 0

        amount_segment = get_amount_segment(
            amount
        )

        # Device
        if "DeviceType" in raw_df.columns:

            device = row["DeviceType"]

            if pd.isna(device):
                device = "Unknown"

            device = str(device)

        else:
            device = "Unknown"

        fraud_probability = probabilities[i, 1]

        prediction_set = get_prediction_set(
            probabilities[i],
            mode,
            amount_segment,
            device
        )

        decision = get_decision(
            prediction_set
        )

        transaction_id = (
            row["TransactionID"]
            if "TransactionID" in raw_df.columns
            else i + 1
        )

        actual_label = None

        if "isFraud" in raw_df.columns:
            actual_label = int(
                row["isFraud"]
            )

        results.append(
            {
                "Transaction ID": transaction_id,
                "Amount": amount,
                "Amount Segment": amount_segment,
                "Device Type": device,
                "Fraud Probability": fraud_probability,
                "Non-Fraud Probability": float(probabilities[i, 0]),
                "Prediction Set": prediction_set,
                "Decision": decision,
                "Actual Label": actual_label,
                "_Source Row": i
            }
        )

    return results


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ CoverFlow Settings")

mode = st.sidebar.selectbox(
    "Calibration Mode",
    [
        "Class-Conditional",
        "Amount Segment",
        "Device Segment"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Calibration Target

**95% Coverage**

Prediction sets represent model uncertainty.

Singleton sets can be automatically resolved,
while ambiguous sets are routed for analyst review.
"""
)


# =========================================================
# PERFORMANCE DASHBOARD
# =========================================================
st.markdown("## 📊 System Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "XGBoost ROC-AUC",
    f"{metrics['roc_auc']:.3f}"
)

c2.metric(
    "Class Coverage",
    f"{metrics['class_conditional_coverage'] * 100:.2f}%"
)

c3.metric(
    "Amount Coverage",
    f"{metrics['amount_segment_coverage'] * 100:.2f}%"
)

c4.metric(
    "Device Coverage",
    f"{metrics['device_segment_coverage'] * 100:.2f}%"
)

st.divider()


# =========================================================
# TRANSACTION ANALYSIS
# =========================================================
st.markdown("## 🔍 Transaction Triage")

input_mode = st.radio(
    "Choose transaction source",
    [
        "Demo Transactions",
        "Upload CSV"
    ],
    horizontal=True
)


# =========================================================
# DEMO MODE
# =========================================================
if input_mode == "Demo Transactions":

    st.info(
        "Select a real transaction sampled from the "
        "IEEE-CIS test split used during CoverFlow evaluation."
    )

    transaction_options = []

    for i in range(len(demo_df)):

        row = demo_df.iloc[i]

        transaction_id = row.get(
            "TransactionID",
            i + 1
        )

        actual = int(
            row["isFraud"]
        )

        label = (
            "Fraud"
            if actual == 1
            else "Non-Fraud"
        )

        transaction_options.append(
            f"Transaction {transaction_id} — Actual: {label}"
        )

    selected = st.selectbox(
        "Select Demo Transaction",
        range(len(transaction_options)),
        format_func=lambda x: transaction_options[x]
    )

    selected_df = demo_df.iloc[
        [selected]
    ].copy()
    
    if st.button("🔍 Find Ambiguous Demo Transactions"):

        with st.spinner("Checking demo transactions..."):

            all_results = analyze_transactions(
                demo_df,
                mode
            )

            ambiguous_rows = []

            for result in all_results:

                if len(result["Prediction Set"]) == 2:

                    ambiguous_rows.append({

                        "Transaction ID":
                            result["Transaction ID"],

                        "Amount":
                            result["Amount"],

                        "Device":
                            result["Device Type"],

                        "Fraud Probability (%)":
                            round(
                                result["Fraud Probability"] * 100,
                                2
                            ),

                        "Prediction Set":
                            "{Non-Fraud, Fraud}",

                        "Decision":
                            result["Decision"]

                    })

            if ambiguous_rows:

                st.success(
                    f"Found {len(ambiguous_rows)} ambiguous transactions!"
                )

                st.dataframe(
                    pd.DataFrame(ambiguous_rows),
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "Select any Transaction ID from this table "
                    "in the Demo Transaction dropdown and click "
                    "Analyze Transaction."
                )

            else:

                st.warning(
                    "No ambiguous transactions found in this demo "
                    "dataset for the selected calibration mode. "
                    "Try another calibration mode."
                )


    selected_row = selected_df.iloc[0]

    # Basic transaction information
    st.markdown("### Transaction Information")

    t1, t2, t3, t4 = st.columns(4)

    transaction_id = selected_row.get(
        "TransactionID",
        selected + 1
    )

    amount = selected_row.get(
        "TransactionAmt",
        0
    )

    device = selected_row.get(
        "DeviceType",
        "Unknown"
    )

    if pd.isna(device):
        device = "Unknown"

    product = selected_row.get(
        "ProductCD",
        "Unknown"
    )

    t1.metric(
        "Transaction ID",
        str(transaction_id)
    )

    t2.metric(
        "Transaction Amount",
        f"{amount:.2f}"
    )

    t3.metric(
        "Device",
        str(device)
    )

    t4.metric(
        "Product",
        str(product)
    )

    if st.button(
        "🔎 Analyze Transaction",
        type="primary",
        use_container_width=True
    ):

        results = analyze_transactions(
            selected_df,
            mode
        )

        result = results[0]

        st.divider()

        st.markdown(
            "## CoverFlow Decision"
        )

        probability = result[
            "Fraud Probability"
        ]

        prediction_set = result[
            "Prediction Set"
        ]

        decision = result[
            "Decision"
        ]

        actual = result[
            "Actual Label"
        ]

        # ---------------------------------------------
        # Probability / Segment Information
        # ---------------------------------------------
        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        r2.metric(
            "Amount Segment",
            result["Amount Segment"]
        )

        r3.metric(
            "Device Segment",
            result["Device Type"]
        )

        actual_text = (
            "Fraud"
            if actual == 1
            else "Non-Fraud"
        )

        r4.metric(
            "Actual Label",
            actual_text
        )

        st.markdown(
            "### Conformal Prediction Set"
        )

        if len(prediction_set) == 2:

            st.warning(
                "Prediction Set: "
                "{ Non-Fraud, Fraud }"
            )

            st.caption(
                "The system cannot confidently "
                "exclude either class."
            )

        elif prediction_set == ["Fraud"]:

            st.error(
                "Prediction Set: { Fraud }"
            )

        elif prediction_set == ["Non-Fraud"]:

            st.success(
                "Prediction Set: { Non-Fraud }"
            )

        else:

            st.warning(
                "Prediction Set is empty. "
                "Transaction is routed for review."
            )

        st.markdown(
            "### Final Triage Decision"
        )

        if decision == "Auto Resolve":

            st.success(
                "✅ AUTO RESOLVE — "
                "Transaction classified as confident non-fraud."
            )

        elif decision == "Fraud Alert":

            st.error(
                "🚨 FRAUD ALERT — "
                "Transaction requires fraud investigation."
            )

        else:

            st.warning(
                "⚠️ ANALYST REVIEW — "
                "Model uncertainty requires human review."
            )

        # Compare actual outcome
        if (
            actual == 1 and
            "Fraud" in prediction_set
        ) or (
            actual == 0 and
            "Non-Fraud" in prediction_set
        ):

            st.caption(
                "✓ Actual class is contained in the "
                "conformal prediction set."
            )

        else:

            st.caption(
                "Actual class is not contained in the "
                "prediction set (miscoverage case)."
            )
            
        # =========================================================
        # AMBIGUOUS TRANSACTION EXPLANATION
        # =========================================================

        if len(prediction_set) == 2:

            st.divider()

            st.markdown(
                "## 🧠 Why Does This Transaction Need Review?"
            )

            explanation = explain_ambiguous_transaction(
                [
                    result["Non-Fraud Probability"],
                    result["Fraud Probability"]
                ],
                mode,
                result["Amount Segment"],
                result["Device Type"]
            )

    # Display both class probabilities
            e1, e2 = st.columns(2)

            e1.metric(
                "Non-Fraud Probability",
                f"{result['Non-Fraud Probability'] * 100:.2f}%"
            )

            e2.metric(
                "Fraud Probability",
                f"{result['Fraud Probability'] * 100:.2f}%"
            )

            st.markdown("### Conformal Threshold Analysis")

            explanation_df = pd.DataFrame(
                explanation["details"]
            )

            st.dataframe(
                explanation_df,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "⚠️ Both classes are included in the prediction set."
            )

            st.markdown("### Explanation for Financial Analyst")

            st.info(explanation["explanation"])

            
            st.markdown("### Transaction-Specific Investigation Recommendations")

            recommendations = generate_investigation_recommendations(
                selected_row,
                result["Fraud Probability"]
            )

            for i, recommendation in enumerate(recommendations, 1):

                st.markdown(f"**{i}.** {recommendation}")


                st.caption(
                    "CoverFlow provides decision support. The final determination "
                    "must be made by an authorized human analyst."
                )

            
        

        


# =========================================================
# CSV UPLOAD MODE — BATCH TRIAGE & ANALYST WORKSPACE
# =========================================================
else:

    st.info(
        "Upload transaction data to run batch fraud triage, "
        "identify uncertain predictions and prepare an analyst review queue."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        help="The file should contain the transaction features used by the trained model."
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        upload_digest = hashlib.sha256(file_bytes).hexdigest()
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}:{upload_digest}"

        # Clear stale results whenever a different file is uploaded.
        if st.session_state.get("batch_upload_key") != upload_key:
            st.session_state["batch_upload_key"] = upload_key
            st.session_state["batch_raw_df"] = None
            st.session_state["batch_results"] = None
            st.session_state["batch_view"] = "All Transactions"
            st.session_state["batch_view_radio"] = "All Transactions"
            st.session_state["batch_selected_transaction"] = None

        try:
            raw_df = pd.read_csv(io.BytesIO(file_bytes))
            st.session_state["batch_raw_df"] = raw_df
            register_batch(
                upload_key,
                uploaded_file.name,
                uploaded_file.size,
                len(raw_df),
                mode
            )

            st.markdown("### Uploaded Transactions")
            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"{len(raw_df):,} transactions loaded from **{uploaded_file.name}**.")

            if raw_df.empty:
                st.warning("The uploaded CSV is empty. Please upload a file containing transactions.")
            else:
                if st.button(
                    "🚀 Analyze Transactions",
                    type="primary",
                    use_container_width=True
                ):
                    with st.spinner("Preprocessing transactions and generating conformal prediction sets..."):
                        batch_results = analyze_transactions(raw_df, mode)
                        persist_batch_predictions(
                            upload_key,
                            batch_results,
                            mode
                        )
                        st.session_state["batch_results"] = batch_results
                        st.session_state["batch_calibration_mode"] = mode
                        st.session_state["batch_view"] = "All Transactions"
                        st.session_state["batch_selected_transaction"] = None
                    st.success("Batch analysis completed. Your results are ready below.")

        except Exception as e:
            st.error("Unable to read or analyze the uploaded CSV.")
            st.exception(e)

        # Results remain visible on subsequent Streamlit reruns.
        batch_results = st.session_state.get("batch_results")
        stored_raw_df = st.session_state.get("batch_raw_df")

        if batch_results is not None and stored_raw_df is not None:
            results_df = pd.DataFrame(batch_results)

            def prediction_set_label(prediction_set):
                if len(prediction_set) == 2:
                    return "{Non-Fraud, Fraud}"
                if len(prediction_set) == 1:
                    return "{" + prediction_set[0] + "}"
                return "{}"

            # A clean display table; internal source-row references stay hidden.
            display_rows = []
            for result in batch_results:
                display_rows.append({
                    "Transaction ID": result["Transaction ID"],
                    "Amount": result["Amount"],
                    "Fraud Probability (%)": round(result["Fraud Probability"] * 100, 2),
                    "Amount Segment": result["Amount Segment"],
                    "Device Type": result["Device Type"],
                    "Prediction Set": prediction_set_label(result["Prediction Set"]),
                    "Decision": result["Decision"]
                })

            batch_display_df = pd.DataFrame(display_rows)

            # Queues are based on the model's conformal prediction set / triage decision.
            ambiguous_results = [
                r for r in batch_results if len(r["Prediction Set"]) == 2
            ]
            analyst_results = [
                r for r in batch_results if r["Decision"] == "Analyst Review"
            ]
            fraud_results = [
                r for r in batch_results if r["Decision"] == "Fraud Alert"
            ]
            auto_results = [
                r for r in batch_results if r["Decision"] == "Auto Resolve"
            ]

            total = len(batch_results)
            review_count = len(analyst_results)
            fraud_count = len(fraud_results)
            auto_count = len(auto_results)
            ambiguous_count = len(ambiguous_results)
            workload = (review_count / total * 100) if total else 0

            st.divider()
            st.markdown("## 📊 Batch Triage Summary")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Transactions", f"{total:,}")
            k2.metric("Auto Resolved", f"{auto_count:,}")
            k3.metric("Analyst Review", f"{review_count:,}")
            k4.metric("Fraud Alerts", f"{fraud_count:,}")
            k5.metric("Ambiguous Sets", f"{ambiguous_count:,}")

            st.metric("Current Batch Analyst Workload", f"{workload:.2f}%")
            st.progress(min(max(workload / 100, 0.0), 1.0))
            st.caption(
                "Analyst workload is the share of transactions whose conformal decision "
                "is Analyst Review. Fraud Alerts are shown in a separate investigation queue."
            )

            st.divider()
            st.markdown("## 🔎 Batch Transaction Explorer")

            view_options = [
                "All Transactions",
                "Ambiguous Transactions",
                "Analyst Review Queue",
                "Fraud Alerts",
                "Auto Resolved"
            ]

            if st.button("⚠️ Find Ambiguous Transactions", use_container_width=True):
                st.session_state["batch_view"] = "Ambiguous Transactions"
                st.session_state["batch_view_radio"] = "Ambiguous Transactions"

            current_view = st.session_state.get("batch_view", "All Transactions")
            if current_view not in view_options:
                current_view = "All Transactions"

            selected_view = st.radio(
                "Filter batch results",
                view_options,
                index=view_options.index(current_view),
                horizontal=True,
                key="batch_view_radio"
            )
            st.session_state["batch_view"] = selected_view

            view_map = {
                "All Transactions": batch_results,
                "Ambiguous Transactions": ambiguous_results,
                "Analyst Review Queue": analyst_results,
                "Fraud Alerts": fraud_results,
                "Auto Resolved": auto_results
            }
            filtered_results = view_map[selected_view]
            filtered_display = pd.DataFrame([
                {
                    "Transaction ID": r["Transaction ID"],
                    "Amount": r["Amount"],
                    "Fraud Probability (%)": round(r["Fraud Probability"] * 100, 2),
                    "Amount Segment": r["Amount Segment"],
                    "Device Type": r["Device Type"],
                    "Prediction Set": prediction_set_label(r["Prediction Set"]),
                    "Decision": r["Decision"]
                }
                for r in filtered_results
            ])

            st.markdown(f"### {selected_view} ({len(filtered_results):,})")
            if filtered_results:
                st.dataframe(
                    filtered_display,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(f"No records found in **{selected_view}** for this batch and calibration mode.")

            # Export full batch and focused queues.
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.download_button(
                    "⬇️ Download Full Batch Results",
                    data=batch_display_df.to_csv(index=False).encode("utf-8"),
                    file_name="coverflow_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with export_col2:
                review_export_df = pd.DataFrame([
                    {
                        "Transaction ID": r["Transaction ID"],
                        "Amount": r["Amount"],
                        "Fraud Probability (%)": round(r["Fraud Probability"] * 100, 2),
                        "Amount Segment": r["Amount Segment"],
                        "Device Type": r["Device Type"],
                        "Prediction Set": prediction_set_label(r["Prediction Set"]),
                        "Decision": r["Decision"]
                    }
                    for r in analyst_results
                ])
                st.download_button(
                    "⬇️ Download Analyst Review Queue",
                    data=review_export_df.to_csv(index=False).encode("utf-8"),
                    file_name="coverflow_analyst_review_queue.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=review_export_df.empty
                )

            # Investigation centre covers both ambiguous analyst-review cases and
            # singleton Fraud Alerts, because both need human investigation.
            investigation_results = analyst_results + fraud_results
            st.divider()
            st.markdown("## 🧑‍💼 Analyst Investigation Centre")
            st.caption(
                "Select a transaction to inspect its conformal thresholds and receive "
                "feature-aware investigation suggestions. These are decision-support "
                "checks, not proof that a transaction is fraudulent."
            )

            if investigation_results:
                option_labels = []
                result_by_label = {}
                for idx, result in enumerate(investigation_results):
                    label = (
                        f"{result['Transaction ID']} | {result['Decision']} | "
                        f"Fraud score: {result['Fraud Probability'] * 100:.2f}%"
                    )
                    # Include the ordinal to keep labels unique if IDs repeat.
                    unique_label = f"{idx + 1}. {label}"
                    option_labels.append(unique_label)
                    result_by_label[unique_label] = result

                chosen_label = st.selectbox(
                    "Choose a transaction to investigate",
                    option_labels,
                    key="batch_investigation_selector"
                )
                chosen_result = result_by_label[chosen_label]
                source_row_index = chosen_result["_Source Row"]
                selected_source_row = stored_raw_df.iloc[source_row_index]

                status = chosen_result["Decision"]
                if status == "Fraud Alert":
                    st.error(f"🚨 {status}")
                else:
                    st.warning(f"⚠️ {status}")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Transaction ID", str(chosen_result["Transaction ID"]))
                a2.metric("Fraud Probability", f"{chosen_result['Fraud Probability'] * 100:.2f}%")
                a3.metric("Amount Segment", chosen_result["Amount Segment"])
                a4.metric("Device Type", chosen_result["Device Type"])

                st.markdown("### Conformal Prediction Set")
                st.info(f"Prediction Set: {prediction_set_label(chosen_result['Prediction Set'])}")

                explanation = explain_ambiguous_transaction(
                    [
                        chosen_result["Non-Fraud Probability"],
                        chosen_result["Fraud Probability"]
                    ],
                    mode,
                    chosen_result["Amount Segment"],
                    chosen_result["Device Type"]
                )

                st.markdown("### Conformal Threshold Analysis")
                st.dataframe(
                    pd.DataFrame(explanation["details"]),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Why Was This Transaction Flagged?")
                if len(chosen_result["Prediction Set"]) == 2:
                    st.warning(explanation["explanation"])
                elif len(chosen_result["Prediction Set"]) == 0:
                    st.warning(
                        "Neither class met its conformal inclusion threshold. "
                        "The empty prediction set is routed to human review."
                    )
                else:
                    st.info(explanation["explanation"])

                st.markdown("### Transaction-Specific Investigation Recommendations")
                recommendations = generate_investigation_recommendations(
                    selected_source_row,
                    chosen_result["Fraud Probability"]
                )
                for rec_index, recommendation in enumerate(recommendations, start=1):
                    st.markdown(f"**{rec_index}.** {recommendation}")

                st.caption(
                    "CoverFlow provides decision support only. An authorized financial "
                    "analyst must verify supporting evidence and make the final determination."
                )

                # ---------------------------------------------------------
                # ANALYST FEEDBACK + AUDIT TRAIL
                # ---------------------------------------------------------
                st.markdown("### 📝 Record Analyst Decision")
                st.caption(
                    "Record the analyst's independent assessment and supporting notes. "
                    "This feedback does not alter or retrain the fraud model."
                )

                feedback_key = (
                    f"{st.session_state.get('batch_upload_key', 'batch')}:"
                    f"{source_row_index}"
                )
                previous_feedback = get_latest_feedback(
                    st.session_state.get("batch_upload_key", ""),
                    source_row_index
                ) or {}
                feedback_options = [
                    "Confirmed Fraud",
                    "Confirmed Legitimate",
                    "Escalate for Further Review"
                ]
                previous_choice = previous_feedback.get("Analyst Decision")
                default_index = (
                    feedback_options.index(previous_choice)
                    if previous_choice in feedback_options else 0
                )

                with st.form(
                    key=f"analyst_feedback_form_{source_row_index}",
                    clear_on_submit=False
                ):
                    analyst_decision = st.radio(
                        "Analyst assessment",
                        feedback_options,
                        index=default_index,
                        horizontal=True,
                        key=f"analyst_decision_{source_row_index}"
                    )
                    analyst_notes = st.text_area(
                        "Investigation notes",
                        value=previous_feedback.get("Investigation Notes", ""),
                        placeholder=(
                            "Record evidence checked, verification performed, "
                            "and the reason for your assessment..."
                        ),
                        max_chars=2000,
                        key=f"analyst_notes_{source_row_index}"
                    )
                    feedback_submitted = st.form_submit_button(
                        "💾 Save Analyst Decision",
                        type="primary",
                        use_container_width=True
                    )

                if feedback_submitted:
                    event_time = datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    )
                    feedback_record = {
                        "Batch File": st.session_state.get("batch_upload_key", ""),
                        "Transaction ID": chosen_result["Transaction ID"],
                        "Source Row": source_row_index,
                        "Model Decision": chosen_result["Decision"],
                        "Prediction Set": prediction_set_label(
                            chosen_result["Prediction Set"]
                        ),
                        "Fraud Probability (%)": round(
                            chosen_result["Fraud Probability"] * 100, 2
                        ),
                        "Analyst Decision": analyst_decision,
                        "Investigation Notes": analyst_notes.strip(),
                        "Timestamp": event_time
                    }

                    # Persist the latest assessment and append an immutable audit event.
                    save_analyst_assessment(
                        feedback_record,
                        st.session_state.get("batch_upload_key", ""),
                        source_row_index
                    )
                    st.success(
                        f"Analyst decision permanently saved for transaction "
                        f"{chosen_result['Transaction ID']}."
                    )

                # Display the latest saved decision and the event history for this row.
                latest_feedback = get_latest_feedback(
                    st.session_state.get("batch_upload_key", ""),
                    source_row_index
                )
                if latest_feedback:
                    st.markdown("#### Latest Saved Analyst Assessment")
                    st.write(
                        f"**Assessment:** {latest_feedback['Analyst Decision']}"
                    )
                    st.write(
                        f"**Saved:** {latest_feedback['Timestamp']}"
                    )
                    if latest_feedback["Investigation Notes"]:
                        st.write(
                            f"**Notes:** {latest_feedback['Investigation Notes']}"
                        )

                transaction_history = get_transaction_audit_history(
                    st.session_state.get("batch_upload_key", ""),
                    source_row_index
                )
                if transaction_history:
                    with st.expander("View transaction audit history"):
                        st.dataframe(
                            pd.DataFrame(transaction_history),
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.success("There are no transactions requiring investigation in this batch.")

            st.divider()
            st.markdown("## 📚 Batch Analyst Feedback & Audit Trail")

            current_batch_key = st.session_state.get("batch_upload_key", "")
            current_batch_audit = get_batch_audit_history(current_batch_key)
            current_batch_feedback = get_batch_feedback(current_batch_key)

            audit_col1, audit_col2 = st.columns(2)
            audit_col1.metric(
                "Transactions with Saved Feedback",
                len(current_batch_feedback)
            )
            audit_col2.metric(
                "Audit Events Recorded",
                len(current_batch_audit)
            )

            if current_batch_feedback:
                st.markdown("### Latest Analyst Assessments")
                st.dataframe(
                    pd.DataFrame(current_batch_feedback),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "No analyst decisions have been recorded for this batch yet. "
                    "Select a transaction above and submit an assessment."
                )

            if current_batch_audit:
                audit_df = pd.DataFrame(current_batch_audit)
                st.download_button(
                    "⬇️ Download Batch Audit Trail",
                    data=audit_df.to_csv(index=False).encode("utf-8"),
                    file_name="coverflow_batch_audit_trail.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button(
                    "⬇️ Download Batch Audit Trail",
                    disabled=True,
                    use_container_width=True
                )

            st.caption(
                f"Persistent storage: batch predictions, latest analyst assessments and append-only audit events "
                f"are stored in SQLite ({DB_PATH}). The database is local to this "
                f"application environment; use a persistent disk/volume when deploying "
                f"so records survive redeployments. Authentication, role-based access "
                f"control, encryption and tamper-evident logging are separate production "
                f"hardening steps."
            )

            with st.expander("🗄️ Database Storage & Historical Transaction Search"):
                db_batches, db_predictions, db_feedback, db_events = get_database_overview()
                db1, db2, db3, db4 = st.columns(4)
                db1.metric("Stored Batches", db_batches)
                db2.metric("Stored Predictions", db_predictions)
                db3.metric("Latest Assessments", db_feedback)
                db4.metric("Audit Events", db_events)
                st.caption(
                    "Counts cover the SQLite database, not only the currently selected batch. "
                    "Model predictions and analyst decisions are stored separately."
                )

                history_query = st.text_input(
                    "Search historical transactions by Transaction ID or batch filename",
                    placeholder="Enter a transaction ID or part of a CSV filename",
                    key="database_history_search"
                )
                if history_query.strip():
                    historical_rows = search_persisted_transactions(history_query)
                    if historical_rows:
                        st.dataframe(
                            pd.DataFrame(historical_rows),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No stored transactions matched that search.")

            st.divider()
            st.markdown("### Batch Decision Notes")
            st.caption(
                "Auto Resolve means the conformal set contains only Non-Fraud. "
                "Fraud Alert means it contains only Fraud. Analyst Review includes "
                "ambiguous two-class sets and empty prediction sets. A Fraud Alert is "
                "a model-generated alert, not a confirmed fraud finding."
            )

    else:
        st.caption("Upload a CSV file to activate batch analysis.")


# =========================================================
# OPERATIONAL MONITORING
# =========================================================
st.divider()

st.markdown(
    "## 📈 Coverage & Workload Monitoring"
)

comparison = pd.DataFrame(
    {
        "Calibration Method": [
            "Class Conditional",
            "Amount Segment",
            "Device Segment"
        ],

        "Coverage (%)": [
            metrics[
                "class_conditional_coverage"
            ] * 100,

            metrics[
                "amount_segment_coverage"
            ] * 100,

            metrics[
                "device_segment_coverage"
            ] * 100
        ],

        "Mean Set Size": [
            metrics[
                "class_conditional_mean_set_size"
            ],

            metrics[
                "amount_segment_mean_set_size"
            ],

            metrics[
                "device_segment_mean_set_size"
            ]
        ],

        "Analyst Workload (%)": [
            metrics[
                "class_conditional_workload"
            ],

            metrics[
                "amount_segment_workload"
            ],

            metrics[
                "device_segment_workload"
            ]
        ]
    }
)

st.dataframe(
    comparison.style.format(
        {
            "Coverage (%)": "{:.2f}",
            "Mean Set Size": "{:.3f}",
            "Analyst Workload (%)": "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# WORKLOAD CHART
# =========================================================
st.markdown(
    "### Analyst Workload by Calibration Method"
)

workload_chart = (
    comparison[
        [
            "Calibration Method",
            "Analyst Workload (%)"
        ]
    ]
    .set_index(
        "Calibration Method"
    )
)

st.bar_chart(
    workload_chart
)


# =========================================================
# COVERAGE CHART
# =========================================================
st.markdown(
    "### Coverage by Calibration Method"
)

coverage_chart = (
    comparison[
        [
            "Calibration Method",
            "Coverage (%)"
        ]
    ]
    .set_index(
        "Calibration Method"
    )
)

st.bar_chart(
    coverage_chart
)


# =========================================================
# PROJECT EXPLANATION
# =========================================================
st.divider()

with st.expander(
    "ℹ️ How CoverFlow Works"
):

    st.markdown(
        """
### 1. Fraud Detection

An **XGBoost classifier** estimates the probability
that a financial transaction is fraudulent.

### 2. Conformal Calibration

CoverFlow applies **conformal prediction** to convert
model probabilities into prediction sets with a target
coverage level of approximately **95%**.

### 3. Segment-Conditional Calibration

Coverage can be evaluated and calibrated across
operationally meaningful groups such as:

- Transaction amount bands
- Device type

This helps expose uncertainty that may be hidden by
aggregate metrics.

### 4. Workload-Aware Triage

CoverFlow converts prediction-set uncertainty into
operational decisions:

**{Non-Fraud} → Auto Resolve**

**{Fraud} → Fraud Alert**

**{Non-Fraud, Fraud} → Analyst Review**

This allows the system to balance predictive
uncertainty with human analyst workload.
"""
    )


st.caption(
    "CoverFlow — Segment-Conditional Conformal Prediction "
    "for Auditable, Workload-Aware Fraud Triage"
)