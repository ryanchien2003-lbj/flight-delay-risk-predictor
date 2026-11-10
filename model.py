"""Train and evaluate a flight delay classifier using schedule-time features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "FlightDelays.csv"
VALIDATION_START = pd.Timestamp("2004-01-19")
TEST_START = pd.Timestamp("2004-01-25")
FEATURES = ["origin", "destination", "carrier", "weekday", "hour"]


def load_flights(path=DATA):
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["FL_DATE"], format="%m/%d/%Y")
    status = raw["Flight Status"].str.strip().str.lower()
    if not status.isin(["ontime", "delayed"]).all():
        raise ValueError("Unexpected or missing flight status label.")
    return pd.DataFrame({
        "date": dates,
        "origin": raw["ORIGIN"],
        "destination": raw["DEST"],
        "carrier": raw["CARRIER"],
        "weekday": dates.dt.dayofweek,
        "hour": raw["CRS_DEP_TIME"] // 100,
        "delayed": status.eq("delayed").astype(int),
    }).sort_values("date")


def candidates():
    estimators = {
        "Historical prevalence": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=500),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=12, random_state=42, n_jobs=2
        ),
    }
    return {
        name: make_pipeline(
            ColumnTransformer([("schedule", OneHotEncoder(handle_unknown="ignore"), FEATURES)]),
            estimator,
        )
        for name, estimator in estimators.items()
    }


def evaluate(model, rows):
    probabilities = model.predict_proba(rows[FEATURES])[:, 1]
    actual = rows["delayed"]
    return {
        "roc_auc": roc_auc_score(actual, probabilities),
        "average_precision": average_precision_score(actual, probabilities),
        "brier": brier_score_loss(actual, probabilities),
    }


def train(flights):
    train_rows = flights[flights["date"] < VALIDATION_START]
    validation = flights[flights["date"].between(VALIDATION_START, TEST_START, inclusive="left")]
    test = flights[flights["date"] >= TEST_START]
    development = flights[flights["date"] < TEST_START]
    models = candidates()
    validation_scores = {}
    for name, model in models.items():
        model.fit(train_rows[FEATURES], train_rows["delayed"])
        validation_scores[name] = evaluate(model, validation)
    selected = min(("Logistic regression", "Random forest"),
                   key=lambda name: validation_scores[name]["brier"])
    test_scores = {}
    for name, model in models.items():
        model.fit(development[FEATURES], development["delayed"])
        test_scores[name] = evaluate(model, test)
    winner = models[selected]
    importance = permutation_importance(
        winner, test[FEATURES], test["delayed"],
        scoring="neg_brier_score", n_repeats=10, random_state=42,
    )
    summary = {
        "rows": {"train": len(train_rows), "validation": len(validation), "test": len(test)},
        "test_delay_rate": test["delayed"].mean(),
        "selected": selected,
        "validation": validation_scores,
        "test": test_scores,
        "importance": dict(zip(FEATURES, importance.importances_mean)),
    }
    return winner, development, summary


if __name__ == "__main__":
    _, _, summary = train(load_flights())
    print(json.dumps(summary, indent=2))
