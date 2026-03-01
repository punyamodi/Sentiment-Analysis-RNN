import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Negative", "Positive"]
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }
