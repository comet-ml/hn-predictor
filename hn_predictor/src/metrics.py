from sklearn.metrics import (
log_loss,
accuracy_score,
precision_score,
recall_score
)


def compute_metrics(predictions, target):
    return {
        "loss": log_loss(target, predictions),
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions),
        "recall": recall_score(target, predictions)
    }