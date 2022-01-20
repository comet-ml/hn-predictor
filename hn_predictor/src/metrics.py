from sklearn.metrics import (
confusion_matrix,
log_loss,
accuracy_score,
precision_score,
recall_score
)


def compute_metrics(predictions, target):
    return {
        # "confusion_matrix": confusion_matrix(target, predictions),
        "loss": log_loss(target, predictions),
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions),
        "recall": recall_score(target, predictions)
    }