# evaluation.py

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def collect_predictions(model, dataloader, criterion, device):
    """
    Runs model on a dataloader and returns loss, labels, predictions, and probabilities.

    Assumes binary classification using BCEWithLogitsLoss.
    """
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().long().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss, all_labels, all_preds, all_probs


def calculate_metrics(labels, preds):
    """
    Calculates standard binary classification metrics.
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds),
    }


def evaluate_model(model, dataloader, criterion, device):
    """
    Full evaluation wrapper.
    """
    loss, labels, preds, probs = collect_predictions(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    metrics = calculate_metrics(labels, preds)
    metrics["loss"] = loss

    return metrics, labels, preds, probs


def print_confusion_matrix(cm, class_names=None):
    """
    Pretty prints a binary confusion matrix.
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    print("\nConfusion Matrix")
    print("-" * 40)
    print(f"{'':15s}{'Pred 0':>10s}{'Pred 1':>10s}")
    print(f"{'True 0':15s}{cm[0][0]:10d}{cm[0][1]:10d}")
    print(f"{'True 1':15s}{cm[1][0]:10d}{cm[1][1]:10d}")


def print_metrics_report(title, metrics, class_names=None):
    """
    Prints a formatted metrics block.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")

    print_confusion_matrix(metrics["confusion_matrix"], class_names)


def print_epoch_report(
    epoch,
    num_epochs,
    train_metrics,
    val_metrics,
    class_names=None,
):
    """
    Prints a clean report for one epoch.
    """
    print("\n" + "#" * 70)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("#" * 70)

    print_metrics_report("Training Metrics", train_metrics, class_names)
    print_metrics_report("Validation Metrics", val_metrics, class_names)


def print_final_test_report(test_metrics, class_names=None):
    """
    Prints final test evaluation report.
    """
    print("\n" + "#" * 70)
    print("Final Test Set Evaluation")
    print("#" * 70)

    print_metrics_report("Test Metrics", test_metrics, class_names)


def print_classification_report(labels, preds, class_names=None):
    """
    Prints sklearn's detailed classification report.
    """
    print("\nDetailed Classification Report")
    print("-" * 60)

    print(
        classification_report(
            labels,
            preds,
            target_names=class_names,
            zero_division=0,
        )
    )