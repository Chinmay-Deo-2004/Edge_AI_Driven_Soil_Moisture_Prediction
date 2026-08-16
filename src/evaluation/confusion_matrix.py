from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true, y_pred, labels, output_path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Soil moisture confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
