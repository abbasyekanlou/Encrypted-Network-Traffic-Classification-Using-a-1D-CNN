import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from torch import nn

from cesnet_datazoo.config import AppSelection, DatasetConfig
from cesnet_datazoo.datasets import CESNET_TLS_Year22

# 1. Basic settings
# ============================================================

NUMBER_OF_CLASSES = 5
NUMBER_OF_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001

device = torch.device("cpu")

print("Device:", device)

RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


def save_and_show_plot(filename):
    """
    Save the current figure in the results folder,
    display it, and then close it.
    """

    plt.tight_layout()

    plt.savefig(
        RESULTS_FOLDER / filename,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()
# 2. Load the real CESNET network dataset
# ============================================================

dataset = CESNET_TLS_Year22(
    data_root="data",
    size="XS",
)

config = DatasetConfig(
    dataset=dataset,

    # Select the five most frequent real applications.
    apps_selection=AppSelection.TOPX_KNOWN,
    apps_selection_topx=NUMBER_OF_CLASSES,

    # Return PyTorch tensors.
    return_tensors=True,

    batch_size=BATCH_SIZE,
    test_batch_size=256,

    # Zero workers is simpler on Windows.
    train_workers=0,
    val_workers=0,
    test_workers=0,

    # We only use the first 30 packets.
    use_packet_histograms=False,
    use_tcp_features=False,
)

dataset.set_dataset_config_and_initialize(config)

train_loader = dataset.get_train_dataloader()
validation_loader = dataset.get_val_dataloader()
test_loader = dataset.get_test_dataloader()

class_names = [
    str(name)
    for name in dataset.get_known_apps()
]

print("\nApplications used in the project:")

for class_number, class_name in enumerate(class_names):
    print(class_number, "=", class_name)


# ============================================================
# 3. Normalize one batch of packet data
# ============================================================

def prepare_packet_data(ppi):
    """
    PPI channel order:

    Channel 0 = inter-packet time
    Channel 1 = packet direction
    Channel 2 = packet size
    """

    ppi = ppi.float().clone()

    # Limit very large times and use logarithmic scaling.
    ppi[:, 0, :] = torch.clamp(
        ppi[:, 0, :],
        min=0,
        max=65000,
    )

    ppi[:, 0, :] = (
        torch.log1p(ppi[:, 0, :])
        / math.log1p(65000)
    )

    # Direction already uses -1, 0, and +1.
    # Therefore, we do not change channel 1.

    # Limit packet size and scale it between 0 and 1.
    ppi[:, 2, :] = torch.clamp(
        ppi[:, 2, :],
        min=0,
        max=1500,
    )

    ppi[:, 2, :] = ppi[:, 2, :] / 1500

    return ppi.to(device)


# ============================================================
# 4. Visualize one real network flow
# ============================================================

_, example_ppi, _, example_labels = next(iter(train_loader))

example_flow = example_ppi[0]
example_label = example_labels[0].item()
example_class_name = class_names[example_label]

packet_numbers = np.arange(
    1,
    example_flow.shape[1] + 1,
)

print("\nExample PPI shape:", example_flow.shape)
print("Example label:", example_class_name)


# ------------------------------------------------------------
# Inter-packet time
# ------------------------------------------------------------

plt.figure(figsize=(9, 4.5))

plt.plot(
    packet_numbers,
    example_flow[0].cpu().numpy(),
    marker="o",
    markersize=4,
    linewidth=1.8,
)

plt.title(
    f"Inter-Packet Time for a Real {example_class_name} Flow"
)

plt.xlabel("Packet number")
plt.ylabel("Inter-packet time (ms)")
plt.grid(True, alpha=0.3)

save_and_show_plot(
    "example_inter_packet_time.png"
)


# ------------------------------------------------------------
# Packet direction
# ------------------------------------------------------------

plt.figure(figsize=(9, 4.5))

plt.step(
    packet_numbers,
    example_flow[1].cpu().numpy(),
    where="mid",
    linewidth=1.8,
)

plt.title(
    f"Packet Direction for a Real {example_class_name} Flow"
)

plt.xlabel("Packet number")
plt.ylabel("Direction")
plt.yticks(
    [-1, 0, 1],
    ["Server → Client", "Padding", "Client → Server"],
)

plt.grid(True, alpha=0.3)

save_and_show_plot(
    "example_packet_direction.png"
)


# ------------------------------------------------------------
# Packet size
# ------------------------------------------------------------

plt.figure(figsize=(9, 4.5))

plt.plot(
    packet_numbers,
    example_flow[2].cpu().numpy(),
    marker="o",
    markersize=4,
    linewidth=1.8,
)

plt.title(
    f"Packet Size for a Real {example_class_name} Flow"
)

plt.xlabel("Packet number")
plt.ylabel("Transport payload size")
plt.grid(True, alpha=0.3)

save_and_show_plot(
    "example_packet_size.png"
)

# ============================================================
# 5. Create a simple CNN
# ============================================================

model = nn.Sequential(
    # Input shape: [batch, 3, 30]

    nn.Conv1d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1,
    ),

    nn.ReLU(),

    # Sequence length: 30 -> 15
    nn.MaxPool1d(kernel_size=2),

    nn.Conv1d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
    ),

    nn.ReLU(),

    # [batch, 32, 15] -> [batch, 32, 1]
    nn.AdaptiveAvgPool1d(1),

    # [batch, 32, 1] -> [batch, 32]
    nn.Flatten(),

    # [batch, 32] -> [batch, 5]
    nn.Linear(
        in_features=32,
        out_features=NUMBER_OF_CLASSES,
    ),
)

model = model.to(device)

print("\nCNN:")
print(model)


# ============================================================
# 6. Loss function and optimizer
# ============================================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# Lists used to create graphs later.
train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []


# ============================================================
# 7. Train the CNN
# ============================================================

for epoch in range(NUMBER_OF_EPOCHS):

    model.train()

    total_loss = 0
    correct_predictions = 0
    number_of_samples = 0

    for _, ppi, _, labels in train_loader:

        ppi = prepare_packet_data(ppi)
        labels = labels.long().to(device)

        # Remove the previous gradients.
        optimizer.zero_grad()

        # CNN prediction.
        logits = model(ppi)

        # Compare prediction with true label.
        loss = loss_function(logits, labels)

        # Calculate gradients.
        loss.backward()

        # Update the CNN parameters.
        optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        number_of_samples += labels.size(0)

    train_loss = total_loss / number_of_samples

    train_accuracy = (
        correct_predictions / number_of_samples
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    validation_total_loss = 0
    validation_correct = 0
    validation_samples = 0

    with torch.no_grad():

        for _, ppi, _, labels in validation_loader:
            ppi = prepare_packet_data(ppi)
            labels = labels.long().to(device)

            logits = model(ppi)

            validation_batch_loss = loss_function(
                logits,
                labels,
            )

            predictions = logits.argmax(dim=1)

            validation_total_loss += (
                    validation_batch_loss.item()
                    * labels.size(0)
            )

            validation_correct += (
                    predictions == labels
            ).sum().item()

            validation_samples += labels.size(0)

    validation_loss = (
            validation_total_loss
            / validation_samples
    )

    validation_accuracy = (
            validation_correct
            / validation_samples
    )

    validation_losses.append(
        validation_loss
    )

    validation_accuracies.append(
        validation_accuracy
    )

    print(
        f"Epoch {epoch + 1:2d}/{NUMBER_OF_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {validation_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation accuracy: {validation_accuracy:.2%}")


# ============================================================
# 8. Test the CNN
# ============================================================

model.eval()

test_correct = 0
test_samples = 0

true_labels = []
predicted_labels = []

with torch.no_grad():

    for _, ppi, _, labels in test_loader:

        ppi = prepare_packet_data(ppi)
        labels = labels.long().to(device)

        logits = model(ppi)

        predictions = logits.argmax(dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_samples += labels.size(0)

        true_labels.extend(
            labels.cpu().numpy()
        )

        predicted_labels.extend(
            predictions.cpu().numpy()
        )


test_accuracy = test_correct / test_samples

print("\nFinal test accuracy:", f"{test_accuracy:.2%}")


# ============================================================
# 9. Training and validation loss
# ============================================================

epoch_numbers = np.arange(
    1,
    NUMBER_OF_EPOCHS + 1,
)

marker_interval = max(
    1,
    NUMBER_OF_EPOCHS // 8,
)

plt.figure(figsize=(8.5, 5))

plt.plot(
    epoch_numbers,
    train_losses,
    label="Training loss",
    linewidth=2,
    marker="o",
    markersize=5,
    markevery=marker_interval,
)

plt.plot(
    epoch_numbers,
    validation_losses,
    label="Validation loss",
    linewidth=2,
    linestyle="--",
    marker="s",
    markersize=5,
    markevery=marker_interval,
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")

plt.gca().xaxis.set_major_locator(
    MaxNLocator(integer=True)
)

plt.grid(True, alpha=0.3)
plt.legend()

save_and_show_plot(
    "training_validation_loss.png"
)

# ============================================================
# 10. Training and validation accuracy
# ============================================================

plt.figure(figsize=(8.5, 5))

plt.plot(
    epoch_numbers,
    train_accuracies,
    label="Training accuracy",
    linewidth=2,
    marker="o",
    markersize=5,
    markevery=marker_interval,
)

plt.plot(
    epoch_numbers,
    validation_accuracies,
    label="Validation accuracy",
    linewidth=2,
    linestyle="--",
    marker="s",
    markersize=5,
    markevery=marker_interval,
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.ylim(0, 1.02)

plt.gca().yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.gca().xaxis.set_major_locator(
    MaxNLocator(integer=True)
)

plt.grid(True, alpha=0.3)
plt.legend()

save_and_show_plot(
    "training_validation_accuracy.png"
)
# ============================================================
# 11. Raw confusion matrix
# ============================================================

plt.figure(figsize=(8, 7))

axis = plt.gca()

ConfusionMatrixDisplay.from_predictions(
    true_labels,
    predicted_labels,
    display_labels=class_names,
    xticks_rotation=45,
    values_format="d",
    ax=axis,
)

plt.title(
    f"Test Confusion Matrix — Accuracy: {test_accuracy:.2%}"
)

plt.xlabel("Predicted application")
plt.ylabel("True application")

save_and_show_plot(
    "confusion_matrix_raw.png"
)


# ============================================================
# 12. Normalized confusion matrix
# ============================================================

plt.figure(figsize=(8, 7))

axis = plt.gca()

ConfusionMatrixDisplay.from_predictions(
    true_labels,
    predicted_labels,
    display_labels=class_names,
    normalize="true",
    values_format=".2f",
    xticks_rotation=45,
    ax=axis,
)

plt.title("Normalized Test Confusion Matrix")

plt.xlabel("Predicted application")
plt.ylabel("True application")

save_and_show_plot(
    "confusion_matrix_normalized.png"
)

# ============================================================
# 13. Classification report
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
)