# Encrypted Network Traffic Classification with a 1D CNN

This project uses a one-dimensional convolutional neural network (CNN) in
PyTorch to classify encrypted network traffic from packet metadata.

The model does **not** inspect encrypted payload content. It uses the first 30
packets of each real network flow and learns patterns from:

- inter-packet time;
- packet direction; and
- transport-payload size.

## Project objective

The goal is to investigate whether a small CNN can distinguish encrypted
applications from packet behavior alone. The current experiment uses the five
most frequent selected classes from the CESNET-TLS-Year22 configuration:
AppNexus, DNS over HTTPS, Snapchat, Spotify, and TikTok.

## Dataset

The project uses **CESNET-TLS-Year22**, a real year-spanning encrypted TLS
traffic dataset collected on the CESNET backbone network. The dataset contains
180 web-service labels and packet sequences describing the first 30 packets of
each flow.

Dataset paper:

> K. Hynek, J. Luxemburk, J. Pešek, T. Čejka, and P. Šiška,
> “CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from
> backbone lines,” *Scientific Data*, vol. 11, article 1156, 2024.
> DOI: https://doi.org/10.1038/s41597-024-03927-4

CESNET DataZoo documentation:

- https://cesnet.github.io/cesnet-datazoo/
- https://github.com/CESNET/cesnet-datazoo

The dataset itself is **not included** in this repository. The script accesses
the `XS` packaged version through CESNET DataZoo. The `data/` folder is excluded
from Git by `.gitignore`.

## Input representation

Every sample has this shape:

```text
[3 channels, 30 packets]
```

The channels are:

```text
Channel 0: inter-packet time (IPT)
Channel 1: packet direction (DIR)
Channel 2: packet size (SIZE)
```

A training batch has shape:

```text
[batch size, 3, 30]
```

## Preprocessing

- Inter-packet time is clipped and logarithmically scaled.
- Packet direction remains encoded as `-1`, `0`, or `+1`.
- Packet size is clipped at 1500 and divided by 1500.

## CNN architecture

```text
Input: 3 × 30
    ↓
Conv1D: 3 → 16 channels
    ↓
ReLU
    ↓
MaxPool1D
    ↓
Conv1D: 16 → 32 channels
    ↓
ReLU
    ↓
Global average pooling
    ↓
Linear classifier: 32 → 5 classes
```

The model is intentionally small so that the complete training process is easy
to understand.

## Baseline result

The current baseline experiment achieved approximately **76.14% test
accuracy**.

Approximate recall by class:

| Class | Recall |
|---|---:|
| AppNexus | 87.45% |
| DNS over HTTPS | 82.63% |
| Snapchat | 67.22% |
| Spotify | 27.61% |
| TikTok | 86.23% |

Spotify is the weakest class. Many Spotify flows were predicted as Snapchat,
which suggests that their first-30-packet patterns overlap under this simple
feature set and architecture.

## Current confusion matrix

![Test confusion matrix](results/confusion_matrix.png)

## Installation

CESNET DataZoo 0.2.0 requires Python 3.10 or newer. Create a virtual
environment and install the requirements.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project

```bash
python simple_real_cnn.py
```

The first run may take longer because dataset files must be downloaded and
prepared.

The script will:

1. load real encrypted traffic;
2. select five application classes;
3. train a small 1D CNN;
4. print the test accuracy and classification report;
5. save the model locally; and
6. save plots in `results/`.

## Generated result files

After running the code, the `results/` folder contains:

- `example_inter_packet_time.png`;
- `example_packet_direction.png`;
- `example_packet_size.png`;
- `training_loss.png`;
- `accuracy_curve.png`;
- `confusion_matrix.png`; and
- `normalized_confusion_matrix.png`.

The model file `traffic_cnn.pt` is excluded from Git because trained models can
be large and can be regenerated from the code.

## Limitations

- Only the first 30 packets are used.
- Only five classes are considered.
- The CNN uses only packet timing, direction, and size.
- The classes are imbalanced.
- Global average pooling removes some packet-position information.
- The system assumes that every test flow belongs to a known class.

## Planned improvements

- weighted cross-entropy for class imbalance;
- more training epochs and early stopping;
- batch normalization;
- preservation of packet-position information;
- macro F1-score as a model-selection metric;
- unknown-application rejection;
- temporal-drift analysis; and
- comparison with LSTM and classical machine-learning baselines.

## Repository structure

```text
encrypted-traffic-cnn-cesnet/
├── simple_real_cnn.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── GITHUB_UPLOAD.md
└── results/
    ├── confusion_matrix.png
    ├── normalized_confusion_matrix.png
    ├── class_recall.png
    └── README.md
```

## License

The source code in this repository is released under the MIT License. The
CESNET dataset and CESNET DataZoo package remain subject to their own licenses
and citation requirements.
