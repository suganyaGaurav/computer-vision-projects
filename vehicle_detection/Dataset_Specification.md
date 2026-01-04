Dataset Specification

This project expects spectrogram data derived from radio signals.

The dataset is not included in this repository.
Users may use any equivalent spectrogram dataset that matches the following specification.

🔹 Data Representation

Each spectrogram represents:

A 2D time–frequency representation of a radio signal

Converted into a grayscale image

Flattened and stored as a CSV row

🔹 Input Image Shape
Property	Value
Height	64 pixels
Width	128 pixels
Channels	1 (grayscale)
Flattened length	8192 values (64 × 128)

Each row in images.csv must contain 8192 numeric values.

🔹 Label Format

Labels must be one-hot encoded.

Class Index	Class Name
0	Squiggle
1	Narrowband
2	Narrowband Drift
3	Noise

Each row in labels.csv must contain 4 values.

Example:

[0, 1, 0, 0] → Narrowband

🔹 Expected Directory Structure
data/
├── train/
│   ├── images.csv
│   └── labels.csv
└── valid/
    ├── images.csv
    └── labels.csv

🔹 CSV Constraints

No headers

Numeric values only

Consistent row counts between images and labels

Values should be normalized (recommended range: 0–1)

🔹 Dataset Sources (Examples)

Users may generate or obtain spectrograms from:

Radio astronomy signal datasets

SETI-related public signal datasets

Any time–frequency spectrogram data with equivalent shape

The training pipeline is dataset-agnostic as long as the above structure is respected.
