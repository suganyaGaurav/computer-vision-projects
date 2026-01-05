# Handwritten Word Recognition using CNN + BiLSTM + CTC

## Problem Statement

Handwritten word recognition is a challenging computer vision task due to:

- Variability in handwriting styles
- Variable image sizes and aspect ratios
- Variable-length word sequences
- Lack of explicit character-level alignment

The goal of this project is to build an **end-to-end handwritten word recognition system** that converts word images directly into text without performing character-level segmentation.

---

## Dataset & Preprocessing

- Handwritten word images organized in a hierarchical folder structure
- Labels provided at **word level**
- Key preprocessing steps:
  - Aspect-ratio–preserving resize to `(128 × 32)` or `(32 × 128)`
  - Symmetric padding to handle variable image sizes
  - Grayscale normalization to pixel values in `[0, 1]`
  - Dynamic character vocabulary construction
  - Padding of label sequences to the maximum word length

---

## Method 1: TensorFlow OCR Pipeline (CNN + BiLSTM + CTC)

### Approach

- Convolutional Neural Networks (CNNs) extract spatial features from handwritten word images
- Feature maps are reshaped into temporal sequences
- Bidirectional LSTM layers model sequential dependencies between characters
- Connectionist Temporal Classification (CTC) loss enables training without character-level alignment

### Key Characteristics

- Implemented using `tf.data` pipelines for efficient data loading
- Edit distance used as the primary evaluation metric
- Greedy CTC decoding used during inference
- No pretrained OCR models or language models used

### Evaluation

- Mean edit distance tracked at the end of each epoch
- Visual comparison of predicted text vs ground truth
- Robust handling of variable-length word predictions

---

## Method 2: Classical OCR Pipeline with Explicit Encoding

### Approach

- Manual image resizing and padding to a fixed resolution
- Explicit character set definition (letters and digits)
- Labels encoded into integer sequences
- Deep CNN stack for hierarchical feature extraction
- Bidirectional LSTM layers for sequence modeling
- CTC loss implemented using the Keras backend

### Training Strategy

- Explicit handling of:
  - Input sequence lengths
  - Label sequence lengths
  - Padded label tensors
- RMSProp optimizer
- Learning-rate reduction on plateau
- Best-model checkpointing

### Evaluation

- Greedy CTC decoding for inference
- Exact string-level comparison between predictions and ground-truth labels
- Qualitative and quantitative performance assessment

---

## Key Learnings

- Practical implementation of CTC-based sequence learning
- Deep understanding of OCR pipelines beyond pretrained models
- Importance of careful preprocessing in vision tasks
- Trade-offs between automation and explicit control
- CNN–RNN cooperation for vision-to-text problems

---

## Why This Project Matters

This project demonstrates a **from-scratch handwritten word recognition system** built using classical deep learning techniques that form the foundation of modern OCR systems.

It emphasizes:

- Explainability and transparency
- Strong computer vision fundamentals
- Sequence modeling expertise
- Production-minded evaluation practices

---

## Notes

- Dataset files are not included due to size and licensing constraints
- This repository focuses on methodology, modeling decisions, and results
