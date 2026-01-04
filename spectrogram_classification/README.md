# Spectrogram Classification using CNN

## Problem Statement
Classify 2D spectrogram images of radio signals from space into four categories:
- Squiggle
- Narrowband
- Narrowband Drift
- Noise

## Approach
- CNN-based deep learning model
- Data augmentation to improve generalization
- Categorical cross-entropy loss
- Adam optimizer with learning rate decay

## Observations
Despite a limited number of samples per class and relatively small image sizes,
the model achieves strong validation accuracy.

## Technologies Used
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn (evaluation)
