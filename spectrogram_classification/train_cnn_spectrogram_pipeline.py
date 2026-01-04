"""
Spectrogram Classification using Convolutional Neural Networks

Objective:
Classify 2D spectrogram images of radio signals into four classes:
- squiggle
- narrowband
- narrowbanddrd
- noise
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# -------------------------------
# Load dataset
# -------------------------------
train_images = pd.read_csv("data/train/images.csv", header=None)
train_labels = pd.read_csv("data/train/labels.csv", header=None)
val_images = pd.read_csv("data/valid/images.csv", header=None)
val_labels = pd.read_csv("data/valid/labels.csv", header=None)

# -------------------------------
# Reshape images
# -------------------------------
X_train = train_images.values.reshape(-1, 64, 128, 1)
X_val = val_images.values.reshape(-1, 64, 128, 1)
y_train = train_labels.values
y_val = val_labels.values

# -------------------------------
# Data augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3
)

val_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3
)

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential([
    Conv2D(64, (5, 5), padding="same", input_shape=(64, 128, 1)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (5, 5), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (5, 5), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.4),

    Dense(4, activation="softmax")
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=5,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Training
# -------------------------------
checkpoint = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_loss",
    save_weights_only=True,
    mode="min"
)

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True),
    steps_per_epoch=len(X_train) // 32,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
    validation_steps=len(X_val) // 32,
    epochs=12,
    callbacks=[checkpoint]
)

# -------------------------------
# Evaluation
# -------------------------------
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(model.predict(X_val), axis=1)

print(classification_report(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
