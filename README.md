# Iranian Food Classification 🍲

This repository contains a lightweight Convolutional Neural Network (CNN) model for classifying **21 different types of Iranian foods** using computer vision techniques.

## 📌 Project Overview
- **Goal:** Build a deep learning model that can recognize and classify Iranian foods into 21 categories.
- **Dataset:**
  - Training data: labeled images of 21 food categories.
  - Validation data: split from the training data.
  - Test data: unlabeled images used for final evaluation.
- **Model Type:** Lightweight CNN with depthwise separable convolutions.
- **Framework:** TensorFlow / Keras.

## 📊 Performance
- **Training Accuracy:** ~71%
- **Test Accuracy:** ~65%

> Note: Accuracy can be improved with additional data augmentation, hyperparameter tuning, or using pre-trained models (transfer learning).

## 🏗️ Model Architecture
The model uses a series of **Separable Convolution Blocks (SepBlock)** combined with batch normalization, ReLU activation, and residual connections for efficiency.

### Key Layers:
- **Rescaling**: Normalize input images to [0,1].
- **Conv2D + BatchNorm + ReLU**: Initial feature extraction.
- **SepBlock**: Depthwise separable convolutions with residuals.
- **Global Average Pooling**: Reduces dimensions before classification.
- **Dense Softmax Layer**: Final classifier for 21 food categories.

## 🚀 Code
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def SepBlock(x, filters, strides=1):
    shortcut = x
    x = layers.SeparableConv2D(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if strides == 1 and shortcut.shape[-1] == x.shape[-1]:
        x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x


def build_lightweight_cnn(input_shape=(256,256,3), num_classes=21, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = SepBlock(x, 32, strides=1)
    x = SepBlock(x, 64, strides=2)
    x = SepBlock(x, 64, strides=1)
    x = SepBlock(x, 128, strides=2)
    x = SepBlock(x, 128, strides=1)
    x = SepBlock(x, 256, strides=2)
    x = SepBlock(x, 256, strides=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="light_cnn")
    return model


if __name__ == "__main__":
    model = build_lightweight_cnn()
    model.summary()
```


## 📈 Possible Improvements
- Add **data augmentation** (rotation, flipping, zooming).
- Use **transfer learning** with models like MobileNetV2, EfficientNet, or ResNet.
- Fine-tune hyperparameters (learning rate, dropout, batch size).

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.<br/>
Email: **yousef.yousefian.85@gmail.com**

---
✨ Built with TensorFlow & Keras for Iranian cuisine recognition!
