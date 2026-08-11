# CIFAR-10 Image Classification with a Convolutional Neural Network

A complete image classification project built with **Python and PyTorch**, using the **CIFAR-10 dataset** to train a Convolutional Neural Network (CNN) capable of classifying images into ten different categories.

The project is structured as a small, reproducible machine learning system rather than as a single training script. It separates dataset preparation, data loading, model architecture, training, inference, and experimentation into independent components.

The main objective is not only to obtain a working classifier, but also to demonstrate a clear understanding of the fundamentals behind convolutional neural networks, image preprocessing, supervised learning, model evaluation, and PyTorch's training workflow.

---

## Project Overview

Image classification is a supervised learning problem in which a model learns to associate an input image with one of a predefined set of classes.

For this project, the model is trained on **CIFAR-10**, a standard computer vision benchmark containing:

* **50,000 training images**
* **10,000 test images**
* **10 classes**
* RGB images of **32 × 32 pixels**

The ten classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The model receives an RGB image with shape:

```text
3 × 32 × 32
```

and produces ten output values, one for each class.

The predicted class is obtained from the output with the highest score.

---

## Why a CNN?

A Convolutional Neural Network is particularly appropriate for image data because it can learn spatial patterns directly from the input.

Instead of treating every pixel as an independent feature, convolutional layers learn local visual features such as:

* edges
* textures
* shapes
* more complex visual patterns

As the information passes through successive convolutional layers, the network can build increasingly abstract representations of the image.

The architecture used in this project follows this general progression:

```text
Input Image
    ↓
Convolution
    ↓
ReLU
    ↓
Max Pooling
    ↓
Convolution
    ↓
ReLU
    ↓
Max Pooling
    ↓
Convolution
    ↓
ReLU
    ↓
Max Pooling
    ↓
Flatten
    ↓
Fully Connected Layer
    ↓
ReLU
    ↓
Fully Connected Layer
    ↓
10 Class Scores
```

This architecture is intentionally compact. The objective is to build a model that is understandable, reproducible, and appropriate for CIFAR-10 without introducing unnecessary complexity.

---

# Model Architecture

The CNN is implemented in:

```text
src/model.py
```

The network contains three convolutional blocks.

### First convolution

```text
3 → 16 channels
```

The first layer receives the three RGB channels and learns 16 feature maps.

### Second convolution

```text
16 → 32 channels
```

The second layer increases the number of learned features.

### Third convolution

```text
32 → 64 channels
```

The third layer produces a richer representation containing 64 feature maps.

Each convolution uses:

```text
kernel_size = 3
padding = 1
```

This preserves the spatial dimensions before pooling.

ReLU activation is used after every convolution.

Max pooling with:

```text
kernel_size = 2
stride = 2
```

reduces the spatial resolution after each convolutional block.

Starting from:

```text
32 × 32
```

the three pooling operations reduce the spatial dimensions to:

```text
16 × 16
8 × 8
4 × 4
```

Therefore, after the final convolutional block the tensor has:

```text
64 × 4 × 4
```

which is flattened before entering the fully connected layers.

The classifier then uses:

```text
64 × 4 × 4 → 128 → 10
```

The final layer outputs ten values corresponding to the ten CIFAR-10 classes.

---

# Data Preprocessing

Dataset and preprocessing logic are implemented in:

```text
src/dataset.py
```

The training pipeline uses data augmentation to make the model less dependent on the exact appearance or position of individual training examples.

Training images use:

* Random cropping with padding
* Random horizontal flipping
* Conversion to tensors
* Normalization

The test pipeline does not use random augmentation because evaluation should be deterministic.

The CIFAR-10 normalization statistics used are:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
```

Normalization helps place the input values on a more suitable scale for neural network training.

The separation between training and test transformations is intentional:

```text
Training:
RandomCrop
RandomHorizontalFlip
ToTensor
Normalize

Testing / inference:
Resize
ToTensor
Normalize
```

Random transformations are therefore applied only during training.

---

# Data Loading

DataLoader configuration is handled by:

```text
src/dataloader.py
```

The project separates dataset creation from batch loading.

The training DataLoader uses:

```text
batch_size = 128
shuffle = True
```

while the test DataLoader uses:

```text
batch_size = 100
shuffle = False
```

Shuffling the training dataset helps prevent the model from relying on the order of the training examples.

The test set is not shuffled because there is no training benefit from doing so and deterministic evaluation is preferable.

The number of workers can also be configured through the DataLoader interface.

---

# Training

Training is implemented in:

```text
src/train.py
```

The training process follows the standard supervised deep learning workflow:

1. Load the training and test datasets.
2. Create the DataLoaders.
3. Instantiate the CNN.
4. Select the available computation device.
5. Define the loss function.
6. Define the optimizer.
7. Iterate through the training epochs.
8. Evaluate the model after each epoch.
9. Track loss and accuracy.
10. Save the best-performing model.

The project automatically selects the best available device:

```python
mps
cuda
cpu
```

in that order.

This makes the project portable across Apple Silicon Macs, NVIDIA systems, and CPU-only environments.

---

# Loss Function

The model uses:

```python
nn.CrossEntropyLoss()
```

This is appropriate for multi-class classification where each image belongs to exactly one class.

The CNN outputs ten class scores, and `CrossEntropyLoss` compares these outputs with the correct class labels during training.

---

# Optimizer

The optimizer used is:

```python
optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam was selected because it provides adaptive learning rates for the model parameters and is a strong baseline for this type of classification problem.

The initial learning rate is:

```text
0.001
```

---

# Training and Validation Metrics

The training pipeline tracks:

* Training loss
* Training accuracy
* Validation loss
* Validation accuracy

Accuracy is calculated as:

```text
correct predictions / total predictions
```

Tracking both loss and accuracy is useful because they provide different information.

Loss measures how far the model's predictions are from the target labels, while accuracy measures how many predictions are actually correct.

Comparing training and validation metrics also helps identify potential overfitting.

---

# Model Checkpointing

The project does not simply save the final model.

Instead, it keeps track of the best validation accuracy and saves the corresponding checkpoint:

```text
models/best_model.pth
```

The checkpoint contains:

* Model parameters
* Optimizer parameters
* Epoch number
* Validation accuracy
* CIFAR-10 class names

This makes the trained model reproducible and allows training information to be retained alongside the model weights.

---

# Inference

Inference is implemented in:

```text
src/predict.py
```

The prediction pipeline accepts an external image from the command line.

The image is:

1. Loaded with Pillow.
2. Converted to RGB.
3. Resized/preprocessed.
4. Normalized using the CIFAR-10 statistics.
5. Converted into a batch of one image.
6. Passed through the trained CNN.
7. Converted into probabilities using softmax.
8. Mapped to the corresponding CIFAR-10 class.

The script returns:

```text
Predicted class
Confidence
```

For example:

```text
Using device: cpu
Predicted class: Ship
Confidence: 82.41%
```

The confidence represents the model's softmax probability for its predicted class. It should be interpreted as the model's confidence in its prediction, rather than as a guaranteed probability that the prediction is correct.

---

# Project Structure

The repository is organized as follows:

```text
cifar10-cnn/
│
├── data/
│   └── ...
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `data/`

Contains the downloaded CIFAR-10 dataset.

This directory is generated locally and should not be committed to Git.

### `models/`

Contains trained model checkpoints.

The main checkpoint produced by the training script is:

```text
models/best_model.pth
```

### `notebooks/`

Contains exploratory experiments and visual analysis.

The notebook is intended to demonstrate the experimental side of the project, while the reusable implementation remains in `src/`.

### `src/dataset.py`

Responsible for:

* CIFAR-10 loading
* Training transformations
* Test transformations
* Dataset configuration

### `src/dataloader.py`

Responsible for:

* Batch creation
* Shuffling
* DataLoader configuration

### `src/model.py`

Contains the CNN architecture.

### `src/train.py`

Contains:

* Training loop
* Validation loop
* Device selection
* Loss calculation
* Optimization
* Metric tracking
* Model checkpointing

### `src/predict.py`

Provides command-line inference on individual images.

---

# Installation

## Requirements

The project was developed using:

```text
Python 3.12
PyTorch 2.11.0
torchvision 0.26.0
NumPy 2.5.2
Matplotlib 3.11.1
Jupyter
```

The exact Python package dependencies are listed in:

```text
requirements.txt
```

Python **3.12.x** is recommended for reproducing the development environment.

---

# Setup

Clone the repository:

```bash
git clone <repository-url>
cd cifar10-cnn
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The project intentionally uses a virtual environment so that its dependencies remain isolated from the system Python installation.

---

# Verify the Environment

Before training, verify that PyTorch is available:

```bash
python -c "import torch; print(torch.__version__)"
```

Expected output should be compatible with the version specified in `requirements.txt`.

You can also check the available computation backend:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

On Apple Silicon, PyTorch may report that MPS is built but unavailable depending on the installed macOS/PyTorch environment.

If MPS is unavailable, the project automatically falls back to CPU.

---

# Downloading CIFAR-10

The dataset is downloaded automatically the first time the project loads it.

Run:

```bash
python -c "from src.dataset import get_datasets; train, test = get_datasets(); print('Train:', len(train)); print('Test:', len(test))"
```

The expected dataset sizes are:

```text
Train: 50000
Test: 10000
```

The dataset will be stored under:

```text
data/
```

If the dataset has already been downloaded, torchvision will reuse the local copy rather than downloading it again.

---

# Training the Model

From the project root, run:

```bash
python src/train.py
```

The training script will:

* Load CIFAR-10
* Create the DataLoaders
* Build the CNN
* Select the available device
* Train for the configured number of epochs
* Evaluate after every epoch
* Print training and validation metrics
* Save the best checkpoint

The best model will be written to:

```text
models/best_model.pth
```

Example output:

```text
Using device: cpu

Epoch 1/10 | Train Loss: ... | Train Accuracy: ... | Validation Loss: ... | Validation Accuracy: ...
Epoch 2/10 | Train Loss: ... | Train Accuracy: ... | Validation Loss: ... | Validation Accuracy: ...
...
Training completed.
Best validation accuracy: ...
Model saved to: models/best_model.pth
```

---

# Running Inference

Once a model has been trained, an individual image can be classified with:

```bash
python src/predict.py --image path/to/image.jpg
```

The default checkpoint is:

```text
models/best_model.pth
```

A different checkpoint can be specified with:

```bash
python src/predict.py \
    --image path/to/image.jpg \
    --model path/to/model.pth
```

Example:

```text
Using device: cpu
Predicted class: Automobile
Confidence: 91.27%
```

---

# Running the Notebook

The project also includes:

```text
notebooks/experiments.ipynb
```

The notebook is intended for experimentation and analysis rather than containing the core reusable implementation.

Start Jupyter with:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Alternatively, open the notebook directly in Visual Studio Code.

When VS Code asks for a kernel, select the Python interpreter from the project's virtual environment:

```text
.venv
```

The selected interpreter should correspond to:

```text
.venv/bin/python
```

This is important because the notebook must use the same environment in which PyTorch, torchvision, NumPy, and the other project dependencies are installed.

---

# Reproducibility

The project separates experimentation from the reusable training code.

The main Python modules are responsible for the actual machine learning pipeline, while the notebook is used to inspect results and experiment with the model.

This structure makes it possible to reproduce the training process without depending on notebook execution order.

For a fully controlled experiment, the following should be kept consistent:

* Python version
* Package versions
* Dataset
* Model architecture
* Batch sizes
* Learning rate
* Optimizer
* Number of epochs
* Data preprocessing

Neural network training can still show small differences between runs depending on hardware and backend behavior.

---

# Design Decisions

Several design choices were made deliberately.

### Modular architecture

The project separates data handling, model definition, training, and inference.

This makes individual components easier to understand, test, modify, and reuse.

### Data augmentation

Random cropping and horizontal flipping are applied during training to increase variation in the training samples and reduce dependence on exact image positioning.

### Normalization

CIFAR-10-specific mean and standard deviation values are used rather than arbitrary normalization values.

### Validation during training

The test dataset is evaluated after each epoch in the current implementation so that model performance can be monitored throughout training.

### Best-model checkpointing

The model with the highest validation accuracy is saved rather than simply saving the last epoch.

### Device abstraction

The training and inference code automatically selects between Apple MPS, NVIDIA CUDA, and CPU.

This makes the project portable across different machines.

---

# What I Learned From This Project

This project helped connect the theoretical concepts of convolutional neural networks with a complete PyTorch implementation.

The main concepts demonstrated are:

* Supervised image classification
* Convolutional layers
* Feature maps
* ReLU activation
* Max pooling
* Flattening
* Fully connected layers
* Cross-entropy loss
* Gradient-based optimization
* Adam optimizer
* Backpropagation
* Data augmentation
* Image normalization
* Batch training
* Model evaluation
* Checkpointing
* Softmax probabilities
* Inference
* Hardware-aware training

One of the key ideas behind the architecture is that convolutional layers progressively transform raw pixel information into increasingly useful feature representations.

The first layers operate on relatively simple visual patterns, while deeper layers can combine these patterns into more meaningful representations for classification.

The training process then adjusts the model parameters through backpropagation so that the predicted class scores become increasingly aligned with the correct labels.

---

# Limitations

This is intentionally a relatively small CNN designed for CIFAR-10.

It is not intended to compete with modern state-of-the-art computer vision architectures.

There are several directions that could improve performance:

* Deeper CNN architectures
* Batch normalization
* Dropout
* Learning-rate scheduling
* More advanced data augmentation
* Residual connections
* Transfer learning
* Modern architectures such as ResNet or EfficientNet
* Hyperparameter optimization
* More extensive experiment tracking

These approaches would also provide useful extensions for future versions of the project.

---

# Possible Future Improvements

A natural next step would be to compare the baseline CNN against a stronger architecture and quantify the improvement.

For example:

```text
Baseline CNN
     ↓
Batch Normalization
     ↓
Learning Rate Scheduler
     ↓
Regularization
     ↓
Residual CNN
     ↓
Transfer Learning
```

Future experiments could compare:

* Validation accuracy
* Training time
* Number of parameters
* Convergence speed
* Generalization performance

This would turn the project from a single-model implementation into a more systematic computer vision experiment.

---

# Technologies

The project uses:

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Matplotlib**
* **Jupyter**
* **Pillow**
* **Git**

---

# Project Goal

The goal of this project is to demonstrate an end-to-end understanding of a computer vision workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
DataLoader
   ↓
CNN Architecture
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimization
   ↓
Validation
   ↓
Best Model Checkpoint
   ↓
Inference
```

Rather than treating the neural network as a black box, the project is structured so that each stage of the pipeline can be inspected and understood independently.

---

# License

This project is intended as a personal machine learning portfolio project for educational and professional development purposes.
