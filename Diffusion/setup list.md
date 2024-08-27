# Model Setup Checklist

## 1. Hyperparameters

- **Number of Epochs**
- **Batch Size**
- **Dropout Rate**
- **Regularizer Strength**
- **Learning Rate**

## 2. Initial Setup

### Dataset Preparation

- **Dataset Download and Transforms**: Including any data augmentation.
- **Dataloader Setup**
- **Datasampler Setup**

### Reproducibility and Device Setup

- **Seeding Initialization for Reproducibility**
- **Device Type**: Configuration for using CPU/GPU.

### Model and Training Configuration

- **Model Initialization**
- **Weight Initialization**: Method (e.g., Xavier, He, Orthogonal).
- **Precision**: Decide on mixed precision (FP16) or full precision (FP32).

### Training Control

- **Loss Function**: Type of loss function(s) used.
- **Regularization**: L1/L2 regularization strength.
- **Optimizer**: Choice of optimizer (e.g., Adam, SGD) and related parameters.
- **Learning Rate Scheduler**: Type and parameters (e.g., StepLR, ReduceLROnPlateau).
- **Gradient Clipping**: Clip value if using gradient clipping.
- **Early Stopping**: Criteria for early stopping.
- **Train and Eval Setup**

### Model Persistence

- **Checkpointing Logic**
- **Checkpoint Loading **

## 3. Additional Configurations

- **Logging Setup**: Set up logging for metrics, losses, and other relevant information.
- **Evaluation Metrics**: Define the metrics used for evaluation during training and validation (e.g., accuracy, F1 score).
- **Distributed Training Setup**: Configuration for multi-GPU or distributed training (e.g., DDP).
