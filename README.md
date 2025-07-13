# GL-Attention: Global-Local Self-Attention for Computer Vision

A flexible and configurable implementation of Global-Local Self-Attention mechanisms for computer vision tasks, with support for multiple fusion strategies and comprehensive experiment tracking.

## 🏗️ Architecture Overview

### Core Components

#### 1. **GLSABlock - The Heart of the Architecture**
```
Input Features (x) → Global Self-Attention (A) → Local Attention (L) → Fusion → Residual → Output
                                ↓                     ↓          ↓         ↓
                           Multi-Head              CBAM/BAM/    Configurable  Configurable
                           Self-Attention          scSE         Fusion        Residual
```

**Key Innovation**: Combines global context (via self-attention) with local features (via CNN-based attention) through configurable fusion strategies.

#### 2. **Multi-Stage Processing Pipeline**

```python
# Stage 1: Global Context Extraction
qkv = Conv1x1(x)  # Project to Q, K, V
q, k, v = chunk(qkv, 3)
A = MultiHeadAttention(q, k, v)  # Global features

# Stage 2: Local Feature Enhancement  
f1 = x + A  # Intermediate fusion
L = LocalAttention(f1)  # CBAM/BAM/scSE

# Stage 3: Configurable Fusion
R = FusionStrategy(A, L, x)  # Multiple strategies available

# Stage 4: Configurable Residual
output = ResidualStrategy(x, R)  # Multiple strategies available
```

### 🔧 Fusion Strategies

| Strategy | Description | Papers | Use Case |
|----------|-------------|--------|----------|
| `original` | Simple multiplicative fusion: `A ⊙ L` | Base implementation | Baseline comparison |
| `learnable_gate` | Learnable gating: `gate·A + (1-gate)·L` | Highway Networks (2015) | Adaptive feature selection |
| `weighted_add` | Weighted combination: `α·A + β·L + γ·(A⊙L)` | EfficientDet (2020) | Multi-scale fusion |
| `channel_attn` | Channel-wise attention fusion | SE-Net (2018) | Channel importance weighting |
| `spatial_attn` | Spatial attention fusion | CBAM (2018) | Spatial importance weighting |
| `cross_attn` | Cross-attention between A and L | Transformer variants | Complex feature interaction |

### 🔗 Residual Strategies

| Strategy | Description | Papers | Benefits |
|----------|-------------|--------|----------|
| `original` | Standard residual: `x + R` | ResNet (2016) | Gradient flow, training stability |
| `learnable_gate` | Gated residual: `gate·x + (1-gate)·R` | Highway Networks (2015) | Adaptive skip connections |
| `progressive` | Weighted residual: `x·(1-α) + R·α` | ReZero (2021) | Controlled feature evolution |
| `drop_path` | Stochastic depth | Stochastic Depth (2016) | Regularization, robustness |

## 📁 Project Structure

```
GL-Attention/
├── attention/                    # Attention mechanisms
│   ├── GLSABlock.py             # 🎯 Main GLSA implementation
│   ├── fusion_strategies.py     # 🔄 Fusion strategy implementations
│   ├── CBAM.py                  # Convolutional Block Attention
│   ├── BAM.py                   # Bottleneck Attention Module
│   └── scSE.py                  # Spatial and Channel Squeeze-Excitation
├── backbone/                     # Backbone networks
│   ├── ResNet.py                # ResNet18 with GLSA integration
│   └── VGG.py                   # VGG16 with GLSA integration
├── datasets/                     # Dataset handling
│   └── dataset.py               # Unified dataset interface
├── trainer/                      # Training framework
│   └── trainer.py               # 🏋️ Enhanced trainer with logging
├── utils/                        # Utilities
│   ├── logger.py                # 📊 Comprehensive logging system
│   ├── grad_cam.py              # Visualization tools
│   └── utils.py                 # General utilities
├── logs/                         # Generated logs
├── train.py                     # 🚀 Main training script
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GL-Attention

# Install dependencies
pip install -r requirement.txt
```

### Basic Training

```bash
# Train with default settings (original GLSA)
python train.py --attention CBAM --dataset HAM10000

# Train with learnable gate fusion
python train.py --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive

# Train with verbose logging (see batch-level metrics)
python train.py --attention CBAM --fusion_strategy channel_attn --verbose
```

### Advanced Configuration

```bash
# Full configuration example
python train.py \
    --dataset HAM10000 \
    --backbone ResNet18 \
    --attention CBAM \
    --fusion_strategy learnable_gate \
    --residual_strategy progressive \
    --batch_size 32 \
    --lr 1e-3 \
    --max_epoch 50 \
    --optimizer AdamW \
    --lr_scheduler ReduceLROnPlateau \
    --verbose \
    --wandb_project gl-attention-experiments \
    --wandb_run experiment_v1
```

## 🧪 Experiment Management

### Fusion Strategy Comparison

```bash
# Compare different fusion strategies
python train.py --fusion_strategy original --wandb_run baseline
python train.py --fusion_strategy learnable_gate --wandb_run learnable_gate_exp  
python train.py --fusion_strategy channel_attn --wandb_run channel_attn_exp
python train.py --fusion_strategy weighted_add --wandb_run weighted_add_exp
```

### Automated Experiment Naming

The system automatically creates descriptive run names:
- Format: `{base_name}_{fusion_strategy}_{residual_strategy}_{attention_type}`
- Example: `experiment_v1_learnable_gate_progressive_CBAM`

## 📊 Logging and Monitoring

### Comprehensive Logging System

```python
# Three levels of logging
📄 Console Output     # Real-time progress (INFO level)
📁 Detailed Logs      # Complete training logs (DEBUG level)  
📈 Metrics Files      # Structured metrics for analysis
```

### Log Files Generated

```
logs/
├── GL-Attention_20250713_140530.log          # Detailed training logs
├── training_metrics_20250713_140530.log      # Structured metrics (CSV-like)
└── training_log.csv                          # Epoch-level metrics
```

### Monitoring Options

```bash
# Normal training (clean console output)
python train.py --attention CBAM

# Verbose training (batch-level metrics)  
python train.py --attention CBAM --verbose

# Real-time log monitoring
tail -f logs/GL-Attention_*.log
```

### What Gets Logged

- ✅ **Configuration**: All hyperparameters and model settings
- ✅ **Model Info**: Architecture, parameter counts, fusion strategies  
- ✅ **Dataset Info**: Sample counts, class distribution
- ✅ **Training Progress**: Epoch summaries, timing information
- ✅ **Batch Metrics**: Loss, accuracy, learning rate (when verbose)
- ✅ **Learning Rate Changes**: Automatic detection and logging
- ✅ **Checkpoints**: Best model saves with metrics
- ✅ **Early Stopping**: Trigger events and reasons

## 🎯 Supported Datasets

| Dataset | Classes | Domain | Usage |
|---------|---------|--------|-------|
| **HAM10000** | 7 | Medical (Skin lesions) | Skin cancer classification |
| **ISIC-2018-Task-3** | 7 | Medical (Dermoscopy) | Skin lesion diagnosis |
| **STL10** | 10 | Natural images | General object recognition |
| **Caltech101** | 101 | Objects | Fine-grained classification |
| **Caltech256** | 256 | Objects | Large-scale classification |
| **Oxford-IIIT Pets** | 37 | Animals | Pet breed classification |

## 🏗️ Model Architectures

### Backbone Integration

```python
# ResNet18 + GLSA
model = ResNet18(
    pretrained=True,
    attn_type='CBAM', 
    num_classes=7
)

# VGG16 + GLSA  
model = VGG16(
    pretrained=True,
    attn_type='CBAM',
    num_classes=7
)
```

### GLSA Block Configuration

```python
# Configurable GLSA Block
glsa_block = GLSABlock(
    channels=256,                    # Feature channels
    num_heads=8,                     # Multi-head attention heads
    attn_type='CBAM',               # Local attention type
    fusion_strategy='learnable_gate', # Fusion strategy
    residual_strategy='progressive'   # Residual strategy
)
```

## 📈 Performance Monitoring

### WandB Integration

```python
# Automatic logging to Weights & Biases
configs = {
    "wandb_project": "gl-attention-research",
    "wandb_run": "experiment_v1", 
    "wandb_api_key": "your_api_key"
}
```

**Tracked Metrics:**
- Training/Validation Loss & Accuracy
- Learning Rate schedules
- Model parameters and architecture
- Fusion strategy configurations
- Training time and efficiency metrics

### CSV Export

```python
# Epoch-level metrics automatically saved
training_log.csv:
epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate
1,1.8532,45.23,1.7234,52.15,0.001000
2,1.6234,58.32,1.5123,62.45,0.001000
```

## 🔬 Research Features

### Ablation Studies

```bash
# Study fusion strategies
for strategy in original learnable_gate channel_attn weighted_add; do
    python train.py --fusion_strategy $strategy --wandb_run ablation_fusion_$strategy
done

# Study residual strategies  
for strategy in original learnable_gate progressive drop_path; do
    python train.py --residual_strategy $strategy --wandb_run ablation_residual_$strategy
done
```

### Hyperparameter Sweeps

```bash
# Learning rate sweep
for lr in 1e-2 1e-3 1e-4; do
    python train.py --lr $lr --wandb_run lr_sweep_$lr
done

# Batch size sweep
for bs in 16 32 64; do  
    python train.py --batch_size $bs --wandb_run bs_sweep_$bs
done
```

### Architecture Analysis

```python
# Model parameter counting
model = ResNet18(attn_type='CBAM', num_classes=7)
param_info = model.mha_block.count_parameters()

print(f"Total parameters: {param_info['total']:,}")
print(f"Fusion parameters: {param_info['fusion']:,}")  
print(f"Residual parameters: {param_info['residual']:,}")
```

## 🛠️ Customization Guide

### Adding New Fusion Strategies

```python
# In fusion_strategies.py
class CustomFusion(BaseFusion):
    def __init__(self, channels):
        super().__init__(channels)
        # Initialize your custom layers
        
    def forward(self, A, L, x_in=None):
        # Implement your fusion logic
        return fused_features

# Register the strategy
FUSION_STRATEGIES['custom'] = CustomFusion
```

### Adding New Residual Strategies

```python
# In fusion_strategies.py  
class CustomResidual(ResidualStrategy):
    def forward(self, x_in, fused_features):
        # Implement your residual logic
        return output

# Register the strategy
RESIDUAL_STRATEGIES['custom'] = CustomResidual
```

### Custom Datasets

```python
# In datasets/dataset.py
class CustomDataset(GeneralDataset):
    def __init__(self, dataset_path):
        # Initialize your dataset
        pass
        
    def get_splits(self, val_size=0.2, seed=42, image_size=224):
        # Implement train/val split logic
        return train_dataset, test_dataset
```

## 📊 Results Analysis

### Log Analysis Scripts

```bash
# Extract epoch metrics
grep "EPOCH" logs/training_metrics_*.log > epoch_metrics.csv

# Extract batch metrics  
grep "BATCH" logs/training_metrics_*.log > batch_metrics.csv

# Analyze learning rate changes
grep "Learning rate changed" logs/GL-Attention_*.log
```

### Performance Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training metrics
df = pd.read_csv('training_log.csv')

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)  
plt.plot(df['epoch'], df['train_accuracy'], label='Train')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation')
plt.legend()
plt.title('Accuracy Curves')
plt.show()
```

## 🤝 Contributing

### Code Style
- Follow PEP 8 conventions
- Add docstrings to all functions
- Use type hints where appropriate
- Write unit tests for new features

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📚 References

### Core Papers
- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Attention Mechanisms**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **CBAM**: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- **BAM**: [Bottleneck Attention Module](https://arxiv.org/abs/1807.06514)
- **SE-Net**: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

### Fusion Strategy References
- **Highway Networks**: [Highway Networks](https://arxiv.org/abs/1505.00387)
- **EfficientDet**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- **ReZero**: [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
- **Stochastic Depth**: [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet and VGG implementations based on torchvision
- Attention mechanisms inspired by CBAM, BAM, and SE-Net papers
- Logging system designed for reproducible research
- Training framework optimized for medical imaging tasks

---

**Happy Experimenting! 🧪✨**

For questions and support, please open an issue in the repository.