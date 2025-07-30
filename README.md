# GL-Attention: Global-Local Self-Attention for Computer Vision

A flexible and configurable implementation of Global-Local Self-Attention mechanisms for computer vision tasks, with support for multiple fusion strategies and comprehensive experiment tracking.

## 🏗️ Architecture Overview

### Core Components

#### 1. **GLSABlock - The Heart of the Architecture**
```
Input Features (x) → Global Self-Attention (A) → Local Attention (L) → Fusion → Residual → Output
                                ↓                     ↓          ↓         ↓
                           Multi-Head              CBAM         Configurable  Configurable
                           Self-Attention                       Fusion        Residual
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
L = CBAM(f1)  # Convolutional Block Attention Module

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
│   ├── CBAM.py                  # Convolutional Block Attention Module
│   └── __init__.py              # Module exports
├── backbone/                     # Backbone networks
│   ├── models.py                # Unified backbone system (ResNet + VGG)
│   └── __init__.py              # Module exports
├── datasets/                     # Dataset handling
│   ├── dataset.py               # Unified dataset interface (HAM10000, ISIC-2018)
│   └── __init__.py              # Module exports
├── trainer/                      # Training framework
│   ├── trainer.py               # 🏋️ Enhanced trainer with logging
│   └── __init__.py              # Module exports
├── utils/                        # Utilities
│   ├── logger.py                # 📊 Comprehensive logging system
│   ├── losses.py                # 🎯 Advanced loss functions (Focal, Class-Balanced)
│   ├── grad_cam.py              # Visualization tools
│   └── utils.py                 # General utilities
├── logs/                         # Generated training logs (auto-created)
├── train.py                     # 🚀 Main training script
├── requirement.txt              # Dependencies
└── README.md                    # This documentation
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
# Train with default settings (original GLSA + Focal Loss)
python train.py --attention CBAM --dataset HAM10000

# Train with advanced fusion strategies
python train.py --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive

# Train with class imbalance handling
python train.py --attention CBAM --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0 --verbose
```

### Advanced Configuration

```bash
# Full configuration example
python train.py \
    --dataset HAM10000 \
    --backbone resnet50 \
    --attention CBAM \
    --fusion_strategy learnable_gate \
    --residual_strategy progressive \
    --batch_size 32 \
    --lr 1e-3 \
    --max_epoch 50 \
    --optimizer AdamW \
    --lr_scheduler ReduceLROnPlateau \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --verbose \
    --wandb_project gl-attention-experiments \
    --wandb_run experiment_v1
```

## 🧪 Experiment Management

### Fusion Strategy Comparison

```bash
# Compare different fusion strategies (with class imbalance handling)
python train.py --fusion_strategy original --loss_type focal --wandb_run baseline
python train.py --fusion_strategy learnable_gate --loss_type focal --wandb_run learnable_gate_exp  
python train.py --fusion_strategy channel_attn --loss_type focal --wandb_run channel_attn_exp
python train.py --fusion_strategy weighted_add --loss_type focal --wandb_run weighted_add_exp
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
- ✅ **Dataset Info**: Sample counts, class distribution, imbalance handling
- ✅ **Training Progress**: Epoch summaries, timing information
- ✅ **Batch Metrics**: Loss, accuracy, learning rate (when verbose)
- ✅ **Class Balance**: WeightedRandomSampler usage and strategy
- ✅ **Loss Function**: Focal Loss parameters and effectiveness
- ✅ **Learning Rate Changes**: Automatic detection and logging
- ✅ **Checkpoints**: Best model saves with metrics
- ✅ **Early Stopping**: Trigger events and reasons

## 🎯 Supported Medical Datasets

| Dataset | Classes | Images | Domain | Description |
|---------|---------|--------|--------|-------------|
| **HAM10000** | 7 | 10,015 | Dermatology | Human Against Machine with 10,000 training images for skin lesion classification |
| **ISIC-2018-Task-3** | 7 | 10,015 | Dermoscopy | International Skin Imaging Collaboration Challenge 2018 for lesion diagnosis |

### Dataset Classes

**HAM10000 Classes:**
- `akiec`: Actinic keratoses and intraepithelial carcinoma
- `bcc`: Basal cell carcinoma  
- `bkl`: Benign keratosis-like lesions
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic nevi
- `vasc`: Pyogenic granulomas and hemorrhage

**ISIC-2018-Task-3 Classes:**
- `MEL`: Melanoma
- `NV`: Melanocytic nevus
- `BCC`: Basal cell carcinoma
- `AKIEC`: Actinic keratosis / Bowen's disease
- `BKL`: Benign keratosis
- `DF`: Dermatofibroma
- `VASC`: Vascular lesion

## 🏗️ Model Architectures

### Backbone Integration

```python
# Create model with unified backbone system
model = create_model(
    backbone_name='resnet18',        # Choose from supported backbones
    pretrained=True,                 # Use ImageNet pretrained weights
    num_classes=7,                   # Number of output classes
    attn_type='CBAM',               # Attention mechanism
    fusion_strategy='learnable_gate', # Fusion strategy
    residual_strategy='progressive'   # Residual strategy
)

# Alternative backbone examples
model_resnet50 = create_model('resnet50', pretrained=True, num_classes=7)
model_vgg16 = create_model('vgg16', pretrained=True, num_classes=7)
```

### Backbone Networks

The framework supports multiple backbone architectures from ResNet and VGG families:

**ResNet Family:**
- `resnet18` (11.7M params, 69.8% ImageNet top-1)
- `resnet34` (21.8M params, 73.3% ImageNet top-1)  
- `resnet50` (25.6M params, 80.9% ImageNet top-1)
- `resnet101` (44.5M params, 81.9% ImageNet top-1)
- `resnet152` (60.2M params, 82.3% ImageNet top-1)

**VGG Family:**
- `vgg11` (132.9M params, 69.0% ImageNet top-1)
- `vgg13` (133.0M params, 69.9% ImageNet top-1)
- `vgg16` (138.4M params, 71.6% ImageNet top-1)
- `vgg19` (143.7M params, 72.4% ImageNet top-1)

### GLSA Block Configuration

```python
# Configurable GLSA Block
glsa_block = GLSABlock(
    channels=256,                    # Feature channels
    num_heads=8,                     # Multi-head attention heads
    attn_type='CBAM',               # Local attention type: CBAM or none
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
- Training/Validation Loss & Accuracy (with class-wise breakdown)
- Learning Rate schedules and convergence
- Model parameters and architecture complexity
- Fusion/Residual strategy configurations
- Class imbalance handling effectiveness
- Focal Loss performance vs CrossEntropy
- WeightedRandomSampler impact analysis
- Training time and efficiency metrics

### CSV Export

```python
# Epoch-level metrics automatically saved
training_log.csv:
epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate
1,1.8532,45.23,1.7234,52.15,0.001000
2,1.6234,58.32,1.5123,62.45,0.001000
```

## 🧪 Experimental Protocol for Research Paper

### Overview

This section provides a comprehensive experimental protocol designed for academic paper submission. The experiments are structured to demonstrate the effectiveness of GL-Attention across multiple dimensions: backbone architectures, attention mechanisms, fusion strategies, and datasets.

### 🎯 Experimental Design

#### 1. **Baseline Experiments** (Table 1: Backbone Comparison)

```bash
# Experiment 1.1: ResNet Family Baseline (No Attention + Focal Loss)
python train.py --dataset HAM10000 --backbone resnet18 --attention none --loss_type focal --wandb_run baseline_resnet18_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet34 --attention none --loss_type focal --wandb_run baseline_resnet34_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention none --loss_type focal --wandb_run baseline_resnet50_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet101 --attention none --loss_type focal --wandb_run baseline_resnet101_noattn --max_epoch 100

# Experiment 1.2: VGG Family Baseline (No Attention + Focal Loss)
python train.py --dataset HAM10000 --backbone vgg11 --attention none --loss_type focal --wandb_run baseline_vgg11_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone vgg13 --attention none --loss_type focal --wandb_run baseline_vgg13_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone vgg16 --attention none --loss_type focal --wandb_run baseline_vgg16_noattn --max_epoch 100
python train.py --dataset HAM10000 --backbone vgg19 --attention none --loss_type focal --wandb_run baseline_vgg19_noattn --max_epoch 100
```

#### 2. **CBAM Attention Impact Analysis** (Table 2: CBAM vs No Attention)

```bash
# Experiment 2.1: ResNet + CBAM + Focal Loss
python train.py --dataset HAM10000 --backbone resnet18 --attention CBAM --loss_type focal --wandb_run cbam_resnet18 --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --loss_type focal --wandb_run cbam_resnet50 --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet101 --attention CBAM --loss_type focal --wandb_run cbam_resnet101 --max_epoch 100

# Experiment 2.2: VGG + CBAM + Focal Loss
python train.py --dataset HAM10000 --backbone vgg16 --attention CBAM --loss_type focal --wandb_run cbam_vgg16 --max_epoch 100
python train.py --dataset HAM10000 --backbone vgg19 --attention CBAM --loss_type focal --wandb_run cbam_vgg19 --max_epoch 100
```

#### 3. **Fusion Strategy Ablation Study** (Table 3: Fusion Methods)

```bash
# Experiment 3: Fusion Strategy Comparison on ResNet50 + CBAM + Focal Loss
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy original --loss_type focal --wandb_run fusion_original --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run fusion_learnable_gate --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy weighted_add --loss_type focal --wandb_run fusion_weighted_add --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy channel_attn --loss_type focal --wandb_run fusion_channel_attn --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy spatial_attn --loss_type focal --wandb_run fusion_spatial_attn --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy cross_attn --loss_type focal --wandb_run fusion_cross_attn --max_epoch 100
```

#### 4. **Residual Strategy Ablation Study** (Table 4: Residual Methods)

```bash
# Experiment 4: Residual Strategy Comparison on ResNet50 + CBAM + Best Fusion + Focal Loss
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy original --loss_type focal --wandb_run residual_original --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy learnable_gate --loss_type focal --wandb_run residual_learnable_gate --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --loss_type focal --wandb_run residual_progressive --max_epoch 100
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy drop_path --loss_type focal --wandb_run residual_drop_path --max_epoch 100
```

#### 5. **Cross-Dataset Generalization** (Table 5: Dataset Transfer)

```bash
# Experiment 5.1: HAM10000 Final Configuration (Primary Dataset)
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --loss_type focal --wandb_run final_ham10000 --max_epoch 100

# Experiment 5.2: ISIC-2018-Task-3 Final Configuration (Validation Dataset)  
python train.py --dataset isic-2018-task-3 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --loss_type focal --wandb_run final_isic2018 --max_epoch 100

# Cross-validation with different backbones
python train.py --dataset isic-2018-task-3 --backbone resnet101 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --loss_type focal --wandb_run crossval_isic_resnet101 --max_epoch 100
python train.py --dataset isic-2018-task-3 --backbone vgg19 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --loss_type focal --wandb_run crossval_isic_vgg19 --max_epoch 100
```

#### 6. **Computational Efficiency Analysis** (Table 6: Efficiency Metrics)

```bash
# Experiment 6: Parameter Count and Speed Analysis with Focal Loss
python train.py --dataset HAM10000 --backbone resnet18 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run efficiency_resnet18 --max_epoch 10 --verbose
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run efficiency_resnet50 --max_epoch 10 --verbose  
python train.py --dataset HAM10000 --backbone resnet101 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run efficiency_resnet101 --max_epoch 10 --verbose
python train.py --dataset HAM10000 --backbone vgg16 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run efficiency_vgg16 --max_epoch 10 --verbose
python train.py --dataset HAM10000 --backbone vgg19 --attention CBAM --fusion_strategy learnable_gate --loss_type focal --wandb_run efficiency_vgg19 --max_epoch 10 --verbose
```

### 📊 Expected Paper Tables

#### Table 1: Backbone Architecture Comparison (No Attention)
| Backbone | Params | HAM10000 Acc (%) | ISIC-2018 Acc (%) | Training Time (min) |
|----------|--------|------------------|-------------------|---------------------|
| ResNet18 | 11.7M  | XX.X ± X.X       | XX.X ± X.X        | XX                  |
| ResNet34 | 21.8M  | XX.X ± X.X       | XX.X ± X.X        | XX                  |
| ResNet50 | 25.6M  | XX.X ± X.X       | XX.X ± X.X        | XX                  |
| ResNet101| 44.5M  | XX.X ± X.X       | XX.X ± X.X        | XX                  |
| VGG16    | 138.4M | XX.X ± X.X       | XX.X ± X.X        | XX                  |
| VGG19    | 143.7M | XX.X ± X.X       | XX.X ± X.X        | XX                  |

#### Table 2: CBAM Attention Impact Analysis  
| Model | Baseline (No Attention) | With CBAM | Improvement | P-value |
|-------|-------------------------|-----------|-------------|---------|
| ResNet18 | XX.X ± X.X            | XX.X ± X.X | +X.X       | < 0.05  |
| ResNet50 | XX.X ± X.X            | XX.X ± X.X | +X.X       | < 0.01  |
| ResNet101| XX.X ± X.X            | XX.X ± X.X | +X.X       | < 0.01  |
| VGG16    | XX.X ± X.X            | XX.X ± X.X | +X.X       | < 0.05  |
| VGG19    | XX.X ± X.X            | XX.X ± X.X | +X.X       | < 0.05  |

#### Table 3: Global-Local Fusion Strategy Analysis (ResNet50 + CBAM)
| Fusion Strategy | HAM10000 Acc (%) | ISIC-2018 Acc (%) | Additional Params | Reference |
|-----------------|-------------------|-------------------|-------------------|-----------|
| Original        | XX.X ± X.X        | XX.X ± X.X        | 0                 | Baseline  |
| Learnable Gate  | XX.X ± X.X        | XX.X ± X.X        | +XX.XK            | Highway Networks |
| Weighted Add    | XX.X ± X.X        | XX.X ± X.X        | +XX.XK            | EfficientDet |
| Channel Attn    | XX.X ± X.X        | XX.X ± X.X        | +XX.XK            | SE-Net |
| Spatial Attn    | XX.X ± X.X        | XX.X ± X.X        | +XX.XK            | CBAM |
| Cross Attn      | XX.X ± X.X        | XX.X ± X.X        | +XX.XK            | Transformer |

#### Table 4: Residual Strategy Comparison
| Residual Strategy | HAM10000 Acc (%) | ISIC-2018 Acc (%) | Convergence (epochs) |
|-------------------|-------------------|--------------------|---------------------|
| Original          | XX.X ± X.X        | XX.X ± X.X         | XX                  |
| Learnable Gate    | XX.X ± X.X        | XX.X ± X.X         | XX                  |
| Progressive       | XX.X ± X.X        | XX.X ± X.X         | XX                  |
| Drop Path         | XX.X ± X.X        | XX.X ± X.X         | XX                  |

### 🔄 Automated Experiment Runner

Create this script to run all experiments automatically:

```bash
#!/bin/bash
# run_paper_experiments.sh

echo "🧪 Starting GL-Attention Paper Experiments"
echo "Expected total time: ~48 hours on single GPU"

# Set common parameters with class imbalance handling
COMMON_ARGS="--batch_size 32 --lr 1e-3 --optimizer AdamW --lr_scheduler ReduceLROnPlateau --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0 --max_epoch 100 --early_stopping_patience 15 --wandb_project gl-attention-paper"

echo "📊 Phase 1: Baseline Experiments (8 models x 2 hours = 16 hours)"
for backbone in resnet18 resnet34 resnet50 resnet101 vgg11 vgg13 vgg16 vgg19; do
    echo "Running baseline: $backbone"
    python train.py --dataset HAM10000 --backbone $backbone --attention none --wandb_run baseline_${backbone}_noattn $COMMON_ARGS
done

echo "📊 Phase 2: CBAM Attention Impact Analysis (5 models x 2 hours = 10 hours)"
for backbone in resnet18 resnet50 resnet101 vgg16 vgg19; do
    echo "Running CBAM impact analysis: $backbone"
    python train.py --dataset HAM10000 --backbone $backbone --attention CBAM --wandb_run cbam_$backbone $COMMON_ARGS
done

echo "📊 Phase 3: Fusion Strategy Ablation (6 strategies x 2 hours = 12 hours)"
for strategy in original learnable_gate weighted_add channel_attn spatial_attn cross_attn; do
    echo "Running fusion: $strategy"
    python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy $strategy --wandb_run fusion_$strategy $COMMON_ARGS
done

echo "📊 Phase 4: Residual Strategy Ablation (4 strategies x 2 hours = 8 hours)"
for strategy in original learnable_gate progressive drop_path; do
    echo "Running residual: $strategy"
    python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy $strategy --wandb_run residual_$strategy $COMMON_ARGS
done

echo "📊 Phase 5: Cross-Dataset Validation (2 datasets x 2 hours = 4 hours)"
python train.py --dataset HAM10000 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --wandb_run final_ham10000 $COMMON_ARGS
python train.py --dataset isic-2018-task-3 --backbone resnet50 --attention CBAM --fusion_strategy learnable_gate --residual_strategy progressive --wandb_run final_isic2018 $COMMON_ARGS

echo "🎉 All experiments completed! Check WandB for results."
```

### 📈 Statistical Analysis Guide

```python
# analysis_script.py - Extract results for paper tables
import wandb
import pandas as pd
import numpy as np
from scipy import stats

def extract_experiment_results():
    """Extract results from WandB for paper tables"""
    
    api = wandb.Api()
    runs = api.runs("your-project/gl-attention-paper")
    
    results = []
    for run in runs:
        if run.state == "finished":
            results.append({
                'name': run.name,
                'backbone': run.config.get('backbone'),
                'attention': run.config.get('attention'),
                'fusion_strategy': run.config.get('fusion_strategy'),
                'residual_strategy': run.config.get('residual_strategy'),
                'val_accuracy': run.summary.get('val_accuracy'),
                'train_time': run.summary.get('_runtime'),
                'parameters': run.summary.get('total_parameters')
            })
    
    df = pd.DataFrame(results)
    return df

def generate_paper_tables(df):
    """Generate LaTeX tables for paper"""
    
    # Table 1: Baseline comparison
    baseline_results = df[df['attention'] == 'none']
    print("% Table 1: Baseline Results")
    for _, row in baseline_results.iterrows():
        print(f"{row['backbone']} & {row['parameters']/1e6:.1f}M & {row['val_accuracy']:.1f} \\\\")
    
    # Statistical significance tests
    cbam_results = df[df['attention'] == 'CBAM']
    baseline_results = df[df['attention'] == 'none']
    
    for backbone in ['resnet18', 'resnet50', 'vgg16']:
        cbam_acc = cbam_results[cbam_results['backbone'] == backbone]['val_accuracy'].values
        base_acc = baseline_results[baseline_results['backbone'] == backbone]['val_accuracy'].values
        
        if len(cbam_acc) > 0 and len(base_acc) > 0:
            t_stat, p_value = stats.ttest_ind(cbam_acc, base_acc)
            print(f"{backbone}: CBAM vs Baseline, p-value = {p_value:.4f}")

# Usage
# df = extract_experiment_results()
# generate_paper_tables(df)
```

### 🎯 Success Metrics for Paper

1. **Accuracy Improvement**: Target >2% improvement with CBAM vs baseline
2. **Fusion Strategy**: Best fusion should outperform original by >1%
3. **Residual Strategy**: Progressive residual should show faster convergence
4. **Cross-Dataset**: <5% accuracy drop when transferring between datasets
5. **Efficiency**: Computational overhead <20% compared to baseline
6. **Statistical Significance**: All major claims supported by p < 0.05

### 📝 Paper Writing Checklist

- [ ] Abstract mentions specific accuracy improvements
- [ ] Introduction cites fusion strategy papers (Highway Networks, EfficientDet, etc.)
- [ ] Method section details GLSA block architecture  
- [ ] Results section includes all 6 tables above
- [ ] Discussion analyzes fusion vs residual strategy trade-offs
- [ ] Conclusion highlights medical imaging applicability
- [ ] Supplementary material includes full hyperparameter details

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
- **SE-Net**: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

### Fusion Strategy References
- **Highway Networks**: [Highway Networks](https://arxiv.org/abs/1505.00387)
- **EfficientDet**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- **ReZero**: [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
- **Stochastic Depth**: [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet and VGG implementations based on torchvision
- CBAM attention mechanism inspired by the original paper
- Focal Loss implementation for class imbalance handling
- WeightedRandomSampler for medical dataset imbalance challenges
- Comprehensive logging system designed for reproducible research
- Training framework optimized for medical imaging classification

---

**Happy Experimenting! 🧪✨**

For questions and support, please open an issue in the repository.