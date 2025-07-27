# SOTA Seismic Velocity Inversion Framework

A state-of-the-art ensemble framework for seismic velocity inversion combining multiple cutting-edge techniques from recent research papers.

## Overview

This framework combines several state-of-the-art techniques for seismic velocity inversion:

1. **Multi-Information Diffusion Models** - Inspired by DiffusionVel paper
2. **Advanced Neural Architectures** - Multi-scale convolutions, cross-attention, and transformers
3. **Ensemble Learning** - Multiple models with different architectures
4. **Physics-Informed Loss Functions** - Including MAPE, gradient smoothness, and physics constraints
5. **Test-Time Augmentation** - For improved robustness
6. **Modal Labs Integration** - Cloud GPU training infrastructure

## Key Features

### Advanced Architecture
- **Multi-Scale Convolutional Blocks**: Extract features at different scales
- **Cross-Source Attention**: Learn relationships between different seismic sources
- **Transformer Decoder**: Generate high-resolution velocity models
- **Ensemble Learning**: Combine multiple models for robust predictions

### Training Optimizations
- **Modal Labs Cloud GPU**: Scalable training on A100 GPUs
- **Advanced Data Augmentation**: Time-frequency domain augmentations
- **Physics-Informed Losses**: MAPE + L2 + gradient smoothness
- **Adaptive Learning**: Cosine annealing with warm restarts

### Inference Features
- **Test-Time Augmentation**: Horizontal flips and noise injection
- **Post-Processing**: Smoothing and geological constraints
- **Batch Processing**: Efficient inference on large datasets

## File Structure

```
├── sota_framework.py      # Main framework with model architectures
├── data_utils.py         # Data loading and preprocessing utilities
├── inference.py          # Inference pipeline and prediction generation
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Saved model checkpoints (created during training)
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Modal CLI (for cloud training):
```bash
pip install modal
modal setup
```

## Usage

### 1. Data Preparation

Ensure your data is organized as follows:
```
data/
├── train/
│   ├── sample_001/
│   │   ├── receiver_data_src_1.npy
│   │   ├── receiver_data_src_75.npy
│   │   ├── receiver_data_src_150.npy
│   │   ├── receiver_data_src_225.npy
│   │   ├── receiver_data_src_300.npy
│   │   └── vp_model.npy
│   └── ...
└── test/
    ├── sample_001/
    │   ├── receiver_data_src_1.npy
    │   ├── receiver_data_src_75.npy
    │   ├── receiver_data_src_150.npy
    │   ├── receiver_data_src_225.npy
    │   └── receiver_data_src_300.npy
    └── ...
```

### 2. Training (Modal Labs)

The framework is designed to run on Modal Labs for scalable cloud training:

```python
# Edit configuration in sota_framework.py
config_dict = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'num_models': 3,
}

# Run training
python sota_framework.py
```

### 3. Local Training (Alternative)

For local training, you can extract the model classes and training logic:

```python
from sota_framework import EnsembleVelocityModel, Config, VelocityInversionLoss
from data_utils import create_data_loaders

# Initialize configuration
config = Config()

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_path="./data/train",
    batch_size=config.batch_size
)

# Initialize model and training components
model = EnsembleVelocityModel(config)
criterion = VelocityInversionLoss(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Training loop (implement based on your needs)
```

### 4. Inference

Generate predictions for test data:

```bash
python inference.py \
    --model_path ./models/best_ensemble_model.pth \
    --test_data_path ./data/test \
    --output_path ./submission.npz \
    --batch_size 8 \
    --use_tta \
    --post_process
```

### 5. Quick Test

Test a single sample prediction:

```python
from inference import quick_test_prediction
quick_test_prediction()
```

## Model Architecture Details

### SeismicEncoder
- **Multi-scale convolutions** for each seismic source
- **Cross-source attention** to learn inter-source relationships
- **Temporal-spatial fusion** layers
- **Adaptive pooling** to fixed output size

### VelocityDecoder
- **Transformer decoder** architecture
- **Positional embeddings** for spatial locations
- **Multi-layer perceptrons** for final velocity prediction
- **Physics-aware activations** (SoftPlus + offset)

### EnsembleModel
- **Multiple independent models** with different initializations
- **Learned ensemble weighting**
- **Robust prediction averaging**

## Loss Function

The framework uses a multi-component loss function:

```python
total_loss = (
    mape_loss * 1.0 +      # Primary evaluation metric
    l2_loss * 0.5 +        # Stability
    gradient_loss * 0.3 +  # Smoothness
    physics_loss * 0.2     # Physics consistency
)
```

## Configuration Options

Key configuration parameters in `Config` class:

- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `num_epochs`: Training epochs (default: 100)
- `num_models`: Ensemble size (default: 5)
- `embed_dim`: Model embedding dimension (default: 512)
- `num_heads`: Attention heads (default: 8)
- `num_layers`: Transformer layers (default: 12)

## Performance Optimizations

1. **Data Loading**: Parallel data loading with configurable workers
2. **Memory Management**: Optional data caching for faster training
3. **GPU Utilization**: Optimized for A100 GPUs on Modal Labs
4. **Gradient Clipping**: Prevents exploding gradients
5. **Mixed Precision**: Automatic mixed precision training support

## Expected Results

Based on the SOTA techniques implemented:

- **High Accuracy**: Ensemble approach with advanced architectures
- **Robustness**: Test-time augmentation and post-processing
- **Efficiency**: Optimized training and inference pipelines
- **Scalability**: Cloud-based training on powerful GPUs

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in configuration
- Disable data caching: `cache_data=False`
- Use gradient checkpointing

### Training Issues
- Check data paths and formats
- Verify GPU availability
- Monitor learning rates and losses

### Inference Issues
- Ensure model checkpoint exists
- Check input data format matches training
- Verify output data types (must be float64)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This framework incorporates ideas from several recent research papers:

1. **DiffusionVel** - Multi-information integrated velocity inversion using generative diffusion models
2. **Discrete Adjoint Methods** - Efficient gradient computation techniques
3. **Physics-Informed Neural Networks** - Physics constraints in deep learning
4. **Transformer Architectures** - Attention mechanisms for spatial data
5. **Ensemble Learning** - Multiple model combination strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For questions or issues, please open a GitHub issue or contact the maintainers.