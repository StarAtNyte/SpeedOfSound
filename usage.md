# Usage Guide for SOTA Seismic Velocity Inversion Framework

## Quick Start

### 1. Prerequisites

Ensure you have Modal Labs account set up and authenticated:
```bash
pip install modal
modal setup
```

### 2. Run the Complete Pipeline

The entire framework is contained in a single Modal script. To run the complete pipeline:

```bash
modal run modal_seismic_inversion.py
```

This will execute all three steps:
1. **Data Analysis** - Analyze the training and test data structure
2. **Model Training** - Train the SOTA ensemble model on A100 GPUs
3. **Submission Generation** - Generate predictions for test data with TTA

## Pipeline Steps

### Step 1: Data Analysis
```bash
# To run only data analysis
modal run modal_seismic_inversion.py::analyze_data
```

This analyzes:
- Training data structure and sample counts
- Test data structure and sample counts  
- Data shapes and value ranges
- Sample file formats

### Step 2: Model Training
```bash
# To run only training
modal run modal_seismic_inversion.py::train_sota_model
```

Training features:
- **Ensemble of 3 models** with different architectures
- **Multi-scale convolutional blocks** for feature extraction
- **Cross-source attention** between seismic sources
- **Advanced loss function** (MAPE + L2 + gradient + physics)
- **A100 GPU optimization** with 64GB memory
- **WandB logging** for experiment tracking
- **Automatic checkpointing** and best model saving

Training outputs:
- Model checkpoints saved to `/vol/models/`
- Training logs to `/vol/logs/` 
- WandB experiment tracking

### Step 3: Submission Generation

#### Generate submission from best model:
```bash
modal run modal_seismic_inversion.py::generate_submission
```

#### Generate submission from specific checkpoint:
```bash
# From epoch 10 checkpoint
modal run modal_seismic_inversion.py::generate_submission_from_checkpoint --checkpoint_name "checkpoint_epoch_10.pth"

# From latest checkpoint
modal run modal_seismic_inversion.py::generate_submission_from_checkpoint --checkpoint_name "latest_checkpoint.pth"

# From epoch 15 checkpoint
modal run modal_seismic_inversion.py::generate_submission_from_checkpoint --checkpoint_name "checkpoint_epoch_15.pth"
```

Inference features:
- **Test-time augmentation** (horizontal flip + noise injection)
- **Post-processing** with smoothing and velocity bounds
- **Batch processing** for efficient inference
- **Submission file** in required .npz format
- **Automatic checkpoint detection** and validation
- **Named submission files** with epoch information

## Configuration

All configuration is handled in the `Config` class:

```python
@dataclass
class Config:
    # Data configuration
    input_shape: Tuple[int, int] = (10001, 31)  # Seismic receiver data
    output_shape: Tuple[int, int] = (300, 1259)  # Velocity model
    num_sources: int = 5  # Number of seismic sources
    
    # Model configuration
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 4  # Optimized for A100
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Ensemble configuration
    num_models: int = 3
```

## Data Structure

The framework expects data in Modal volume `speed-structure-vol`:

```
/vol/
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

## Model Architecture

### SeismicEncoder
- **Multi-scale convolutions** (1x1, 3x3, 5x5, 7x7) for each source
- **Cross-source attention** to learn relationships between sources
- **Adaptive pooling** to fixed output size
- **Batch normalization and GELU activation**

### VelocityDecoder  
- **Transformer encoder layers** for global context
- **Progressive upsampling** through CNN decoder
- **Final interpolation** to exact target size (300x1259)
- **SoftPlus activation** to ensure positive velocities

### EnsembleModel
- **3 independent models** with different initializations
- **Learnable ensemble weights** 
- **Weighted averaging** for final predictions

## Loss Function

Multi-component physics-informed loss:

```python
total_loss = (
    mape_loss * 1.0 +      # Primary evaluation metric
    l2_loss * 0.3 +        # Stability  
    gradient_loss * 0.2 +  # Smoothness
    tv_loss * 0.1 +        # Total variation
    physics_loss * 0.1     # Physics constraints
)
```

## Performance Features

### Training Optimizations
- **Modal A100 GPU** with 64GB memory
- **Batch size 4** optimized for memory
- **Gradient clipping** to prevent exploding gradients
- **Cosine annealing** learning rate schedule
- **Data augmentation** (noise, scaling, time shifts)
- **WandB integration** for experiment tracking

### Inference Optimizations
- **Test-time augmentation** for robustness
- **Batch processing** for efficiency  
- **Post-processing** with geological constraints
- **Memory-efficient** inference pipeline

## Expected Output

After running the complete pipeline:

1. **Training logs** in WandB dashboard
2. **Best model checkpoint** at `/vol/models/best_model.pth`
3. **Submission file** at `/vol/results/submission.npz`
4. **Performance statistics** printed to console

## Monitoring

### WandB Integration
The framework automatically logs:
- Training/validation losses
- Individual loss components (MAPE, L2, gradient, etc.)
- Learning rates
- Model performance metrics

### Console Output
Real-time monitoring includes:
- Epoch progress and losses
- Validation metrics
- Model saving notifications
- Data analysis results
- Submission statistics

## Troubleshooting

### Memory Issues
- Reduce `batch_size` from 4 to 2 or 1
- Decrease `num_models` in ensemble
- Reduce `embed_dim` or `num_layers`

### Training Issues
- Check data paths in Modal volume
- Verify GPU availability
- Monitor WandB for training curves
- Check for NaN losses

### Inference Issues  
- Ensure best model checkpoint exists
- Verify test data format matches training
- Check submission file data types (must be float64)

## Modal Labs Specific Notes

### Volume Setup
The framework uses Modal volume `speed-structure-vol` for:
- Persistent data storage
- Model checkpoints
- Results and logs

### GPU Requirements
- **Recommended**: A100 with 40GB+ memory
- **Minimum**: A10G with modifications to batch_size
- **Timeout**: 2 hours for training, 1 hour for inference

### Image Dependencies
All required packages are automatically installed in the Modal image:
- PyTorch 2.0+
- Transformers and diffusers
- Scientific computing stack
- WandB for logging

## Competition Submission

The final submission file will be created at `/vol/results/submission.npz` with:
- **Format**: .npz file with sample_id keys
- **Shape**: Each prediction is (300, 1259)
- **Data type**: numpy.float64
- **Evaluation**: Ready for MAPE scoring

To download the submission file from Modal volume, use Modal CLI or dashboard.