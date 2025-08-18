# Simple Seismic Velocity Inversion

A streamlined framework for seismic velocity inversion using enhanced U-Net architecture on Modal Labs.

## Overview

This framework implements:
- **Enhanced U-Net with Source Attention** - Multi-source seismic data processing
- **Focal MAPE Loss** - Focus on hard examples for better accuracy
- **Hard Example Mining** - Improved convergence on difficult samples
- **Stochastic Weight Averaging** - Better model generalization
- **Modal Labs Integration** - Cloud GPU training on A100s

## File Structure

```
├── modal_seismic_inversion.py  # Main training and inference pipeline
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup Modal CLI:
```bash
pip install modal
modal setup
```

## Usage

### Data Structure
Organize data in `/vol` directory on Modal:
```
/vol/
├── train/
│   └── sample_xxx/
│       ├── receiver_data_src_*.npy
│       └── vp_model.npy
└── test/
    └── sample_xxx/
        └── receiver_data_src_*.npy
```

### Training
```bash
modal run modal_seismic_inversion.py
```

### Configuration
Key parameters in `Config` class:
- `batch_size`: 8 (optimized for A100)
- `learning_rate`: 2e-5 (fine-tuned)
- `num_epochs`: 150 (with early stopping)
- `focal_alpha/gamma`: Focus on hard examples
- `use_swa`: Stochastic weight averaging

### Model Architecture
- **Enhanced U-Net** with source attention
- **96 initial features** for better representation
- **Source encoders** for multi-source processing
- **Focal MAPE loss**
