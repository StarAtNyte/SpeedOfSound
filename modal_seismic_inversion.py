"""
Complete SOTA Seismic Velocity Inversion Framework for Modal Labs
All-in-one script for training, inference, and submission generation
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob
import json
from tqdm import tqdm
import wandb
from scipy import ndimage
from scipy.signal import butter, filtfilt

# Modal setup
app = modal.App("seismic-velocity-inversion-sota")

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "wandb",
        "einops",
        "timm",
        "diffusers",
        "transformers",
        "accelerate",
        "xformers",
        "pillow",
        "opencv-python-headless",
        "tensorboard"
    ])
)

# Modal volume for persistent data storage
volume = modal.Volume.from_name("speed-structure-vol", create_if_missing=True)

@dataclass
class Config:
    """Configuration for the SOTA framework"""
    # Data configuration
    input_shape: Tuple[int, int] = (10001, 31)  # Seismic receiver data shape
    output_shape: Tuple[int, int] = (300, 1259)  # Velocity model shape
    num_sources: int = 5  # Number of seismic sources
    
    # Model configuration
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6  # Reduced for efficiency
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 8  # Increased for speed
    learning_rate: float = 1e-4
    num_epochs: int = 30  # Reduced epochs for faster training
    warmup_steps: int = 1000
    
    # Model configuration - Single optimized model
    num_models: int = 1  # Single model for speed
    
    # Augmentation - Reduced for speed
    use_augmentation: bool = True
    noise_level: float = 0.02
    augment_probability: float = 0.3  # Reduce augmentation frequency
    
    # Paths (all relative to /vol)
    data_path: str = "/vol"
    model_path: str = "/vol/models"
    results_path: str = "/vol/results"
    logs_path: str = "/vol/logs"


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolutional block for feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels//4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels//4, 7, padding=3)
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)
        
        out = torch.cat([x1, x3, x5, x7], dim=1)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class SeismicEncoder(nn.Module):
    """Advanced encoder for seismic data processing"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Input projection for each source
        self.source_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.GELU(),
                MultiScaleConvBlock(64, 128),
                MultiScaleConvBlock(128, 256),
                nn.AdaptiveAvgPool2d((32, 8)),  # Fixed output size
            ) for _ in range(config.num_sources)
        ])
        
        # Cross-source attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * config.num_sources, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 4)),
        )
        
    def forward(self, source_data: List[torch.Tensor]) -> torch.Tensor:
        # Process each source independently
        source_features = []
        for i, data in enumerate(source_data):
            # Add channel dimension if needed
            if data.dim() == 3:
                data = data.unsqueeze(1)
            
            features = self.source_projections[i](data)
            source_features.append(features)
        
        # Apply cross-source attention
        batch_size = source_features[0].size(0)
        
        # Prepare for attention (flatten spatial dims)
        attention_features = []
        for features in source_features:
            b, c, h, w = features.shape
            flat_features = features.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
            attention_features.append(flat_features)
        
        # Cross-attention between sources
        attended_features = []
        for i, query in enumerate(attention_features):
            # Concatenate all other sources as context
            context = torch.cat([attention_features[j] for j in range(len(attention_features))], dim=1)
            attended, _ = self.cross_attention(query, context, context)
            attended_features.append(attended)
        
        # Reshape back to spatial format
        spatial_features = []
        for i, attended in enumerate(attended_features):
            b, hw, c = attended.shape
            h, w = source_features[i].shape[2], source_features[i].shape[3]
            spatial = attended.transpose(1, 2).view(b, c, h, w)
            spatial_features.append(spatial)
        
        # Concatenate and fuse
        fused = torch.cat(spatial_features, dim=1)
        output = self.fusion_conv(fused)
        
        return output


class VelocityDecoder(nn.Module):
    """Advanced decoder for velocity model generation using CNN approach"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Calculate input features from encoder
        encoder_features = 512 * 16 * 4  # From encoder output
        
        # Initial projection
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_features, config.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Transformer layers for global context
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embed_dim * 2,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.num_layers // 2)
        ])
        
        # Upsampling network
        self.upsample_net = nn.Sequential(
            # Start with small spatial size
            nn.Linear(config.embed_dim, 256 * 8 * 16),  # 8x16 feature map
            nn.GELU(),
        )
        
        # CNN decoder layers
        self.decoder_layers = nn.ModuleList([
            # 8x16 -> 16x32
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
            ),
            # 16x32 -> 32x64
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
            ),
            # 32x64 -> 64x128
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
            ),
            # 64x128 -> 128x256
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
            ),
            # 128x256 -> 256x512
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU(),
            ),
        ])
        
        # Final layers to reach target size (300x1259)
        self.final_layers = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(4, 1, 3, padding=1),
        )
        
        # Final interpolation to exact target size
        self.target_h, self.target_w = config.output_shape
        
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_features.size(0)
        
        # Flatten encoded features
        encoded_flat = encoded_features.view(batch_size, -1)
        
        # Project to transformer dimension
        features = self.feature_proj(encoded_flat)
        features = features.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer layers for global context
        for layer in self.transformer_layers:
            features = layer(features)
        
        features = features.squeeze(1)  # Remove sequence dimension
        
        # Initial upsampling
        upsampled = self.upsample_net(features)
        upsampled = upsampled.view(batch_size, 256, 8, 16)
        
        # Progressive upsampling through CNN layers
        x = upsampled
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final convolutions
        x = self.final_layers(x)
        x = x.squeeze(1)  # Remove channel dimension
        
        # Interpolate to exact target size
        x = F.interpolate(
            x.unsqueeze(1), 
            size=(self.target_h, self.target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        # Apply activation to ensure positive velocities
        x = F.softplus(x) + 1.0  # Minimum velocity of 1.0
        
        return x


class SOTAVelocityModel(nn.Module):
    """State-of-the-art velocity inversion model"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = SeismicEncoder(config)
        self.decoder = VelocityDecoder(config)
        
    def forward(self, source_data: List[torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(source_data)
        velocity = self.decoder(encoded)
        return velocity


class EnsembleVelocityModel(nn.Module):
    """Ensemble of SOTA models"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList([
            SOTAVelocityModel(config) for _ in range(config.num_models)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(config.num_models))
        
    def forward(self, source_data: List[torch.Tensor]) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model(source_data)
            predictions.append(pred)
        
        # Weighted ensemble
        stacked_preds = torch.stack(predictions, dim=0)  # (num_models, batch, H, W)
        weights = F.softmax(self.ensemble_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting
        
        ensemble_pred = torch.sum(stacked_preds * weights, dim=0)
        return ensemble_pred


class AdvancedLoss(nn.Module):
    """Advanced loss function for seismic velocity inversion"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Primary MAPE loss (evaluation metric)
        mape = torch.mean(torch.abs((pred - target) / target))
        losses['mape'] = mape
        
        # L2 loss for stability
        l2 = F.mse_loss(pred, target)
        losses['l2'] = l2
        
        # Gradient smoothness loss
        pred_grad_x = torch.diff(pred, dim=1)
        target_grad_x = torch.diff(target, dim=1)
        pred_grad_y = torch.diff(pred, dim=2)
        target_grad_y = torch.diff(target, dim=2)
        
        grad_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        losses['gradient'] = grad_loss
        
        # Total variation loss for smoothness
        tv_loss = torch.mean(torch.abs(pred_grad_x)) + torch.mean(torch.abs(pred_grad_y))
        losses['tv'] = tv_loss
        
        # Physics-based loss (velocity increases with depth generally)
        depth_grad = torch.diff(pred, dim=2)  # Gradient along depth
        physics_loss = torch.mean(F.relu(-depth_grad))  # Penalize decreasing velocity with depth
        losses['physics'] = physics_loss
        
        # Combined loss
        total = (
            mape * 1.0 +
            l2 * 0.3 +
            grad_loss * 0.2 +
            tv_loss * 0.1 +
            physics_loss * 0.1
        )
        losses['total'] = total
        
        return losses


# Data loading and preprocessing functions
def load_seismic_sample(sample_path: str) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Load a single seismic sample"""
    source_coords = [1, 75, 150, 225, 300]
    
    receiver_data = []
    for coord in source_coords:
        file_path = os.path.join(sample_path, f"receiver_data_src_{coord}.npy")
        data = np.load(file_path).astype(np.float32)
        receiver_data.append(data)
    
    # Load velocity model if exists (training data)
    velocity_path = os.path.join(sample_path, "vp_model.npy")
    if os.path.exists(velocity_path):
        velocity_model = np.load(velocity_path).astype(np.float64)
        return receiver_data, velocity_model
    else:
        return receiver_data, None


def preprocess_seismic_data(receiver_data: List[np.ndarray], config: Config) -> List[torch.Tensor]:
    """Preprocess seismic data"""
    processed = []
    
    for data in receiver_data:
        # Apply bandpass filter
        filtered_data = apply_bandpass_filter(data)
        
        # Normalize each trace
        normalized_data = normalize_traces(filtered_data)
        
        # Convert to tensor
        tensor_data = torch.from_numpy(normalized_data).float()
        processed.append(tensor_data)
    
    return processed


def apply_bandpass_filter(data: np.ndarray, low_freq: float = 5.0, high_freq: float = 50.0) -> np.ndarray:
    """Apply bandpass filter to seismic data"""
    sampling_rate = 1000.0  # Assumed sampling rate
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered_data


def normalize_traces(data: np.ndarray) -> np.ndarray:
    """Normalize each trace independently"""
    normalized = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        trace = data[:, i]
        trace_std = np.std(trace)
        trace_mean = np.mean(trace)
        
        if trace_std > 1e-8:
            normalized[:, i] = (trace - trace_mean) / trace_std
        else:
            normalized[:, i] = trace - trace_mean
    
    return normalized


def augment_data(receiver_data: List[torch.Tensor], config: Config) -> List[torch.Tensor]:
    """Apply data augmentation"""
    if not config.use_augmentation:
        return receiver_data
    
    augmented = []
    for data in receiver_data:
        aug_data = data.clone()
        
        # Reduced augmentation frequency for speed
        aug_prob = getattr(config, 'augment_probability', 0.5)
        
        # Random noise
        if torch.rand(1) > (1 - aug_prob):
            noise = torch.randn_like(aug_data) * config.noise_level
            aug_data = aug_data + noise
        
        # Random amplitude scaling
        if torch.rand(1) > (1 - aug_prob):
            scale = torch.rand(1) * 0.4 + 0.8  # Random between 0.8 and 1.2
            aug_data = aug_data * scale
        
        # Random time shift
        if torch.rand(1) > (1 - aug_prob * 0.5):
            shift = torch.randint(-50, 51, (1,)).item()
            aug_data = torch.roll(aug_data, shift, dims=0)
        
        augmented.append(aug_data)
    
    return augmented


@app.function(
    image=image,
    gpu="A100",
    memory=64000,
    timeout=86400,  # 24 hours
    volumes={"/vol": volume}
)
def train_sota_model():
    """Complete training pipeline"""
    
    # Initialize configuration
    config = Config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.logs_path, exist_ok=True)
    
    # Initialize wandb (optional)
    use_wandb = False
    try:
        wandb.init(
            project="seismic-velocity-inversion-sota",
            config=config.__dict__,
            dir=config.logs_path,
            mode="disabled"  # Disable wandb for now
        )
        use_wandb = True
        print("WandB initialized successfully")
    except Exception as e:
        print(f"WandB initialization failed: {e}")
        print("Continuing training without WandB logging")
        use_wandb = False
    
    # Load training data
    train_data_path = os.path.join(config.data_path, "train")
    sample_paths = glob(os.path.join(train_data_path, "*"))
    print(f"Found {len(sample_paths)} training samples")
    
    # Split data into train/val
    val_split = 0.1
    val_size = int(len(sample_paths) * val_split)
    np.random.shuffle(sample_paths)
    val_paths = sample_paths[:val_size]
    train_paths = sample_paths[val_size:]
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Initialize model - Single optimized model
    if config.num_models == 1:
        model = SOTAVelocityModel(config).to(device)
        print("Using single optimized SOTA model")
    else:
        model = EnsembleVelocityModel(config).to(device)
        print(f"Using ensemble of {config.num_models} models")
    criterion = AdvancedLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(config.model_path, "best_model.pth")
    latest_checkpoint_path = os.path.join(config.model_path, "latest_checkpoint.pth")
    
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        print("Starting training from scratch")
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_mapes = []
        
        # Shuffle training data
        np.random.shuffle(train_paths)
        
        # Training batches
        num_batches = len(train_paths) // config.batch_size
        progress_bar = tqdm(range(num_batches), desc="Training")
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + config.batch_size
            batch_paths = train_paths[start_idx:end_idx]
            
            # Load batch data
            batch_receiver = [[] for _ in range(config.num_sources)]
            batch_targets = []
            
            for sample_path in batch_paths:
                receiver_data, velocity_model = load_seismic_sample(sample_path)
                processed_receiver = preprocess_seismic_data(receiver_data, config)
                augmented_receiver = augment_data(processed_receiver, config)
                
                for i, source_data in enumerate(augmented_receiver):
                    batch_receiver[i].append(source_data)
                
                batch_targets.append(torch.from_numpy(velocity_model).float())
            
            # Stack batch data
            stacked_receiver = []
            for i in range(config.num_sources):
                stacked_receiver.append(torch.stack(batch_receiver[i]).to(device))
            
            targets = torch.stack(batch_targets).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(stacked_receiver)
            
            # Compute losses
            losses = criterion(predictions, targets)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Record losses
            train_losses.append(losses['total'].item())
            train_mapes.append(losses['mape'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'MAPE': f"{losses['mape'].item():.4f}"
            })
            
            # Log to wandb if available
            if use_wandb:
                wandb.log({
                    'train/batch_loss': losses['total'].item(),
                    'train/batch_mape': losses['mape'].item(),
                    'train/l2_loss': losses['l2'].item(),
                    'train/gradient_loss': losses['gradient'].item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_mapes = []
        
        with torch.no_grad():
            val_batches = len(val_paths) // config.batch_size
            
            for batch_idx in range(val_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = start_idx + config.batch_size
                batch_paths = val_paths[start_idx:end_idx]
                
                # Load validation batch (no augmentation)
                batch_receiver = [[] for _ in range(config.num_sources)]
                batch_targets = []
                
                for sample_path in batch_paths:
                    receiver_data, velocity_model = load_seismic_sample(sample_path)
                    processed_receiver = preprocess_seismic_data(receiver_data, config)
                    
                    for i, source_data in enumerate(processed_receiver):
                        batch_receiver[i].append(source_data)
                    
                    batch_targets.append(torch.from_numpy(velocity_model).float())
                
                # Stack batch data
                stacked_receiver = []
                for i in range(config.num_sources):
                    stacked_receiver.append(torch.stack(batch_receiver[i]).to(device))
                
                targets = torch.stack(batch_targets).to(device)
                
                # Forward pass
                predictions = model(stacked_receiver)
                losses = criterion(predictions, targets)
                
                val_losses.append(losses['total'].item())
                val_mapes.append(losses['mape'].item())
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_mape = np.mean(train_mapes)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_mape = np.mean(val_mapes) if val_mapes else float('inf')
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train MAPE: {avg_train_mape:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val MAPE: {avg_val_mape:.4f}")
        
        # Log epoch metrics if wandb available
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': avg_train_loss,
                'train/epoch_mape': avg_train_mape,
                'val/epoch_loss': avg_val_loss,
                'val/epoch_mape': avg_val_mape
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_mape': avg_val_mape
            }, best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save latest checkpoint every epoch for resuming
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config.__dict__,
            'epoch': epoch,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }, latest_checkpoint_path)
        
        # Save numbered checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.model_path, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    if use_wandb:
        wandb.finish()
    print(f"Training completed! Best model saved at: {best_model_path}")
    return best_model_path


@app.function(
    image=image,
    gpu="A100",
    memory=64000,
    timeout=86400,  # 24 hours
    volumes={"/vol": volume}
)
def resume_training():
    """Resume training from latest checkpoint"""
    
    # Initialize configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    latest_checkpoint_path = os.path.join(config.model_path, "latest_checkpoint.pth")
    
    if not os.path.exists(latest_checkpoint_path):
        print("No checkpoint found. Starting fresh training...")
        return train_sota_model.remote()
    
    print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
    return train_sota_model.remote()


@app.function(
    image=image,
    gpu="A100",
    memory=32000,
    timeout=3600,
    volumes={"/vol": volume}
)
def generate_submission_from_checkpoint(checkpoint_name: str = "best_model.pth"):
    """Generate predictions from a specific checkpoint"""
    
    # Initialize configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model from specified checkpoint
    model_path = os.path.join(config.model_path, checkpoint_name)
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        print("Available checkpoints:")
        for file in os.listdir(config.model_path):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model - Single optimized model
    if config.num_models == 1:
        model = SOTAVelocityModel(config).to(device)
        print("Loading single optimized SOTA model")
    else:
        model = EnsembleVelocityModel(config).to(device)
        print(f"Loading ensemble of {config.num_models} models")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch_info = checkpoint.get('epoch', 'Unknown')
    val_mape = checkpoint.get('val_mape', 'N/A')
    print(f"Model loaded successfully. Epoch: {epoch_info}, Val MAPE: {val_mape}")
    
    # Load test data
    test_data_path = os.path.join(config.data_path, "test")
    test_sample_paths = glob(os.path.join(test_data_path, "*"))
    print(f"Found {len(test_sample_paths)} test samples")
    
    # Generate predictions
    predictions = {}
    
    with torch.no_grad():
        for sample_path in tqdm(test_sample_paths, desc="Generating predictions"):
            sample_id = os.path.basename(sample_path)
            
            # Load and preprocess data
            receiver_data, _ = load_seismic_sample(sample_path)
            processed_receiver = preprocess_seismic_data(receiver_data, config)
            
            # Convert to batch format
            batch_receiver = []
            for source_data in processed_receiver:
                batch_receiver.append(source_data.unsqueeze(0).to(device))
            
            # Generate base prediction
            pred_velocity = model(batch_receiver)
            
            # Test-time augmentation
            tta_predictions = [pred_velocity]
            
            # Horizontal flip TTA
            flipped_receiver = []
            for source_data in batch_receiver:
                flipped = torch.flip(source_data, dims=[2])  # Flip receiver dimension
                flipped_receiver.append(flipped)
            
            pred_flipped = model(flipped_receiver)
            pred_flipped = torch.flip(pred_flipped, dims=[2])  # Flip back
            tta_predictions.append(pred_flipped)
            
            # Noise injection TTA
            for _ in range(2):
                noisy_receiver = []
                for source_data in batch_receiver:
                    noise = torch.randn_like(source_data) * 0.01
                    noisy = source_data + noise
                    noisy_receiver.append(noisy)
                
                pred_noisy = model(noisy_receiver)
                tta_predictions.append(pred_noisy)
            
            # Average TTA predictions
            final_pred = torch.mean(torch.stack(tta_predictions), dim=0)
            final_pred = final_pred.squeeze(0).cpu().numpy()
            
            # Post-processing
            final_pred = np.clip(final_pred, 1.0, 8.0)  # Reasonable velocity bounds
            
            # Light smoothing
            final_pred = ndimage.gaussian_filter(final_pred, sigma=0.5)
            
            # Ensure correct data type
            final_pred = final_pred.astype(np.float64)
            
            predictions[sample_id] = final_pred
    
    # Create submission file with checkpoint info in name
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    submission_filename = f"submission_epoch_{checkpoint_epoch}_{checkpoint_name.replace('.pth', '')}.npz"
    submission_path = os.path.join(config.results_path, submission_filename)
    print(f"Creating submission file: {submission_path}")
    
    # Verify predictions format
    expected_shape = (300, 1259)
    for sample_id, pred in predictions.items():
        if pred.shape != expected_shape:
            raise ValueError(f"Prediction for {sample_id} has incorrect shape: {pred.shape}")
        if pred.dtype != np.float64:
            raise ValueError(f"Prediction for {sample_id} has incorrect dtype: {pred.dtype}")
    
    # Save submission
    np.savez(submission_path, **predictions)
    
    # Print statistics
    all_values = np.concatenate([pred.flatten() for pred in predictions.values()])
    print(f"\nSubmission Statistics:")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Epoch: {checkpoint_epoch}")
    print(f"  Number of samples: {len(predictions)}")
    print(f"  Prediction shape: {expected_shape}")
    print(f"  Velocity range: [{all_values.min():.4f}, {all_values.max():.4f}]")
    print(f"  Mean velocity: {all_values.mean():.4f}")
    print(f"  Std velocity: {all_values.std():.4f}")
    
    print(f"\nSubmission file created: {submission_path}")
    return submission_path


@app.function(
    image=image,
    gpu="A100", 
    memory=32000,
    timeout=3600,
    volumes={"/vol": volume}
)
def generate_submission():
    """Generate predictions using the best model checkpoint"""
    return generate_submission_from_checkpoint.remote("best_model.pth")


@app.function(
    image=image,
    volumes={"/vol": volume}
)
def analyze_data():
    """Analyze the training and test data"""
    
    print("Analyzing data structure...")
    
    # Check data paths
    train_path = "/vol/train"
    test_path = "/vol/test"
    
    if os.path.exists(train_path):
        train_samples = glob(os.path.join(train_path, "*"))
        print(f"Training samples: {len(train_samples)}")
        
        # Analyze first sample
        if train_samples:
            sample_path = train_samples[0]
            receiver_data, velocity_model = load_seismic_sample(sample_path)
            
            print(f"\nSample analysis:")
            print(f"  Sample ID: {os.path.basename(sample_path)}")
            print(f"  Number of sources: {len(receiver_data)}")
            print(f"  Receiver data shape: {receiver_data[0].shape}")
            print(f"  Velocity model shape: {velocity_model.shape}")
            print(f"  Velocity range: [{velocity_model.min():.4f}, {velocity_model.max():.4f}]")
    else:
        print(f"Training data not found at: {train_path}")
    
    if os.path.exists(test_path):
        test_samples = glob(os.path.join(test_path, "*"))
        print(f"Test samples: {len(test_samples)}")
        
        # Analyze first test sample
        if test_samples:
            sample_path = test_samples[0]
            receiver_data, _ = load_seismic_sample(sample_path)
            
            print(f"\nTest sample analysis:")
            print(f"  Sample ID: {os.path.basename(sample_path)}")
            print(f"  Number of sources: {len(receiver_data)}")
            print(f"  Receiver data shape: {receiver_data[0].shape}")
    else:
        print(f"Test data not found at: {test_path}")


# Main execution functions
@app.local_entrypoint()
def main():
    """Main execution function"""
    
    print("🚀 Starting SOTA Seismic Velocity Inversion Pipeline")
    print("=" * 60)
    
    # Step 1: Analyze data
    print("\n📊 Step 1: Analyzing data...")
    analyze_data.remote()
    
    # Step 2: Train/Resume model
    print("\n🏋️ Step 2: Training/Resuming SOTA model...")
    best_model_path = resume_training.remote()
    print(f"✅ Training completed. Best model: {best_model_path}")
    
    # Step 3: Generate submission
    print("\n🎯 Step 3: Generating submission...")
    submission_path = generate_submission.remote()
    print(f"✅ Submission created: {submission_path}")
    
    print("\n🏆 Pipeline completed successfully!")
    print("Your submission file is ready for upload to the competition platform.")


if __name__ == "__main__":
    # For development/testing
    config = Config()
    print("SOTA Seismic Velocity Inversion Framework")
    print(f"Configuration: {config}")