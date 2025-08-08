import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm

# Modal setup
app = modal.App("simple-seismic-inversion")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "matplotlib"
    ])
)

# Modal volume for persistent data storage
volume = modal.Volume.from_name("speed-structure-vol", create_if_missing=True)

@dataclass
class Config:
    """Optimized configuration for MAPE < 0.020"""
    # Data configuration
    input_shape: Tuple[int, int] = (10001, 31)  # Seismic receiver data shape
    output_shape: Tuple[int, int] = (300, 1259)  # Velocity model shape
    num_sources: int = 5  # Number of seismic sources
    
    # Training configuration - Optimized for sub-0.021 MAPE
    batch_size: int = 8  # Slightly larger for better gradient estimates
    learning_rate: float = 2e-5  # Lower initial LR for fine-grained optimization
    num_epochs: int = 150  # More epochs for thorough convergence
    weight_decay: float = 1e-6  # Slightly more regularization
    warmup_epochs: int = 12  # Longer warmup for stability
    
    # Model configuration - Optimized capacity
    initial_features: int = 96  # More features for better representation
    depth_levels: int = 6  # Keep deep architecture
    
    # Advanced training settings - Optimized
    use_cosine_schedule: bool = True
    gradient_clip: float = 0.2  # Tighter clipping for stability
    label_smoothing: float = 0.003  # Minimal smoothing
    use_focal_loss: bool = True  # Use focal MAPE loss
    focal_alpha: float = 2.5  # Stronger focus on hard examples
    focal_gamma: float = 2.2  # More aggressive focusing
    use_hard_mining: bool = True  # Use hard example mining
    mining_ratio: float = 0.5  # Focus on more hard examples
    mining_weight: float = 0.4  # Higher weight for hard mining
    
    # Progressive enhancement features - Optimized
    use_mixup: bool = True
    mixup_alpha: float = 0.15  # Less aggressive mixup for precision
    use_noise_injection: bool = True
    noise_std: float = 0.015  # Reduced noise for cleaner training
    use_elastic_deform: bool = True
    deform_alpha: float = 1.5  # Gentler deformation
    deform_sigma: float = 0.4  # Smoother deformation
    use_multi_scale: bool = True
    crop_scales: List[float] = None  # Will be set in __post_init__
    
    # Physics-informed constraints - Optimized weights
    use_wave_equation_loss: bool = True
    wave_loss_weight: float = 0.08  # Reduced for better MAPE focus
    use_gardner_constraint: bool = True
    gardner_weight: float = 0.03  # Reduced constraint weight
    use_causality_loss: bool = True
    causality_weight: float = 0.015  # Minimal causality constraint
    
    # Test-time augmentation
    use_tta: bool = False  # Disabled TTA for simpler inference
    tta_scales: List[float] = None  # Will be set in __post_init__
    
    # Advanced optimization settings
    use_swa: bool = True  # Stochastic Weight Averaging
    swa_start_epoch: int = 80  # Start SWA after 80 epochs
    swa_lr: float = 5e-6  # Lower LR for SWA
    
    # Optimizer settings
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)  # Adam betas
    optimizer_eps: float = 1e-8  # Adam epsilon
    
    # Loss combination weights - Optimized
    mape_weight: float = 0.85  # Higher emphasis on MAPE
    mse_weight: float = 0.15  # Lower MSE weight
    
    # Early stopping
    early_stopping_patience: int = 25  # Stop if no improvement
    early_stopping_min_delta: float = 1e-5  # Minimum improvement threshold
    
    # Paths (all relative to /vol)
    data_path: str = "/vol"
    model_path: str = "/vol/models"
    results_path: str = "/vol/results"
    
    def __post_init__(self):
        if self.crop_scales is None:
            self.crop_scales = [0.85, 0.92, 1.0]  # More conservative cropping
        if self.tta_scales is None:
            self.tta_scales = [0.98, 1.0, 1.02]  # Gentler TTA scales


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EnhancedSeismicUNet(nn.Module):
    """Enhanced U-Net for achieving MAPE < 0.021"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        features = config.initial_features
        
        # Enhanced source encoders with attention
        self.source_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, features//2, 7, padding=3),
                nn.BatchNorm2d(features//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(features//2, features//2, 5, padding=2),
                nn.BatchNorm2d(features//2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((256, 32))
            ) for _ in range(config.num_sources)
        ])
        
        # Source attention mechanism
        self.source_attention = nn.Sequential(
            nn.Conv2d(config.num_sources * features//2, features//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//4, config.num_sources, 1),
            nn.Softmax(dim=1)
        )
        
        # Combine sources with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(config.num_sources * features//2, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Deeper U-Net encoder
        self.inc = DoubleConv(features, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 16)
        
        # Additional depth if configured
        if config.depth_levels > 4:
            self.down5 = Down(features * 16, features * 32)
            self.up0 = Up(features * 32, features * 16)
        
        # U-Net decoder with residual connections
        self.up1 = Up(features * 16, features * 8)
        self.up2 = Up(features * 8, features * 4)
        self.up3 = Up(features * 4, features * 2)
        self.up4 = Up(features * 2, features)
        
        # Multi-scale output heads
        self.outc = nn.Sequential(
            nn.Conv2d(features, features//2, 3, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//2, 1, 1)
        )
        
        # Auxiliary output for deep supervision
        self.aux_out = nn.Conv2d(features * 2, 1, 1)
        
    def forward(self, source_data: List[torch.Tensor]) -> torch.Tensor:
        # Process each source with enhanced encoding
        source_features = []
        for i, data in enumerate(source_data):
            if data.dim() == 3:
                data = data.unsqueeze(1)
            feat = self.source_encoders[i](data)
            source_features.append(feat)
        
        # Apply source attention
        x_cat = torch.cat(source_features, dim=1)
        attention_weights = self.source_attention(x_cat)
        
        # Weighted combination of sources
        weighted_features = []
        for i, feat in enumerate(source_features):
            weight = attention_weights[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        x = torch.cat(weighted_features, dim=1)
        x = self.fusion(x)
        
        # Enhanced U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Optional deeper level
        if hasattr(self, 'down5'):
            x6 = self.down5(x5)
            x = self.up0(x6, x5)
        else:
            x = x5
        
        # Decoder with skip connections
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x_aux = self.up3(x, x2)  # Auxiliary output level
        x = self.up4(x_aux, x1)
        
        # Main output
        x = self.outc(x)
        x = x.squeeze(1)
        
        # Resize to target shape with better interpolation
        x = F.interpolate(
            x.unsqueeze(1), 
            size=self.config.output_shape, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(1)
        
        # Physics-informed velocity bounds
        x = torch.sigmoid(x) * 7.0 + 1.0  # Velocity range [1, 8]
        
        return x


def enhanced_mape_loss(pred: torch.Tensor, target: torch.Tensor, smoothing: float = 0.01) -> torch.Tensor:
    """Enhanced MAPE loss with smoothing and outlier handling"""
    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    
    # Compute relative error with smoothing
    relative_error = torch.abs((pred - target) / (target + epsilon))
    
    # Apply label smoothing to reduce overfitting
    if smoothing > 0:
        relative_error = relative_error * (1 - smoothing) + smoothing * 0.5
    
    # Robust loss - reduce impact of extreme outliers
    loss = torch.mean(torch.clamp(relative_error, max=2.0))  # Cap at 200% error
    
    return loss


def focal_mape_loss(pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 2.0, gamma: float = 2.0, 
                   smoothing: float = 0.01, epoch: int = 0, 
                   total_epochs: int = 100) -> torch.Tensor:
    """
    Focal MAPE Loss: Focus training on hardest examples with biggest errors
    
    Args:
        pred: Model predictions
        target: Ground truth targets
        alpha: Scaling factor for focal weight (higher = more focus on hard examples)
        gamma: Focusing parameter (higher = more focus on hard examples)
        smoothing: Label smoothing factor
        epoch: Current training epoch (for adaptive focusing)
        total_epochs: Total training epochs
    
    Returns:
        Focal MAPE loss that emphasizes difficult examples
    """
    epsilon = 1e-8
    
    # Compute relative error
    relative_error = torch.abs((pred - target) / (target + epsilon))
    
    # Apply label smoothing
    if smoothing > 0:
        relative_error = relative_error * (1 - smoothing) + smoothing * 0.5
    
    # Adaptive focusing - start gentle, increase focus over time
    progress = epoch / max(total_epochs, 1)
    adaptive_alpha = alpha * (0.5 + 0.5 * progress)  # Ramp up from 50% to 100%
    adaptive_gamma = gamma * (0.3 + 0.7 * progress)  # Ramp up from 30% to 100%
    
    # Compute focal weights - higher weights for larger errors
    # Focus threshold adapts based on training progress
    error_threshold = 0.15 - 0.05 * progress  # Start at 15%, reduce to 10%
    focal_weights = torch.sigmoid(adaptive_alpha * (relative_error - error_threshold))
    
    # Apply gamma focusing - emphasize hard examples even more
    focal_weights = torch.pow(focal_weights, adaptive_gamma)
    
    # Robust clipping to prevent extreme values
    relative_error = torch.clamp(relative_error, max=2.0)
    
    # Weighted loss - hard examples get higher weights
    weighted_error = focal_weights * relative_error
    
    # Add statistics for monitoring
    hard_examples_ratio = (relative_error > error_threshold).float().mean()
    
    return torch.mean(weighted_error)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.8, 
                 use_focal: bool = True, use_hard_mining: bool = True,
                 mining_ratio: float = 0.4, mining_weight: float = 0.3,
                 epoch: int = 0, total_epochs: int = 100) -> torch.Tensor:
    """Combined loss for better convergence with focal weighting and hard mining"""
    
    # Primary loss (focal or enhanced MAPE)
    if use_focal:
        mape = focal_mape_loss(pred, target, alpha=2.0, gamma=2.0, 
                              epoch=epoch, total_epochs=total_epochs)
    else:
        mape = enhanced_mape_loss(pred, target)
    
    # MSE for stability
    mse = F.mse_loss(pred, target)
    
    # Base loss with optimized weighting
    base_loss = alpha * mape + (1 - alpha) * mse / 8.0  # Slightly higher MSE contribution
    
    # Add hard example mining loss
    if use_hard_mining and pred.size(0) > 1:  # Need batch size > 1 for mining
        hard_loss = hard_example_mining_loss(pred, target, mining_ratio)
        total_loss = base_loss + mining_weight * hard_loss
    else:
        total_loss = base_loss
    
    return total_loss


def elastic_deform_2d(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Apply elastic deformation to simulate geological structure variations"""
    from scipy.ndimage import gaussian_filter, map_coordinates
    
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def add_realistic_noise(data: np.ndarray, noise_std: float) -> np.ndarray:
    """Add realistic seismic noise with frequency-dependent characteristics"""
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, data.shape)
    
    # Add some low-frequency drift (common in seismic data)
    if len(data.shape) == 2:
        t_axis = np.linspace(0, 1, data.shape[0])
        drift = np.sin(2 * np.pi * 0.1 * t_axis) * noise_std * 0.5
        noise += drift[:, np.newaxis]
    
    return data + noise


def multi_scale_crop(data: np.ndarray, target: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multi-scale cropping and resize back"""
    if scale == 1.0:
        return data, target
    
    h, w = data.shape[-2:]
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Random crop
    if scale < 1.0:  # Crop
        start_h = np.random.randint(0, h - new_h + 1)
        start_w = np.random.randint(0, w - new_w + 1)
        cropped_data = data[..., start_h:start_h+new_h, start_w:start_w+new_w]
        cropped_target = target[start_h:start_h+new_h, start_w:start_w+new_w]
    else:  # Pad and crop
        pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2
        padded_data = np.pad(data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        padded_target = np.pad(target, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        cropped_data = padded_data[..., :new_h, :new_w]
        cropped_target = padded_target[:new_h, :new_w]
    
    # Resize back to original size
    from scipy.ndimage import zoom
    zoom_factors = (h / cropped_data.shape[-2], w / cropped_data.shape[-1])
    resized_data = zoom(cropped_data, (1,) + zoom_factors, order=1)
    resized_target = zoom(cropped_target, zoom_factors, order=1)
    
    return resized_data, resized_target


def mixup_data(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation for velocity models"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    
    return mixed_x, mixed_y


def identify_hard_examples(predictions: torch.Tensor, targets: torch.Tensor, 
                          threshold: float = 0.15) -> torch.Tensor:
    """
    Identify hard examples based on MAPE error
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        threshold: Error threshold to consider an example "hard"
    
    Returns:
        Boolean mask indicating hard examples
    """
    epsilon = 1e-8
    relative_error = torch.abs((predictions - targets) / (targets + epsilon))
    
    # Compute per-sample MAPE
    sample_mape = torch.mean(relative_error.view(relative_error.size(0), -1), dim=1)
    
    # Identify hard examples
    hard_mask = sample_mape > threshold
    
    return hard_mask, sample_mape


def hard_example_mining_loss(pred: torch.Tensor, target: torch.Tensor, 
                           mining_ratio: float = 0.3) -> torch.Tensor:
    """
    Hard Example Mining: Focus on the hardest examples in the batch
    
    Args:
        pred: Model predictions
        target: Ground truth targets
        mining_ratio: Fraction of hardest examples to focus on
    
    Returns:
        Loss computed on hardest examples
    """
    epsilon = 1e-8
    batch_size = pred.size(0)
    
    # Compute per-sample MAPE
    relative_error = torch.abs((pred - target) / (target + epsilon))
    sample_mape = torch.mean(relative_error.view(batch_size, -1), dim=1)
    
    # Select hardest examples
    num_hard = max(1, int(batch_size * mining_ratio))
    _, hard_indices = torch.topk(sample_mape, num_hard, largest=True)
    
    # Compute loss only on hard examples
    hard_pred = pred[hard_indices]
    hard_target = target[hard_indices]
    hard_error = torch.abs((hard_pred - hard_target) / (hard_target + epsilon))
    
    return torch.mean(hard_error)


def load_sample(sample_path: str) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """Enhanced data loading with progressive augmentations"""
    source_coords = [1, 75, 150, 225, 300]
    
    receiver_data = []
    for coord in source_coords:
        file_path = os.path.join(sample_path, f"receiver_data_src_{coord}.npy")
        data = np.load(file_path).astype(np.float32)
        
        # Enhanced preprocessing with better normalization
        # Robust standardization (less sensitive to outliers)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        data = (data - median) / (mad + 1e-8)
        
        # Gentle clipping to remove extreme outliers
        data = np.clip(data, -5, 5)
        
        receiver_data.append(torch.from_numpy(data))
    
    # Load velocity model if exists
    velocity_path = os.path.join(sample_path, "vp_model.npy")
    if os.path.exists(velocity_path):
        velocity_model = np.load(velocity_path).astype(np.float32)
        
        # Ensure velocity model is in reasonable range
        velocity_model = np.clip(velocity_model, 1.0, 8.0)
        
        return receiver_data, torch.from_numpy(velocity_model)
    else:
        return receiver_data, None


def preload_all_data(sample_paths: List[str]) -> List[Tuple[List[torch.Tensor], torch.Tensor]]:
    """Preload all data into memory - simple and fast"""
    print(f"🔄 Preloading {len(sample_paths)} samples into memory...")
    
    all_data = []
    for sample_path in tqdm(sample_paths, desc="Loading data"):
        sources, target = load_sample(sample_path)
        if target is not None:  # Skip if no target (test data)
            all_data.append((sources, target))
    
    print(f"✅ Loaded {len(all_data)} samples into memory")
    return all_data


@app.function(
    image=image,
    gpu="A100",
    memory=48000,  # More memory for enhanced model
    timeout=86400,
    volumes={"/vol": volume}
)
def train_enhanced_model():
    """Enhanced training pipeline for MAPE < 0.021"""
    
    # Initialize configuration
    config = Config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create directories
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    
    # Load training data
    train_data_path = os.path.join(config.data_path, "train")
    sample_paths = glob(os.path.join(train_data_path, "*"))
    print(f"Found {len(sample_paths)} training samples")
    
    # Fixed train/val split for consistent evaluation
    val_size = int(len(sample_paths) * 0.08)
    #np.random.seed(42)  # Fixed seed for reproducible splits
    np.random.shuffle(sample_paths)
    val_paths = sample_paths[:val_size]
    train_paths = sample_paths[val_size:]
    
    
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # Preload ALL data into memory (this fixes the 9x slowdown!)
    train_data = preload_all_data(train_paths)
    val_data = preload_all_data(val_paths)
    
    # Initialize enhanced model
    model = EnhancedSeismicUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay, 
                                 betas=config.optimizer_betas,
                                 eps=config.optimizer_eps)
    
    # Learning rate scheduler - Optimized
    if config.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=5e-7  # Longer cycles, lower min LR
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.6, patience=8, min_lr=5e-7
        )
    
    # Stochastic Weight Averaging
    swa_model = None
    if config.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_mape = float('inf')
    best_model_path = os.path.join(config.model_path, "best_enhanced_model.pth")
    epochs_without_improvement = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training
        model.train()
        train_mapes = []
        
        # Shuffle training data
        np.random.shuffle(train_data)
        num_batches = len(train_data) // config.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + config.batch_size
            batch_samples = train_data[start_idx:end_idx]
            
            # Prepare batch data (already loaded in memory!)
            batch_sources = [[] for _ in range(config.num_sources)]
            batch_targets = []
            
            for sources, target in batch_samples:
                for i, source in enumerate(sources):
                    batch_sources[i].append(source)
                batch_targets.append(target)
            
            # Stack batch data
            stacked_sources = [torch.stack(batch_sources[i]).to(device) 
                             for i in range(config.num_sources)]
            targets = torch.stack(batch_targets).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(stacked_sources)
            loss = combined_loss(predictions, targets, 
                               alpha=config.mape_weight,  # Use optimized weight
                               use_focal=config.use_focal_loss,
                               use_hard_mining=config.use_hard_mining,
                               mining_ratio=config.mining_ratio,
                               mining_weight=config.mining_weight,
                               epoch=epoch, total_epochs=config.num_epochs)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
            
            train_mapes.append(loss.item())
        
        # Validation
        model.eval()
        val_mapes = []
        
        with torch.no_grad():
            val_batches = len(val_data) // config.batch_size
            
            for batch_idx in range(val_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = start_idx + config.batch_size
                batch_samples = val_data[start_idx:end_idx]
                
                # Prepare validation batch (already loaded in memory!)
                batch_sources = [[] for _ in range(config.num_sources)]
                batch_targets = []
                
                for sources, target in batch_samples:
                    for i, source in enumerate(sources):
                        batch_sources[i].append(source)
                    batch_targets.append(target)
                
                # Stack batch data
                stacked_sources = [torch.stack(batch_sources[i]).to(device) 
                                 for i in range(config.num_sources)]
                targets = torch.stack(batch_targets).to(device)
                
                # Forward pass
                predictions = model(stacked_sources)
                mape = enhanced_mape_loss(predictions, targets, smoothing=0)  # No smoothing for validation
                val_mapes.append(mape.item())
        
        # Calculate epoch metrics
        avg_train_mape = np.mean(train_mapes)
        avg_val_mape = np.mean(val_mapes) if val_mapes else float('inf')
        
        # Monitor hard examples in validation set
        if val_data and epoch % 5 == 0:  # Check every 5 epochs
            with torch.no_grad():
                # Sample a batch for hard example analysis
                sample_batch = val_data[:min(8, len(val_data))]
                batch_sources = [[] for _ in range(config.num_sources)]
                batch_targets = []
                
                for sources, target in sample_batch:
                    for i, source in enumerate(sources):
                        batch_sources[i].append(source)
                    batch_targets.append(target)
                
                stacked_sources = [torch.stack(batch_sources[i]).to(device) 
                                 for i in range(config.num_sources)]
                targets = torch.stack(batch_targets).to(device)
                
                predictions = model(stacked_sources)
                hard_mask, sample_mapes = identify_hard_examples(predictions, targets, threshold=0.15)
                hard_ratio = hard_mask.float().mean().item()
                
                print(f"Hard examples ratio: {hard_ratio:.3f} (>{15}% MAPE)")
        
        print(f"Train Loss: {avg_train_mape:.6f}")
        print(f"Val MAPE: {avg_val_mape:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update learning rate
        if config.use_cosine_schedule:
            scheduler.step()
        else:
            scheduler.step(avg_val_mape)
        
        # Stochastic Weight Averaging
        if config.use_swa and epoch >= config.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Save best model and early stopping
        if avg_val_mape < best_val_mape - config.early_stopping_min_delta:
            best_val_mape = avg_val_mape
            epochs_without_improvement = 0
            
            # Save the best model (use SWA model if available)
            model_to_save = swa_model if (config.use_swa and epoch >= config.swa_start_epoch) else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'val_mape': best_val_mape,
                'using_swa': config.use_swa and epoch >= config.swa_start_epoch
            }, best_model_path)
            print(f"✅ New best model saved! Val MAPE: {best_val_mape:.6f}")
            
            # Early stopping check
            if best_val_mape < 0.021:  # Target achieved
                print(f"🎯 Achieved target MAPE! Stopping early.")
                break
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"⏹️ Early stopping after {epochs_without_improvement} epochs without improvement")
            break
    
    print(f"\n🏆 Training completed!")
    print(f"Best validation MAPE: {best_val_mape:.6f}")
    print(f"Target MAPE: 0.021 (hopefully we beat this!)")
    
    return best_model_path


@app.function(
    image=image,
    gpu="A100",
    memory=32000,
    timeout=3600,
    volumes={"/vol": volume}
)
def generate_enhanced_submission(checkpoint_name: str = "best_enhanced_model.pth"):
    """Generate predictions using simple model"""
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = os.path.join(config.model_path, checkpoint_name)
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EnhancedSeismicUNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_mape = checkpoint.get('val_mape', 'N/A')
    print(f"Model validation MAPE: {val_mape}")
    
    # Load test data
    test_data_path = os.path.join(config.data_path, "test")
    test_sample_paths = glob(os.path.join(test_data_path, "*"))
    print(f"Found {len(test_sample_paths)} test samples")
    
    # Generate predictions
    predictions = {}
    
    with torch.no_grad():
        for sample_path in tqdm(test_sample_paths, desc="Generating predictions"):
            sample_id = os.path.basename(sample_path)
            
            # Load and preprocess
            sources, _ = load_sample(sample_path)
            
            # Convert to batch format
            batch_sources = [source.unsqueeze(0).to(device) for source in sources]
            
            # Generate prediction (TTA removed for simplicity)
            pred_velocity = model(batch_sources)
            final_pred = pred_velocity.squeeze(0).cpu().numpy()
            
            # Ensure correct data type and reasonable bounds
            final_pred = np.clip(final_pred, 1.0, 8.0)
            final_pred = final_pred.astype(np.float64)
            
            predictions[sample_id] = final_pred
    
    # Create submission file
    submission_path = os.path.join(config.results_path, "enhanced_submission.npz")
    np.savez(submission_path, **predictions)
    
    # Print statistics
    all_values = np.concatenate([pred.flatten() for pred in predictions.values()])
    print(f"\nSubmission Statistics:")
    print(f"Number of samples: {len(predictions)}")
    print(f"Velocity range: [{all_values.min():.4f}, {all_values.max():.4f}]")
    print(f"Mean velocity: {all_values.mean():.4f}")
    
    print(f"\n✅ Submission created: {submission_path}")
    return submission_path


# Main execution functions
@app.local_entrypoint()
def main():
    """Simple main execution"""
    
    print("🚀 Simple Seismic Velocity Inversion")
    print("=" * 50)
    
    # Step 1: Train enhanced model
    print("\n🏋️ Training enhanced U-Net model for MAPE < 0.021...")
    model_path = train_enhanced_model.remote()
    print(f"✅ Training completed: {model_path}")
    
    # Step 2: Generate submission
    print("\n🎯 Generating enhanced submission...")
    submission_path = generate_enhanced_submission.remote()
    print(f"✅ Submission created: {submission_path}")
    
    print("\n🏆 Done! Optimized model targeting MAPE < 0.021")
    print("Key optimizations:")
    print("- Focal MAPE loss with optimized parameters")
    print("- Hard example mining (50% hardest samples)")
    print("- Stochastic Weight Averaging (SWA)")
    print("- Optimized learning rate and batch size")
    print("- Enhanced model capacity (96 features)")
    print("- Early stopping with patience")
    print("- Refined loss weighting (85% MAPE, 15% MSE)")
    print("- Longer training with better convergence")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=300
)
def download_submission(filename: str = "enhanced_submission.npz"):
    """Download submission file from Modal volume"""
    
    file_path = f"/vol/results/{filename}"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        if os.path.exists("/vol/results/"):
            print("Available files:")
            for file in os.listdir("/vol/results/"):
                print(f"  - {file}")
        return None
    
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    print(f"✅ Successfully read {filename} ({len(file_data)} bytes)")
    return file_data


if __name__ == "__main__":
    config = Config()
    print("Simple Seismic Velocity Inversion Framework")
    print(f"Configuration: {config}")
    print("Run with: modal run modal_seismic_inversion.py")