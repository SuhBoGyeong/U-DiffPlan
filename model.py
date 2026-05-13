"""
U-DiffPlan: Conditional Diffusion Model for Vehicle Trajectory Prediction.

This module implements a diffusion-based trajectory prediction model that combines:
- A conditional denoising diffusion model for trajectory generation
- An auxiliary encoder for strategy/behavior classification
- Track center line conditioning for improved predictions
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import math
from tqdm import tqdm

from plot_func import getTrack

# Set random seeds for reproducibility
np.random.seed(555)
torch.manual_seed(555)


def loss_func_UDiffPlan(real_data_future, fake_data_future, label, pred_label, x_fidx, y_fidx, config, leader):
    """
    Compute combined loss for trajectory and label prediction.

    Args:
        real_data_future: Ground truth future trajectory (batch, features, seq_len)
        fake_data_future: Predicted future trajectory (batch, features, seq_len)
        label: Ground truth strategy labels (batch, n_classes)
        pred_label: Predicted strategy labels (batch, n_classes)
        x_fidx: Feature index for x coordinate
        y_fidx: Feature index for y coordinate
        config: Configuration object
        leader: Leader vehicle indicator (unused, kept for interface compatibility)

    Returns:
        tuple: (trajectory_loss, label_loss)
    """
    traj_loss = (F.mse_loss(real_data_future[:, x_fidx, :], fake_data_future[:, x_fidx, :], reduction='sum') +
                 F.mse_loss(real_data_future[:, y_fidx, :], fake_data_future[:, y_fidx, :], reduction='sum'))
    label_loss = F.mse_loss(pred_label[:, 0], label[:, 0], reduction='sum')
    return traj_loss, label_loss


class UDiffPlan(nn.Module):
    """
    Conditional Diffusion Model for Vehicle Trajectory Prediction.

    Combines a denoising diffusion model with an auxiliary encoder to predict
    future vehicle trajectories conditioned on past observations, road geometry,
    and driving strategy.

    Architecture:
        - Kappa Transform: Projects road curvature to sequence length
        - Condition Encoder: Extracts latent features for strategy classification
        - Denoising UNet: Iteratively denoises trajectory predictions
    """

    def __init__(self, config, device, unet_params, beta_schedule, module_name, args):
        """
        Initialize UDiffPlan model.

        Args:
            config: Configuration object with model hyperparameters
            device: Torch device (cuda/cpu)
            unet_params: Dictionary of UNet architecture parameters
            beta_schedule: Noise schedule configuration for diffusion
            module_name: Name identifier for the module
            args: Additional arguments (step, gpu, resume flags)
        """
        super().__init__()

        self.device = device
        self.args = args
        self.config = config

        # Core dimensions from config
        self.seq_len = config.sequence_length
        self.n_labels = config.labels
        self.N_samples = config.N_samples

        # Feature indices
        self.x_idx = config.x_fidx
        self.y_idx = config.y_fidx

        # Import UNet here to avoid circular imports
        from train_generation_fun_modules_sr3 import UNet

        self.denoise_fn = UNet(**unet_params)
        self.beta_schedule = beta_schedule

        # Kappa (road curvature) transformation layer
        self.kappa_transform = nn.Sequential(
            nn.Linear(config.kappa_len, self.seq_len)
        )

        # Build condition encoder
        self.encoder = self._build_encoder(config)

        # Calculate latent size for prediction head
        latent_size = self._compute_latent_size(config)

        # Strategy prediction head
        self.pred_model = nn.Sequential(
            nn.Linear(latent_size, self.n_labels),
            nn.Softmax(dim=1)
        )

    def _build_encoder(self, config):
        """Build the auxiliary condition encoder network."""
        seq = self.seq_len
        chns = config.chns
        k, s, p = config.k_E, config.s_E, config.p_E
        dropout = config.dropout_E

        return nn.Sequential(
            nn.Conv1d(config.input_dim, seq * chns[0], k[0], s[0], p[0]),
            nn.BatchNorm1d(seq * chns[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(seq * chns[0], seq * chns[1], k[1], s[1], p[1]),
            nn.BatchNorm1d(seq * chns[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(seq * chns[1], seq * chns[2], k[2], s[2], p[2]),
            nn.BatchNorm1d(seq * chns[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(seq * chns[2], seq, k[3], s[3], p[3]),
            nn.BatchNorm1d(seq),
        )

    def _compute_latent_size(self, config):
        """Compute the flattened latent size after encoder and chunking."""
        with torch.no_grad():
            dummy = torch.zeros(1, config.input_dim, self.seq_len)
            out = self.encoder(dummy)
            if out.size(-1) % 2 == 1:
                out = F.pad(out, (0, 1), mode='replicate')
            return self.seq_len * (out.size(-1) // 2)

    def set_loss(self, loss_fn):
        """Set the loss function for diffusion training."""
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """
        Initialize the noise schedule for diffusion process.

        Computes and registers buffers for:
        - gammas: cumulative product of alphas (signal retention)
        - posterior coefficients for denoising

        Args:
            device: Target device for tensors
            phase: 'train' or 'test' to select schedule parameters
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas

        self.num_timesteps = int(betas.shape[0])
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # Forward diffusion coefficients
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # Posterior distribution coefficients (for reverse process)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        """Predict clean sample y_0 from noisy sample y_t and predicted noise."""
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise)

    def q_posterior(self, y_0_hat, y_t, t):
        """Compute posterior distribution q(y_{t-1} | y_t, y_0)."""
        mean = (extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t)
        log_var = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return mean, log_var

    def p_mean_variance(self, y_t, t, clip_denoised=True, y_cond=None):
        """Compute mean and variance for reverse diffusion step p(y_{t-1} | y_t)."""
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        predicted_noise = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
        y_0_hat = self.predict_start_from_noise(y_t, t, predicted_noise)

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        return self.q_posterior(y_0_hat, y_t, t)

    def q_sample(self, y_0, sample_gammas, noise=None):
        """Sample from forward diffusion process q(y_t | y_0)."""
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        """Single reverse diffusion step: sample y_{t-1} from p(y_{t-1} | y_t)."""
        mean, log_var = self.p_mean_variance(y_t, t, clip_denoised, y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        sample = mean + noise * (0.5 * log_var).exp()
        return mean, log_var, noise, sample

    @torch.no_grad()
    def restoration(self, y_cond, mask=None, sample_num=8):
        """
        Run full reverse diffusion process to generate trajectory samples.

        Args:
            y_cond: Conditioning information (batch, channels, 1, seq_len)
            mask: Optional mask for inpainting
            sample_num: Number of intermediate samples to store

        Returns:
            tuple: (final_mean, final_log_var, final_noise, final_sample, intermediate_samples)
        """
        batch_size = y_cond.shape[0]
        sample_interval = self.num_timesteps // sample_num

        # Initialize with random noise
        y_t = torch.randn(batch_size, 2, 1, y_cond.shape[-1],
                          device=y_cond.device, dtype=y_cond.dtype)
        intermediates = y_t

        # Reverse diffusion loop
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=y_cond.device, dtype=torch.long)
            mean, log_var, noise, y_t = self.p_sample(y_t, t, y_cond=y_cond)

            if i % sample_interval == 0:
                intermediates = torch.cat([intermediates, y_t], dim=0)

        return mean, log_var, noise, y_t, intermediates

    def _denorm(self, val, idx, norm, offset=None):
        """Denormalize value: val * range + min + offset."""
        result = val * norm[idx, 1] + norm[idx, 0]
        return result + offset if offset is not None else result

    def _norm(self, val, idx, norm, offset=None):
        """Normalize value: (val - offset - min) / range."""
        if offset is not None:
            val = val - offset
        return (val - norm[idx, 0]) / norm[idx, 1]

    def _get_center_line(self, TV_past, norm, TV_init_org):
        """Get normalized center line coordinates from TV position."""
        x, y = self.x_idx, self.y_idx
        offset_x, offset_y = TV_init_org[:, x, -1], TV_init_org[:, y, -1]

        # Denormalize initial position
        init_point = torch.stack([
            self._denorm(TV_past[:, x, -1], x, norm, offset_x),
            self._denorm(TV_past[:, y, -1], y, norm, offset_y)
        ], dim=1)

        # Get center line from track
        center = getCenter(init_point, None, self.device, self.config.map_file, x, y, self.seq_len)

        # Normalize back
        return torch.stack([
            self._norm(center[:, 0], x, norm, offset_x.unsqueeze(-1)),
            self._norm(center[:, 1], y, norm, offset_y.unsqueeze(-1))
        ], dim=1)

    def _encode_condition(self, EV_past, TV_past, EV_kappa, condition, center_line):
        """
        Build and encode the full conditioning tensor.

        Combines: EV trajectory, TV trajectory, road curvature, strategy, center line
        """
        # Concatenate vehicle trajectories
        y_cond = torch.cat([EV_past.unsqueeze(1), TV_past.unsqueeze(1)], dim=2)

        # Transform and add kappa (road curvature)
        z_kappa_EV = self.kappa_transform(EV_kappa)
        z_kappa_TV = self.kappa_transform(EV_kappa)  # Note: uses EV_kappa for both
        z_kappa = torch.stack([z_kappa_EV, z_kappa_TV], dim=1)

        # Build full condition
        y_cond = torch.cat([y_cond, z_kappa.unsqueeze(1)], dim=2)
        y_cond = torch.cat([y_cond, condition.unsqueeze(1)], dim=2)
        y_cond = torch.cat([y_cond, center_line.unsqueeze(1)], dim=2)

        return y_cond

    def _encode_latent(self, y_cond):
        """Encode condition to latent space with VAE-style reparameterization."""
        encoded = self.encoder(y_cond.squeeze(1))

        # Ensure even size for chunking
        if encoded.size(-1) % 2 == 1:
            encoded = F.pad(encoded, (0, 1), mode='replicate')

        mu, log_var = encoded.chunk(2, dim=-1)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std

        return mu, log_var, z.view(z.size(0), -1)

    def forward(self, EV_past, EV_future, EV_kappa, TV_past, TV_future, TV_kappa,
                condition, EV_init_org, TV_init_org, norm,
                mode='generate', diff_mode='pretrain', mask=None, noise=None, **kwargs):
        """
        Forward pass for trajectory prediction.

        Args:
            EV_past: Ego vehicle past trajectory (batch, features, seq_len)
            EV_future: Ego vehicle future trajectory (not used in generation)
            EV_kappa: Road curvature for EV
            TV_past: Target vehicle past trajectory
            TV_future: Target vehicle future trajectory (ground truth for training)
            TV_kappa: Road curvature for TV (not used)
            condition: Strategy condition (batch, 1, seq_len)
            EV_init_org: EV original initial position
            TV_init_org: TV original initial position
            norm: Normalization values
            mode: Operation mode (default: 'generate')
            diff_mode: 'pretrain' for training, 'test' for inference
            mask: Optional inpainting mask
            noise: Optional fixed noise for reproducibility

        Returns:
            tuple: (loss, samples, pred_label, mean, std, latent_mu, latent_log_var)
        """
        # Get center line conditioning
        center_line = self._get_center_line(TV_past, norm, TV_init_org)

        # Build and encode conditioning
        y_cond = self._encode_condition(EV_past, TV_past, EV_kappa, condition, center_line)

        # Encode to latent and predict strategy
        mu, log_var, latent = self._encode_latent(y_cond)
        pred_label = self.pred_model(latent)

        # Prepare condition for diffusion (permute for UNet)
        y_cond = y_cond.permute(0, 2, 1, 3)

        if diff_mode == 'pretrain':
            # Training: compute diffusion loss
            y_0 = TV_future[:, :2, :].unsqueeze(1)
            batch_size = y_0.shape[0]

            # Sample random timesteps and interpolate gammas
            t = torch.randint(1, self.num_timesteps, (batch_size,), device=y_0.device).long()
            gamma_prev = extract(self.gammas, t - 1, x_shape=(1, 1))
            gamma_curr = extract(self.gammas, t, x_shape=(1, 1))
            sample_gammas = gamma_prev + (gamma_curr - gamma_prev) * torch.rand((batch_size, 1), device=y_0.device)

            # Forward diffusion
            noise = default(noise, lambda: torch.randn_like(y_0))
            y_noisy = self.q_sample(y_0, sample_gammas.view(-1, 1, 1, 1), noise)

            # Predict noise
            y_noisy = y_noisy.permute(0, 2, 1, 3)
            noise_pred = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas.view(batch_size, -1))
            noise_pred = noise_pred.permute(0, 2, 1, 3)

            loss = self.loss_fn(noise, noise_pred)
        else:
            loss = 0

        # Generate samples via reverse diffusion
        y_cond_repeated = y_cond.repeat_interleave(self.N_samples, dim=0)
        _, _, _, y_t, _ = self.restoration(y_cond_repeated, mask=mask)

        # Reshape and compute statistics
        samples = y_t.view(EV_past.shape[0], self.N_samples, 2, self.seq_len)
        mean = samples.mean(dim=1)
        std = samples.std(dim=1)

        return loss, samples, pred_label, mean, std, mu, log_var


# =============================================================================
# Utility Functions
# =============================================================================

def exists(x):
    """Check if value exists (is not None)."""
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d (or call d if it's a function)."""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """
    Extract values from tensor 'a' at timestep indices 't' and broadcast to x_shape.

    Args:
        a: Lookup table tensor of shape (T,) or (..., T)
        t: Timestep indices of shape (B,)
        x_shape: Target shape for broadcasting (B, ...)

    Returns:
        Extracted values broadcast to shape (B, *x_shape[1:])
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1))).expand(batch_size, *x_shape[1:])


# =============================================================================
# Beta Schedule Functions
# =============================================================================

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    """Create beta schedule with linear warmup phase."""
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_steps = int(n_timestep * warmup_frac)
    betas[:warmup_steps] = np.linspace(linear_start, linear_end, warmup_steps, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    """
    Create noise schedule (beta values) for diffusion process.

    Args:
        schedule: Schedule type ('quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd', 'cosine')
        n_timestep: Number of diffusion timesteps
        linear_start: Starting beta value for linear schedules
        linear_end: Ending beta value for linear schedules
        cosine_s: Small offset for cosine schedule to prevent singularity

    Returns:
        Beta values array of shape (n_timestep,)
    """
    if schedule == 'quad':
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        # Jensen-Shannon divergence schedule: 1/T, 1/(T-1), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == 'cosine':
        # Cosine schedule from "Improved DDPM" paper
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        betas = (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")
    return betas


# =============================================================================
# Track Utility Functions
# =============================================================================

def getCenter(initial_point, end_point, device, map_file, x_idx, y_idx, seq_len=60):
    """
    Get track center line starting from closest point to initial_point.

    Args:
        initial_point: Starting position (batch, 2)
        end_point: Unused, kept for interface compatibility
        device: Torch device
        map_file: Path to track file
        x_idx, y_idx: Coordinate indices
        seq_len: Number of points to return

    Returns:
        Center line (batch, 2, seq_len)
    """
    _, Xref, Yref, *_ = getTrack(map_file)
    Xref = torch.tensor(Xref, dtype=torch.float32, device=device)
    Yref = torch.tensor(Yref, dtype=torch.float32, device=device)

    # Find closest point and extract sequence
    dist = (Xref - initial_point[:, x_idx:x_idx+1])**2 + (Yref - initial_point[:, y_idx:y_idx+1])**2
    start_idx = dist.argmin(dim=1)
    offsets = torch.arange(seq_len, device=device)
    indices = (start_idx.unsqueeze(1) + offsets) % Xref.size(0)

    return torch.stack([Xref[indices], Yref[indices]], dim=1)
