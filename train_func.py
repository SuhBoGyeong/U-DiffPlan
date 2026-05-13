import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter


def _normalize_features(past, future, norm_values, mode, is_future):
    """Normalize features using min-max scaling."""
    normalized_past = np.zeros_like(past)
    normalized_future = np.zeros_like(future) if is_future else None

    for i in range(past.shape[2]):
        if mode == 'test':
            min_val, range_val = norm_values[i, 0], norm_values[i, 1]
        else:
            if is_future:
                min_val = min(past[:, :, i].min(), future[:, :, i].min())
                range_val = max(past[:, :, i].max(), future[:, :, i].max()) - min_val
            else:
                min_val = past[:, :, i].min()
                range_val = past[:, :, i].max() - min_val
            norm_values[i] = [min_val, range_val]

        if range_val == 0:
            normalized_past[:, :, i] = past[:, :, i] - min_val
            if is_future:
                normalized_future[:, :, i] = future[:, :, i] - min_val
        else:
            normalized_past[:, :, i] = (past[:, :, i] - min_val) / range_val
            if is_future:
                normalized_future[:, :, i] = (future[:, :, i] - min_val) / range_val

    return normalized_past, normalized_future, norm_values


def _normalize_kappa(kappa, norm_values, mode):
    """Normalize kappa values using min-max scaling."""
    if mode == 'test':
        min_val, range_val = norm_values[0, 0], norm_values[0, 1]
    else:
        min_val = kappa.min()
        range_val = kappa.max() - min_val
        norm_values[0] = [min_val, range_val]

    if range_val == 0:
        return kappa - min_val, norm_values
    return (kappa - min_val) / range_val, norm_values


def data_prepare(EV_past, EV_future, EV_kappa, TV_past, TV_future, TV_kappa, config, mode='train'):
    """
    Normalize vehicle trajectory data using min-max scaling.

    Args:
        mode: 'train' to compute and save normalization values, 'test' to load saved values
    """
    is_future = config.is_train
    data_dir = config.data_processed_dir

    # Load or initialize normalization values
    if mode == 'test':
        norm_EV = np.load(f'{data_dir}/data_normalizing_values_EV.npy')
        norm_TV = np.load(f'{data_dir}/data_normalizing_values_TV.npy')
        norm_EV_kappa = np.load(f'{data_dir}/data_normalizing_values_EV_kappa.npy')
        norm_TV_kappa = np.load(f'{data_dir}/data_normalizing_values_TV_kappa.npy')
    else:
        norm_EV = np.zeros((len(config.features_EV), 2))
        norm_TV = np.zeros((len(config.features_TV), 2))
        norm_EV_kappa = np.zeros((1, 2))
        norm_TV_kappa = np.zeros((1, 2))

    # Normalize features
    normalized_EV_past, normalized_EV_future, norm_EV = _normalize_features(
        EV_past, EV_future, norm_EV, mode, is_future)
    normalized_TV_past, normalized_TV_future, norm_TV = _normalize_features(
        TV_past, TV_future, norm_TV, mode, is_future)

    # Normalize kappa
    normalized_EV_kappa, norm_EV_kappa = _normalize_kappa(EV_kappa, norm_EV_kappa, mode)
    normalized_TV_kappa, norm_TV_kappa = _normalize_kappa(TV_kappa, norm_TV_kappa, mode)

    # Save normalization values in train mode
    if mode != 'test':
        np.save(f'{data_dir}/data_normalizing_values_EV.npy', norm_EV)
        np.save(f'{data_dir}/data_normalizing_values_TV.npy', norm_TV)
        np.save(f'{data_dir}/data_normalizing_values_EV_kappa.npy', norm_EV_kappa)
        np.save(f'{data_dir}/data_normalizing_values_TV_kappa.npy', norm_TV_kappa)

    return (normalized_EV_past, normalized_EV_future, normalized_EV_kappa,
            normalized_TV_past, normalized_TV_future, normalized_TV_kappa,
            norm_EV, norm_EV_kappa, norm_TV, norm_TV_kappa)


class CustomDataset(Dataset):
    """
    Custom Dataset for vehicle trajectory data.

    Args:
        data (dict): Dictionary containing vehicle data with structure:
            {
                'EV': {'past', 'future', 'kappa', 'past_org', 'future_org'},
                'TV': {'past', 'future', 'kappa', 'past_org', 'future_org', 'past_init', 'future_init'},
                'Y', 'Y_prob', 'leader', 'road'
            }
    """
    KEYS = ['EV_past', 'EV_future', 'EV_kappa', 'TV_past', 'TV_future', 'TV_kappa',
            'Y', 'Y_prob', 'leader', 'EV_past_org', 'EV_future_org',
            'TV_past_org', 'TV_future_org', 'TV_past_init', 'TV_future_init', 'road']

    def __init__(self, data):
        self.data = {}
        for key in self.KEYS:
            if '_' in key and key.split('_')[0] in ['EV', 'TV']:
                parts = key.split('_')
                vehicle = parts[0]
                field = '_'.join(parts[1:])
                value = data[vehicle][field]
            else:
                value = data[key]

            if isinstance(value, torch.Tensor):
                self.data[key] = value.float()
            else:
                self.data[key] = torch.Tensor(value).float()

    def __len__(self):
        return len(self.data['EV_past'])

    def __getitem__(self, idx):
        return tuple(self.data[key][idx] for key in self.KEYS)

def preprocess_init_relative_TV(EV_past, EV_future, TV_past, TV_future, config):
    """Subtract TV's initial position from all trajectories to make them relative."""
    x_idx, y_idx = config.x_fidx, config.y_fidx

    # Get TV initial position for broadcasting: (N, 1)
    tv_init_x = TV_past[:, -1, x_idx:x_idx+1]
    tv_init_y = TV_past[:, -1, y_idx:y_idx+1]

    def subtract_init(arr):
        result = arr.copy()
        result[:, :, x_idx] -= tv_init_x
        result[:, :, y_idx] -= tv_init_y
        return result

    EV_past_process = subtract_init(EV_past)
    TV_past_process = subtract_init(TV_past)

    if EV_future is not None and TV_future is not None:
        EV_future_process = subtract_init(EV_future)
        TV_future_process = subtract_init(TV_future)
    else:
        EV_future_process, TV_future_process = None, None

    return EV_past_process, EV_future_process, TV_past_process, TV_future_process, TV_past, TV_future

def create_data_dict(EV_past, EV_future, EV_kappa, TV_past, TV_future, TV_kappa,
                     Y, Y_prob, leader, road=None,
                     EV_past_org=None, EV_future_org=None,
                     TV_past_org=None, TV_future_org=None,
                     TV_past_init=None, TV_future_init=None):
    """Create a structured dictionary from vehicle data arrays."""
    if (EV_future is None) or (TV_future is None):
        return {
            'EV': {
                'past': EV_past, 'kappa': EV_kappa,
                'past_org': EV_past.copy(),
            },
            'TV': {
                'past': TV_past, 'kappa': TV_kappa,
                'past_org': TV_past.copy(), 
                'past_init': TV_past.copy(),
            },
            'Y': Y, 'Y_prob': Y_prob, 'leader': leader, 'road': road
        }
    else:
        return {
            'EV': {
                'past': EV_past, 'future': EV_future, 'kappa': EV_kappa,
                'past_org': EV_past.copy(), 'future_org': EV_future.copy()
            },
            'TV': {
                'past': TV_past, 'future': TV_future, 'kappa': TV_kappa,
                'past_org': TV_past.copy(), 'future_org': TV_future.copy(),
                'past_init': TV_past.copy(), 'future_init': TV_future.copy()
            },
            'Y': Y, 'Y_prob': Y_prob, 'leader': leader, 'road': road
        }

def split_data_dict(data, test_size, random_state=1234):
    """Split data dictionary into two parts (train/test or train/val)."""
    keys_order = [
        ('EV', 'past'), ('EV', 'future'), ('EV', 'kappa'),
        ('TV', 'past'), ('TV', 'future'), ('TV', 'kappa'),
        'Y', 'Y_prob', 'leader',
        ('EV', 'past_org'), ('EV', 'future_org'),
        ('TV', 'past_org'), ('TV', 'future_org'),
        ('TV', 'past_init'), ('TV', 'future_init'), 'road'
    ]

    arrays = []
    for key in keys_order:
        if isinstance(key, tuple):
            val = data[key[0]][key[1]]
        else:
            val = data[key]
        if val is not None:
            arrays.append(val)

    split_result = train_test_split(*arrays, test_size=test_size, shuffle=True, random_state=random_state)

    data1, data2 = {'EV': {}, 'TV': {}}, {'EV': {}, 'TV': {}}
    idx = 0
    for key in keys_order:
        if isinstance(key, tuple):
            if data[key[0]][key[1]] is not None:
                data1[key[0]][key[1]] = split_result[idx * 2]
                data2[key[0]][key[1]] = split_result[idx * 2 + 1]
                idx += 1
            else:
                data1[key[0]][key[1]] = None
                data2[key[0]][key[1]] = None
        else:
            if data[key] is not None:
                data1[key] = split_result[idx * 2]
                data2[key] = split_result[idx * 2 + 1]
                idx += 1
            else:
                data1[key] = None
                data2[key] = None

    return data1, data2

def dict_to_tensor(data, device):
    """Convert all numpy arrays in dictionary to tensors on device."""
    result = {'EV': {}, 'TV': {}}
    for vehicle in ['EV', 'TV']:
        for key, val in data[vehicle].items():
            if val is not None:
                result[vehicle][key] = torch.tensor(val).float().to(device)
    for key in ['Y', 'Y_prob', 'leader', 'road']:
        if data[key] is not None:
            result[key] = torch.tensor(data[key]).float().to(device)
    return result

def apply_preprocessing(data, config):
    is_train = config.is_train   
    """Apply relative TV preprocessing and save original data."""
    data['EV']['past_org'] = data['EV']['past'].copy()
    data['TV']['past_org'] = data['TV']['past'].copy()

    if is_train and 'future' in data['EV']: 
        data['EV']['future_org'] = data['EV']['future'].copy()
        data['TV']['future_org'] = data['TV']['future'].copy()

    (data['EV']['past'], data['EV']['future'],
     data['TV']['past'], data['TV']['future'],
     data['TV']['past_init'], data['TV']['future_init']) = preprocess_init_relative_TV(
        data['EV']['past'], data['EV'].get('future', None),
        data['TV']['past'], data['TV'].get('future', None),
        config
    )
    return data

def apply_normalization(data, config, mode='train'):
    """Apply data normalization and return normalizing values."""
    is_future = config.is_train

    if is_future:
        result = data_prepare(
            data['EV']['past'], data['EV']['future'], data['EV']['kappa'],
            data['TV']['past'], data['TV']['future'], data['TV']['kappa'],
            config,
            mode=mode
        )
        data['EV']['past'], data['EV']['future'], data['EV']['kappa'] = result[0], result[1], result[2]
        data['TV']['past'], data['TV']['future'], data['TV']['kappa'] = result[3], result[4], result[5]
        norm_values = {'EV': result[6], 'EV_kappa': result[7], 'TV': result[8], 'TV_kappa': result[9]}
    else:
        result = data_prepare(
            data['EV']['past'], None, data['EV']['kappa'],
            data['TV']['past'], None, data['TV']['kappa'],
            config,
            mode=mode
        )
        data['EV']['past'], _, data['EV']['kappa'] = result[0], result[1], result[2]
        data['TV']['past'], _, data['TV']['kappa'] = result[3], result[4], result[5]
        norm_values = {'EV': result[6], 'EV_kappa': result[7], 'TV': result[8], 'TV_kappa': result[9]}
    return data, norm_values

def create_metrics():
    """Create empty metrics dictionary."""
    return {'ADE': [0, 0, 0], 'FDE': [0, 0, 0]}

def update_metrics(metrics, new_values):
    """Update metrics with new values."""
    for i in range(3):
        metrics['ADE'][i] += new_values['ADE'][i]
        metrics['FDE'][i] += new_values['FDE'][i]

def print_metrics(metrics, prefix=''):
    """Print metrics in a formatted way."""
    for i in range(3):
        print(f"ADE{i+1}{prefix}: {metrics['ADE'][i]:.4f}, FDE{i+1}{prefix}: {metrics['FDE'][i]:.4f}")

def preprocess_batch(batch_data, seq_length=None):
    """
    Preprocess batch data: permute and optionally truncate.

    Args:
        batch_data: tuple from dataloader (EV_past, EV_future, EV_kappa, TV_past, TV_future, TV_kappa,
                    Y, Y_prob, leader, EV_past_org, EV_future_org, TV_past_org, TV_future_org,
                    TV_past_init, TV_future_init, road)
        seq_length: if provided, truncate future sequences to this length

    Returns:
        dict: preprocessed data with keys matching input names
    """
    (EV_past, EV_future, EV_kappa, TV_past, TV_future, TV_kappa,
     Y, Y_prob, leader, EV_past_org, EV_future_org, TV_past_org, TV_future_org,
     TV_past_init, TV_future_init, road) = batch_data

    # Permute: (batch, seq, feature) -> (batch, feature, seq)
    data = {
        'EV_past': EV_past.permute(0, 2, 1),
        'EV_future': EV_future.permute(0, 2, 1),
        'EV_kappa': EV_kappa,
        'TV_past': TV_past.permute(0, 2, 1),
        'TV_future': TV_future.permute(0, 2, 1),
        'TV_kappa': TV_kappa,
        'Y': Y,
        'Y_prob': Y_prob,
        'leader': leader,
        'EV_past_org': EV_past_org.permute(0, 2, 1),
        'EV_future_org': EV_future_org.permute(0, 2, 1),
        'TV_past_org': TV_past_org.permute(0, 2, 1),
        'TV_future_org': TV_future_org.permute(0, 2, 1),
        'TV_past_init': TV_past_init.permute(0, 2, 1),
        'TV_future_init': TV_future_init.permute(0, 2, 1),
        'road': road,
    }

    # Truncate future sequences if seq_length is provided
    if seq_length is not None:
        data['EV_future'] = data['EV_future'][:, :, :seq_length]
        data['TV_future'] = data['TV_future'][:, :, :seq_length]
        data['EV_future_org'] = data['EV_future_org'][:, :, :seq_length]
        data['TV_future_org'] = data['TV_future_org'][:, :, :seq_length]
        data['TV_future_init'] = data['TV_future_init'][:, :, :seq_length]

    return data

def denormalize_traj(tensor, idx, norm_values, x_fidx, y_fidx, mode=None):
    """Denormalize trajectory coordinates from normalized values."""
    def denorm(val, fidx):
        return val.cpu().detach().numpy() * norm_values[fidx, 1] + norm_values[fidx, 0]

    if mode == 'samples':
        return denorm(tensor[idx, :, x_fidx, :], x_fidx), denorm(tensor[idx, :, y_fidx, :], y_fidx)
    return denorm(tensor[idx, x_fidx, :], x_fidx), denorm(tensor[idx, y_fidx, :], y_fidx)

def add_init_offset(coords, TV_past_init, idx, x_fidx, y_fidx):
    """Add initial point offset to (x, y) coordinates."""
    init_x = TV_past_init[idx, x_fidx, -1].cpu().detach().numpy()
    init_y = TV_past_init[idx, y_fidx, -1].cpu().detach().numpy()
    return coords[0] + init_x, coords[1] + init_y

def process_traj_for_plot(EV_past, EV_future, TV_past, TV_future, pred_traj, pred_traj_samples,
                          idx, norm, TV_past_init, real_Y, pred_Y, real_leader, config):
    """
    Process all trajectories for plotting: denormalize and add init offset.

    Returns:
        dict: {
            'EV_past': (x, y), 'EV_future': (x, y),
            'TV_past': (x, y), 'TV_future': (x, y),
            'pred': (x, y)
        }
    """
    X_FIDX, Y_FIDX = config.x_fidx, config.y_fidx
    FILTER_WINDOW = config.filter_window
    FILTER_ORDER = config.filter_order
    LABEL_NAME = config.label_name
    max_idx = np.argmax(pred_Y[idx].cpu().detach().numpy())

    real_Y = LABEL_NAME[int(real_Y[idx, 0].cpu().detach().numpy())]
    pred_Y = LABEL_NAME[max_idx] if int(real_leader[idx].cpu().detach().numpy()) == 1 else LABEL_NAME[max_idx+2]


    # Denormalize all trajectories
    if config.is_train:
        traj = {
            'EV_past': denormalize_traj(EV_past, idx, norm, X_FIDX, Y_FIDX),
            'EV_future': denormalize_traj(EV_future, idx, norm, X_FIDX, Y_FIDX),
            'TV_past': denormalize_traj(TV_past, idx, norm, X_FIDX, Y_FIDX),
            'TV_future': denormalize_traj(TV_future, idx, norm, X_FIDX, Y_FIDX),
            'pred': denormalize_traj(pred_traj, idx, norm, X_FIDX, Y_FIDX),
            'pred_samples': denormalize_traj(pred_traj_samples, idx, norm, X_FIDX, Y_FIDX, 'samples'),
            'real_Y': real_Y,
            'pred_Y': pred_Y,
        }

        # Add initial point offset to trajectory coordinates only
        traj_keys = ['EV_past', 'EV_future', 'TV_past', 'TV_future', 'pred', 'pred_samples']
    else:
        traj = {
            'EV_past': denormalize_traj(EV_past, idx, norm, X_FIDX, Y_FIDX),
            'TV_past': denormalize_traj(TV_past, idx, norm, X_FIDX, Y_FIDX),
            'pred': denormalize_traj(pred_traj, idx, norm, X_FIDX, Y_FIDX),
            'pred_samples': denormalize_traj(pred_traj_samples, idx, norm, X_FIDX, Y_FIDX, 'samples'),
            'real_Y': real_Y,
            'pred_Y': pred_Y,
        }

        # Add initial point offset to trajectory coordinates only
        traj_keys = ['EV_past', 'TV_past', 'pred', 'pred_samples']

    for key in traj_keys:
        traj[key] = add_init_offset(traj[key], TV_past_init, idx, X_FIDX, Y_FIDX)


    # Apply savgol filter to prediction
    traj['pred'] = (
        savgol_filter(traj['pred'][0], FILTER_WINDOW, FILTER_ORDER),
        savgol_filter(traj['pred'][1], FILTER_WINDOW, FILTER_ORDER)
    )
    return traj

def run_batch(model, batch_data, norm_value, loss_func, config, diff_mode, seq_length=None):
    """
    Process a single batch for both training and validation.

    Returns:
        dict: Contains loss values, predictions, and batch info
    """
    b = preprocess_batch(batch_data, seq_length=seq_length)
    batch_size = b['EV_past'].shape[0]

    condition = b['Y'][:, 1].unsqueeze(1).repeat(1, config.sequence_length).unsqueeze(1)

    loss, pred_traj_samples, pred_Y, pred_traj_mean, pred_traj_log_var, mu_w_center, log_var_w_center = model(
        b['EV_past'], b['EV_future'], b['EV_kappa'],
        b['TV_past'], b['TV_future'], b['TV_kappa'], condition,
        b['EV_past_org'], b['TV_past_org'], norm_value,
        mode='generate', diff_mode=diff_mode
    )

    loss_traj, loss_label = loss_func(
        b['TV_future'], pred_traj_mean, b['Y_prob'][:, 0, :], pred_Y,
        config.x_fidx, config.y_fidx, config, b['leader']
    )
    total_loss = loss + loss_traj + loss_label

    return {
        'batch': b,
        'batch_size': batch_size,
        'loss': total_loss,
        'loss_traj': loss_traj,
        'loss_label': loss_label,
        'pred_traj_mean': pred_traj_mean,
        'pred_traj_samples': pred_traj_samples,
        'pred_Y': pred_Y,
    }


def evaluation(real_TV_future, pred_traj, normalizing_values, TV_past_init, batch_size, dataloader_len, config):
    """
    Evaluate trajectory prediction with ADE/FDE metrics.

    Horizons are defined in config.eval_horizons (default: [20, 40, 60] at 20Hz = 1s, 2s, 3s)

    Returns:
        dict: {'ADE': [ADE per horizon], 'FDE': [FDE per horizon]}
    """
    x_idx, y_idx = config.x_fidx, config.y_fidx
    norm = normalizing_values

    def denorm_and_offset(tensor, fidx):
        """Denormalize tensor and add initial position offset."""
        val = tensor[:, fidx, :].cpu().detach().numpy() * norm[fidx, 1] + norm[fidx, 0]
        val += TV_past_init[:, fidx, -1:].cpu().detach().numpy()
        return val

    # Denormalize and add offset
    real_x, real_y = denorm_and_offset(real_TV_future, x_idx), denorm_and_offset(real_TV_future, y_idx)
    pred_x, pred_y = denorm_and_offset(pred_traj, x_idx), denorm_and_offset(pred_traj, y_idx)

    # Apply smoothing filter to predictions
    pred_x = savgol_filter(pred_x, config.filter_window, config.filter_order, axis=1)
    pred_y = savgol_filter(pred_y, config.filter_window, config.filter_order, axis=1)

    # Calculate metrics for different horizons
    ade_list, fde_list = [], []
    scale = batch_size * dataloader_len

    for h in config.eval_horizons:
        # ADE: Average Displacement Error
        displacement = np.sqrt((real_x[:, :h] - pred_x[:, :h])**2 + (real_y[:, :h] - pred_y[:, :h])**2)
        ade_list.append(np.sum(displacement.mean(axis=1)) / scale)

        # FDE: Final Displacement Error
        fde_idx = h - 1
        fde = np.sqrt((real_x[:, fde_idx] - pred_x[:, fde_idx])**2 + (real_y[:, fde_idx] - pred_y[:, fde_idx])**2)
        fde_list.append(np.sum(fde) / scale)

    return {'ADE': ade_list, 'FDE': fde_list}
