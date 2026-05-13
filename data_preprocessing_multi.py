"""
Data Preprocessing Script for Multi-Vehicle Racing Data.

This script processes raw simulation data from data_generation.py and creates
training datasets with past/future trajectory windows.

Data format:
    - TV strategy: driving (0) / blocking (1)
    - EV strategy: following (0) / overtaking (1)
    - Label: [TV_strategy, EV_strategy]

Usage:
    python data_preprocessing_multi.py --data_dirs dir1 dir2 --save_dir output
"""

import numpy as np
import os
import argparse
from config_nn import Config_NN

np.random.seed(555)

def load_data_files(data_dir, file_name):
    """
    Load all data files for a given sample.

    Args:
        data_dir: Base directory containing data
        file_name: Name of the data file

    Returns:
        Dictionary containing all loaded data arrays
    """
    return {
        'data': np.load(f'{data_dir}/states/{file_name}'),
        'leader': np.load(f'{data_dir}/leader/{file_name}'),
        'road': np.load(f'{data_dir}/road/{file_name}'),
        'EV_decision': np.load(f'{data_dir}/EV_decision/{file_name}'),
        'TV_decision': np.load(f'{data_dir}/TV_decision/{file_name}'),
        'EV_dec_prob': np.load(f'{data_dir}/EV_decision_prob/{file_name}'),
        'TV_dec_prob': np.load(f'{data_dir}/TV_decision_prob/{file_name}'),
    }


def extract_vehicle_data(data, swap_ev_tv=False):
    """
    Extract EV and TV data from the combined data array.

    Args:
        data: Combined data array of shape (T, 122)
        swap_ev_tv: If True, swap EV and TV data columns

    Returns:
        EV_data, TV_data, EV_kappa, TV_kappa
    """
    if swap_ev_tv:
        TV_data = data[:, :10]
        EV_data = data[:, 10:20]
        TV_kappa = data[:, 20:71]
        EV_kappa = data[:, 71:]
    else:
        EV_data = data[:, :10]
        TV_data = data[:, 10:20]
        EV_kappa = data[:, 20:71]
        TV_kappa = data[:, 71:]

    return EV_data, TV_data, EV_kappa, TV_kappa


def process_sample(loaded_data, ii, sequence_length, swap_ev_tv=False):
    """
    Process a single sample at timestep ii.

    Args:
        loaded_data: Dictionary of loaded data arrays
        ii: Current timestep index
        sequence_length: Number of past timesteps to include
        future_samples: Number of future timesteps to include
        swap_ev_tv: If True, swap EV and TV roles

    Returns:
        Dictionary containing processed sample data
    """
    data = loaded_data['data']
    leader = loaded_data['leader']
    road = loaded_data['road']

    if swap_ev_tv:
        EV_decision = loaded_data['TV_decision']
        TV_decision = loaded_data['EV_decision']
        EV_dec_prob = loaded_data['TV_dec_prob']
        TV_dec_prob = loaded_data['EV_dec_prob']
    else:
        EV_decision = loaded_data['EV_decision']
        TV_decision = loaded_data['TV_decision']
        EV_dec_prob = loaded_data['EV_dec_prob']
        TV_dec_prob = loaded_data['TV_dec_prob']

    EV_data, TV_data, EV_kappa, TV_kappa = extract_vehicle_data(data, swap_ev_tv)

    # Extract time windows
    TV_past_data = TV_data[ii - sequence_length:ii, :]
    TV_future_data = TV_data[ii:ii + sequence_length, :]
    EV_past_data = EV_data[ii - sequence_length:ii, :]
    EV_future_data = EV_data[ii:ii + sequence_length, :]

    TV_curr_kappa = TV_kappa[ii, :]
    EV_curr_kappa = EV_kappa[ii, :]

    # Handle leader and road data for swapped case
    if swap_ev_tv:
        leader_curr = 1 - leader[ii]
        curr_road = road[ii - sequence_length:ii + sequence_length, [1, 0]]
    else:
        leader_curr = leader[ii]
        curr_road = road[ii - sequence_length:ii + sequence_length, :]

    curr_label = np.array([TV_decision[ii, 0], EV_decision[ii, 0]], dtype=int)
    curr_label_prob = np.array([TV_dec_prob[ii, :], EV_dec_prob[ii, :]])
    
    return {
        'TV_past': TV_past_data,
        'TV_future': TV_future_data,
        'EV_past': EV_past_data,
        'EV_future': EV_future_data,
        'TV_kappa': TV_curr_kappa,
        'EV_kappa': EV_curr_kappa,
        'leader': leader_curr,
        'road': curr_road,
        'Y': curr_label,
        'Y_prob': curr_label_prob,
    }


def accumulate_data(accumulated, sample, is_first):
    """
    Add processed sample to accumulated data arrays.

    Args:
        accumulated: Dictionary of accumulated arrays (or None if first)
        sample: Processed sample dictionary
        is_first: Whether this is the first sample

    Returns:
        Updated accumulated dictionary
    """
    if is_first:
        return {
            'TV_past': np.expand_dims(sample['TV_past'], axis=0),
            'TV_future': np.expand_dims(sample['TV_future'], axis=0),
            'EV_past': np.expand_dims(sample['EV_past'], axis=0),
            'EV_future': np.expand_dims(sample['EV_future'], axis=0),
            'TV_kappa': np.expand_dims(sample['TV_kappa'], axis=0),
            'EV_kappa': np.expand_dims(sample['EV_kappa'], axis=0),
            'Y': np.expand_dims(sample['Y'], axis=0),
            'Y_prob': np.expand_dims(sample['Y_prob'], axis=0),
            'leader': np.expand_dims(sample['leader'], axis=0),
            'road': np.expand_dims(sample['road'], axis=0),
        }
    else:
        for key in accumulated:
            accumulated[key] = np.concatenate(
                (accumulated[key], np.expand_dims(sample[key], axis=0)), axis=0)
        return accumulated


def process_data_directory(data_dirs, data_dir, sequence_length, interval,
                           max_count, data_count, accumulated, swap_ev_tv=False):
    """
    Process all files in a data directory.

    Args:
        data_dir: Directory containing raw data
        sequence_length: Number of past timesteps
        future_samples: Number of future timesteps
        interval: Sampling interval
        max_count: Maximum number of samples to collect
        data_count: Current sample count
        accumulated: Accumulated data dictionary
        swap_ev_tv: If True, swap EV and TV roles

    Returns:
        Updated data_count and accumulated dictionary
    """
    for file_name in sorted(os.listdir(data_dirs+'/'+data_dir)):
        print(file_name)

        loaded_data = load_data_files(data_dirs, file_name)
        data = loaded_data['data']

        if data_count >= max_count:
            break

        for ii in range(sequence_length, data.shape[0] - sequence_length, interval):
            data_count += 1

            sample = process_sample(
                loaded_data, ii, sequence_length, swap_ev_tv)

            is_first = (accumulated is None)
            accumulated = accumulate_data(accumulated, sample, is_first)

            print(f'Data count: {data_count}')

            if data_count >= max_count:
                break

    return data_count, accumulated


def save_processed_data(accumulated, save_dir):
    """
    Save all accumulated data to files.

    Args:
        accumulated: Dictionary containing all accumulated arrays
        save_dir: Directory to save processed data
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(f'{save_dir}/TV_past.npy', accumulated['TV_past'])
    np.save(f'{save_dir}/TV_future.npy', accumulated['TV_future'])
    np.save(f'{save_dir}/EV_past.npy', accumulated['EV_past'])
    np.save(f'{save_dir}/EV_future.npy', accumulated['EV_future'])
    np.save(f'{save_dir}/TV_kappa.npy', accumulated['TV_kappa'])
    np.save(f'{save_dir}/EV_kappa.npy', accumulated['EV_kappa'])
    np.save(f'{save_dir}/Y.npy', accumulated['Y'])
    np.save(f'{save_dir}/Y_prob.npy', accumulated['Y_prob'])
    np.save(f'{save_dir}/leader.npy', accumulated['leader'])
    np.save(f'{save_dir}/road.npy', accumulated['road'])


def main():
    parser = argparse.ArgumentParser(description='Preprocess racing simulation data')
    parser.add_argument('--max_samples_phase1', type=int, default=5000,
                        help='Max samples for first phase (normal order)')
    parser.add_argument('--max_samples_phase2', type=int, default=10000,
                        help='Max samples for second phase (swapped EV/TV)')

    args = parser.parse_args()

    config = Config_NN()
    SEQUENCE_LENGTH = config.sequence_length
    INTERVAL = config.interval
    data_dirs = config.data_dir
    data_processed_dir = config.data_processed_dir
    data_count = 0
    MAX_SAMPLES_PHASE1 = config.max_samples_phase1
    MAX_SAMPLES_PHASE2 = config.max_samples_phase2
    accumulated = None

    # Phase 1: Process with normal EV/TV order (TV as leader)
    print("Phase 1: Processing with normal EV/TV order...")
    for data_dir in os.listdir(data_dirs):
        data_count, accumulated = process_data_directory(
            data_dirs, data_dir, SEQUENCE_LENGTH, INTERVAL,
            MAX_SAMPLES_PHASE1, data_count, accumulated, swap_ev_tv=False)

    # Phase 2: Process with swapped EV/TV order (EV as leader)
    print("Phase 2: Processing with swapped EV/TV order...")
    for data_dir in os.listdir(data_dirs):
        data_count, accumulated = process_data_directory(
            data_dirs, data_dir, SEQUENCE_LENGTH, INTERVAL,
            MAX_SAMPLES_PHASE2, data_count, accumulated, swap_ev_tv=True)

    # Print final shapes
    print(f"\nFinal data shapes:")
    print(f"  TV_past: {accumulated['TV_past'].shape}")
    print(f"  TV_future: {accumulated['TV_future'].shape}")
    print(f"  EV_past: {accumulated['EV_past'].shape}")
    print(f"  EV_future: {accumulated['EV_future'].shape}")
    print(f"  TV_kappa: {accumulated['TV_kappa'].shape}")
    print(f"  EV_kappa: {accumulated['EV_kappa'].shape}")
    print(f"  leader: {accumulated['leader'].shape}")
    print(f"  Y: {accumulated['Y'].shape}")

    # Save processed data
    save_processed_data(accumulated, data_processed_dir)
    print(f"\nData saved to {data_processed_dir}")


if __name__ == "__main__":
    main()
