"""
Data Generation Script for Multi-Vehicle Racing Simulation.

This script generates training data by simulating racing scenarios between
an Ego Vehicle (EV) and a Target Vehicle (TV) using MPC controllers.

Usage:
    python data_generation.py --num_samples 400 --save_dir data --mode sample_
"""

import shutil
from multiprocessing import Process, Queue, current_process
import numpy as np
from env import Racing, Config, Config_TV
from env.track import Track
import random
import argparse
import os
from config_nn import Config_NN

def get_initial_state(config, thetaecephiV):
    """
    Convert curvilinear coordinates to Cartesian coordinates.

    Args:
        config: Configuration object containing track information.
        thetaecephiV: [theta, ec, ephi, velocity]
            - theta: Progress along track centerline
            - ec: Lateral deviation from centerline
            - ephi: Heading error
            - velocity: Longitudinal velocity

    Returns:
        x0_curv: Initial state in curvilinear coordinates.
        x0: Initial state in Cartesian coordinates.
    """
    track = Track(config.env.track_filename)
    x0_curv = np.array([*thetaecephiV, 0.0, 0.0, 0.0, 0.0])
    xyref = track(thetaecephiV[0])
    dp = track(thetaecephiV[0], 1)
    psiref = np.arctan2(dp[1], dp[0])
    eX = -np.sin(psiref) * thetaecephiV[1]
    eY = np.cos(psiref) * thetaecephiV[1]
    x = xyref[0] + eX
    y = xyref[1] + eY
    x0 = np.array([[x, y, psiref + thetaecephiV[2]],
                   [thetaecephiV[3], 0.0, 0.0, 0.0, 0.0]])
    return x0_curv, x0


def create_output_dirs(save_dir):
    """Create output directories for saving data."""
    dirs = [
        save_dir +'/states',
        save_dir + '/road',
        save_dir + '/leader',
        save_dir + '/TV_decision',
        save_dir + '/EV_decision',
        save_dir + '/EV_decision_prob',
        save_dir + '/TV_decision_prob'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    return dirs


def worker(queue, n_sam, args, output_dirs):
    """
    Worker function for parallel data generation.

    Args:
        queue: Multiprocessing queue (for interface compatibility).
        n_sam: Sample number for file naming.
        args: Parsed command line arguments.
    """
    config = Config()
    config_TV = Config_TV()
    config.acados.export_dir = "acados_generatedEV" + current_process().name
    config_TV.acados.export_dir = "acados_generatedTV" + current_process().name

    # Random initial conditions
    init_theta = random.uniform(0, 30)
    diff_theta = random.uniform(0.5, 2)
    init_ec_EV = random.uniform(-0.2, 0.2)
    init_ec_TV = random.uniform(-0.2, 0.2)
    init_phi_EV = random.uniform(-0.2, 0.2)
    init_phi_TV = random.uniform(-0.2, 0.2)
    init_vel_EV = random.uniform(0.1, 1.5)
    init_vel_TV = random.uniform(0.1, 1.5)

    config.mpc.x0_curv, config.mpc.x0 = get_initial_state(
        config, [init_theta, init_ec_EV, init_phi_EV, init_vel_EV])
    config_TV.mpc.x0_curv, config_TV.mpc.x0 = get_initial_state(
        config, [init_theta + diff_theta, init_ec_TV, init_phi_TV, init_vel_TV])

    # Set vx_max based on initial leader/follower roles
    # EV starts behind (init_theta < init_theta + diff_theta)
    # EV is follower -> EV gets higher speed limit (1.5), TV gets lower (1.3)
    if args.scenario == 'ev_behind':
        config.mpc.vx_max = 1.5
        config_TV.mpc.vx_max = 1.3
    elif args.scenario == 'tv_behind':
        config.mpc.vx_max = 1.3
        config_TV.mpc.vx_max = 1.5
    # else: use default values from config

    env = Racing(args=args, config=config, config_tv=config_TV,
                 render_mode="single_rgb_array")

    # Initialize data storage
    sim_step = 1
    stacked_array = np.empty((1, 122)) # total states data
    mu_list = np.empty((1, 2)) # road friction coefficients
    leader_list = np.empty((1, 1)) # leader indicator
    TV_dec_list = np.empty((1, 1)) # TV decision list
    EV_dec_list = np.empty((1, 1)) # EV decision list
    TV_dec_prob_list = np.empty((1, 4)) # TV decision probabilities (strategy profile)
    EV_dec_prob_list = np.empty((1, 4)) # EV mode probabilities (strategy profile)

    for k in range(1, args.max_steps + 1):
        x, info, data_generated = env.step()

        print(f'Sample {n_sam}, Step: {sim_step}')
        mu_value = np.array([env.mu, env.mu_tv])

        # Accumulate data
        leader_list = np.concatenate((leader_list, np.reshape(info['leader'], (1, 1))), axis=0)
        mu_list = np.concatenate((mu_list, np.reshape(mu_value, (1, 2))), axis=0)
        TV_dec_list = np.concatenate((TV_dec_list, np.reshape(env.TV_decision, (1, 1))), axis=0)
        EV_dec_list = np.concatenate((EV_dec_list, np.reshape(env.EV_decision, (1, 1))), axis=0)
        EV_dec_prob_list = np.concatenate((EV_dec_prob_list, np.reshape(env.EV_dec_prob, (1, 4))), axis=0)
        TV_dec_prob_list = np.concatenate((TV_dec_prob_list, np.reshape(env.TV_dec_prob, (1, 4))), axis=0)
        stacked_array = np.concatenate((stacked_array, np.reshape(data_generated, (1, 122))), axis=0)

        if info['status'] != 0:
            break

        if sim_step > args.min_steps:
            print(f'Sample {n_sam} saved')

            # Remove initial empty rows
            stacked_array = np.delete(stacked_array, 0, axis=0) 
            leader_list = np.delete(leader_list, 0, axis=0)
            mu_list = np.delete(mu_list, 0, axis=0)
            TV_dec_list = np.delete(TV_dec_list, 0, axis=0)
            EV_dec_list = np.delete(EV_dec_list, 0, axis=0)
            EV_dec_prob_list = np.delete(EV_dec_prob_list, 0, axis=0)
            TV_dec_prob_list = np.delete(TV_dec_prob_list, 0, axis=0)



            # Save data
            prefix = args.mode + str(n_sam)
            np.save(output_dirs[0]+f'/{prefix}', stacked_array)
            np.save(output_dirs[1]+f'/{prefix}', mu_list)
            np.save(output_dirs[2]+f'/{prefix}', leader_list)
            np.save(output_dirs[3]+f'/{prefix}', TV_dec_list)
            np.save(output_dirs[4]+f'/{prefix}', EV_dec_list)
            np.save(output_dirs[5]+f'/{prefix}', EV_dec_prob_list)
            np.save(output_dirs[6]+f'/{prefix}', TV_dec_prob_list)
            break

        sim_step += 1

    # Cleanup temporary acados files
    acados_dir = 'env/mpc/' + config.acados.export_dir
    if os.path.exists(acados_dir):
        shutil.rmtree(acados_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate racing simulation data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU device ID')
    parser.add_argument('--data_gen_mode', '-data_gen', type=str, default='y') # data generation mode
    parser.add_argument('--scenario', '-sc', type=str, default=None,
                        choices=['ev_behind', 'tv_behind'],
                        help='Initial scenario: ev_behind (EV vx_max=1.5, TV=1.3) or tv_behind (EV=1.3, TV=1.5)')

    # Data generation settings
    parser.add_argument('--num_samples', type=int, default=400,
                        help='Number of samples to generate')
    parser.add_argument('--num_processes', type=int, default=10,
                        help='Number of parallel processes')
    parser.add_argument('--save_dir', type=str, default='data',
                        help='Directory to save generated data')
    parser.add_argument('--mode', type=str, default='sample_',
                        help='Prefix for saved file names')
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='Maximum simulation steps per sample')
    parser.add_argument('--min_steps', type=int, default=2000,
                        help='Minimum steps required before saving')
    parser.add_argument('--name', '-name', type=str, default='n')
    parser.add_argument('--vis', '-vis', type=str, default='n', help='visualization')
    parser.add_argument('--vismode', '-vismode', type=str, default='n', help='vis mode')
    parser.add_argument('--uplot', '-uplot', type=str, default='n', help='uncertainty plot')
    parser.add_argument('--data', '-data', type=str, default='n')
    args = parser.parse_args()

    # Generate sample indices
    sample_numbers = list(range(args.num_samples))
    # Create output directories
    config_nn = Config_NN()
    output_dirs = create_output_dirs(config_nn.data_dir)

    # Run parallel workers in batches
    for idx in range(0, len(sample_numbers), args.num_processes):
        workers = []
        batch = sample_numbers[idx:idx + args.num_processes]

        for n_sam in batch:
            queue = Queue()
            process = Process(target=worker, args=(queue, n_sam, args, output_dirs))
            process.start()
            workers.append(process)

        for process in workers:
            process.join()

    print('Data generation complete!')


if __name__ == "__main__":
    main()
