#!/usr/bin/env python3
import time
import numpy as np
from env import Racing, Config, Config_TV
import argparse
import os
from datetime import datetime

def initialize_configs(args=None):
    cfg = Config()
    cfg_tv = Config_TV()

    cfg.acados.export_dir = 'c4'
    cfg_tv.acados.export_dir = 'd4'

    # Set vx_max based on leader/follower assignment
    if args is not None:
        if args.ev_leader == 'y':
            cfg.mpc.vx_max = cfg.vehicle.leader_vx_max
            cfg_tv.mpc.vx_max = cfg_tv.vehicle.follower_vx_max
        else:
            cfg.mpc.vx_max = cfg.vehicle.follower_vx_max
            cfg_tv.mpc.vx_max = cfg_tv.vehicle.leader_vx_max

    # Get initial positions based on leader/follower assignment
    if args is not None and args.ev_leader == 'y':
        ev_x0 = cfg.vehicle.Leader_thetaecephiV
        tv_x0 = cfg_tv.vehicle.Follower_thetaecephiV
    else:
        ev_x0 = cfg.vehicle.Follower_thetaecephiV
        tv_x0 = cfg_tv.vehicle.Leader_thetaecephiV

    x0_list = [ev_x0, tv_x0]

    for i, c in enumerate([cfg, cfg_tv]):
        vehicle_x0 = x0_list[i]

        c.mode.EV_x0_thetaecephiV = x0_list[0]
        c.mode.TV_x0_thetaecephiV = x0_list[1]

        c.mpc.thetaecephiV = vehicle_x0
        
        # Calculate derived kinematic states
        c.mpc.x0_curv = np.array([*c.mpc.thetaecephiV, 0.0, 0.0, 0.0, 0.0])
        c.mpc.xyref = c.mpc.track(c.mpc.thetaecephiV[0])
        c.mpc.dp = c.mpc.track(c.mpc.thetaecephiV[0], 1)
        
        c.mpc.psiref = np.arctan2(c.mpc.dp[1], c.mpc.dp[0])
        
        c.mpc.eX = -np.sin(c.mpc.psiref) * c.mpc.thetaecephiV[1] 
        c.mpc.eY =  np.cos(c.mpc.psiref) * c.mpc.thetaecephiV[1]
        
        c.mpc.x = c.mpc.xyref[0] + c.mpc.eX 
        c.mpc.y = c.mpc.xyref[1] + c.mpc.eY 
        
        c.mpc.x0 = np.array([
            [c.mpc.x, c.mpc.y, c.mpc.psiref + c.mpc.thetaecephiV[2]], 
            [c.mpc.thetaecephiV[3], 0.0, 0.0, 0.0, 0.0]
        ])
    return cfg, cfg_tv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--exp', '-exp', type=str, default=None)
    parser.add_argument('--data_gen_mode', '-data_gen', type=str, default='n') # data generation mode
    parser.add_argument('--ev_leader', '-ev_leader', type=str, default='y', help='Set EV as leader if y, else TV as leader')
    return parser.parse_args()

def main():
    args = get_args()
    config, config_tv = initialize_configs(args)

    # Result directory setup
    sim_result_dir = config.env.sim_result_dir
    os.makedirs(sim_result_dir, exist_ok=True)
    exp_name = args.exp if args.exp is not None else datetime.now().strftime("%Y%m%d_%H%M%S") 

    # Env initialization
    env = Racing(args = args, config = config, config_tv=config_tv, render_mode="rgb_array", exp_name=exp_name)

    # Simulation variables
    tsum = 0.0
    
    N = 10000
    buffer = []
    simX = np.zeros((10000, 10))
    SIM_STEP = 1

    print(f'Starting simulation for experiment: {exp_name}')
    print(f'  vx_max - Leader: {config_tv.mpc.vx_max}, Follower: {config.mpc.vx_max}')
    tic = time.time()
    try:
        for k in range(1, N):
            # --------------------------------------------------
            # 1. Step Environment
            # --------------------------------------------------
            x, info, data_generated = env.step()
            SIM_STEP  += 1

            # --------------------------------------------------
            # 2. Logging & Video Export
            # --------------------------------------------------
            if k <= 50:
                print(f"Step {k}: Solver Status={info['status']}")

            # 4. Periodic Video Export & Terminal Condition
            if env.lap >= 2 or SIM_STEP >= 700 or SIM_STEP % 100 == 0:
                print(f"Step {SIM_STEP} | Laps: {env.lap} | Exporting video...")
                env.export_video(filename=os.path.join(sim_result_dir, f"{exp_name}.mp4"))
                
                if env.lap >= 2 or SIM_STEP >= 700:
                    break
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    toc = time.time()
    print(f"Simulation completed in {toc - tic:.2f} seconds.")

if __name__ == "__main__":
    main()