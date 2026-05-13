#!/usr/bin/env python3
from typing import Optional
from time import asctime

from gym.core import Env
from gym import spaces

import numpy as np
from numpy import array, linspace, c_
from scipy.integrate import solve_ivp


from .track import Track

from .vehicle_model import VehicleModel
from .mpc import MPC, MPC_baseline
from .components.decision_maker import DecisionMaker
from .components.metrics import RacingMetrics
from .components.perception import PerceptionModule
from .components.renderer import Renderer
from .components.visualization import Visualizer
from .components.live_plot import Clock, LivePlot2Players

import random
import casadi as ca
import torch

from simulate_model import Load_Model 

random.seed(77)
np.random.seed(77)


class Racing(Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60
    }

    def __init__(self, args, config, config_tv, render_mode: Optional[str] = None, exp_name: Optional[str] = None):
        self.args = args 
        self.config = config
        self.config_tv = config_tv
        self.sim_result_dir = self.config.env.sim_result_dir
        self.exp_name = exp_name

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.metadata["render_fps"] = self.config.env.render_fps

        self.is_X11_forwarding = self.config.env.is_X11_forwarding
        self.detach_video_export_process = self.config.env.detach_video_export_process
        self.screen = None
        self.clock = None
        self.isopen = None

        self.low = np.array([], dtype=np.float32)
        self.high = np.array([], dtype=np.float32)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.dt = self.config.env.dt
        self.model = VehicleModel(self.config)
        self.model_TV = VehicleModel(self.config_tv)
        self.track = Track(self.config.env.track_filename)

        self.decision_maker = DecisionMaker()

        self.mpc_tv = MPC_baseline(self.model_TV, self.track, self.config_tv)

        self.state_tv = np.hstack(self.config_tv.mpc.x0)

        self.TV_dec_prob = [0, 0, 0.5, 0.5]
        self.TV_decision = random.choices([0, 1, 2, 3], self.TV_dec_prob, k=1)[0]
            
        self.mpc = MPC(self.model, self.track, self.config)

        self.lbu = np.array([self.config.mpc.delta_min, self.config.mpc.D_min])
        self.ubu = np.array([self.config.mpc.delta_max, self.config.mpc.D_max])
        self.lbdu = np.array([self.config.mpc.ddelta_min, self.config.mpc.dD_min])
        self.ubdu = np.array([self.config.mpc.ddelta_max, self.config.mpc.dD_max])

        self.integrator_type = self.config.env.integrator_type
        self.sim_method_num_steps = self.config.env.sim_method_num_steps
        self.state = np.hstack(self.config.mpc.x0)
        

        self.lap = 0
        self.simstep = 0



        self.update_step = 3
        self.fix_decision = 0

        self.EV_prev_mode = 0

        self.S_MAX = self.track.thetaref[-1]


        if self.args.data_gen_mode == 'n':
            self.pred_model = Load_Model(self.args)
        else:
            self.pred_model = None

        self.metrics = RacingMetrics()
        self.perception = PerceptionModule(self.config, self.pred_model)
        self.visualizer = Visualizer(self.config, self.renderer, self.render_mode, self.dt)
    
        self.state_list = []
        self.state_tv_list = []
        self.leader_durations = []

        

        self.predicted_traj_from_EV = None
        self.predicted_traj_from_TV = None

        self.times = 0
        self.mpc_times = 0
        self.wet_road_history = []

        self.mu = self.config.vehicle.mu
        self.mu_tv = self.config.vehicle.mu
        self.mu_list = []
        self.mu_tv_list = []
        self.log_var_r_list = []
        self.theta_EV_init = 0
        
        self.sequence_length = self.config.env.sequence_length

        # Store MPC horizon (N) and trajectory length (N+1) for dynamic slicing
        self.N = self.config.mpc.N
        self.traj_len = self.N + 1  # Trajectory has N+1 points

        self.kapparef = ca.interpolant("kapparef_s", "bspline", [self.track.thetaref], self.track.kapparef)
        self.kapparef_tv = ca.interpolant("kapparef_s", "bspline", [self.track.thetaref], self.track.kapparef)

        self.slippery_intervals = [
            (2, 4, self.config.vehicle.mu_wet3),
            (4, 6, self.config.vehicle.mu_wet2),
            (6, 7, self.config.vehicle.mu_wet3),
        ]

    def _get_mu_for_theta(self, theta):
        """Get friction coefficient for given theta position."""
        for start, end, mu_val in self.slippery_intervals:
            if start <= theta % self.S_MAX < end:
                return mu_val
        return self.config.vehicle.mu

    def _get_prediction_modes(self, is_ev_leader, add_noise=False):
        """Get prediction modes based on leader/follower status."""
        if is_ev_leader:
            base_tv = self.EV_dec_prob[2]
            base_ev = self.TV_dec_prob[0]
            decisions_tv = (2, 3)
            decisions_ev = (0, 1)
        else:
            base_tv = self.EV_dec_prob[0]
            base_ev = self.TV_dec_prob[2]
            decisions_tv = (0, 1)
            decisions_ev = (2, 3)

        if add_noise:
            random_pred_tv = np.random.uniform(-0.1, 0.1) + base_tv
            random_pred_ev = np.random.uniform(-0.1, 0.1) + base_ev
        else:
            random_pred_tv = base_tv
            random_pred_ev = base_ev

        pred_mode_from_TV = [random_pred_tv, 1 - random_pred_tv]
        pred_mode_from_EV = [random_pred_ev, 1 - random_pred_ev]

        return pred_mode_from_TV, pred_mode_from_EV, decisions_tv, decisions_ev

    def _update_decisions(self, pred_mode_from_TV, pred_mode_from_EV, predicted_traj_from_TV, predicted_traj_from_EV):
        """Update EV and TV decisions based on prediction modes."""
        if self.simulation_step % self.update_step == 0:
            if self.fix_decision == 0:
                self.TV_decision, self.TV_dec_prob = self.decision_maker.solve(
                    self.state_tv, self.state, self.mpc_tv.theta, self.mpc.theta,
                    pred_mode_from_TV, predicted_traj_from_TV, self.TV_dec_prob, self.mpc_tv, self.config)
                self.EV_decision, self.EV_dec_prob = self.decision_maker.solve(
                    self.state, self.state_tv, self.mpc.theta, self.mpc_tv.theta,
                    pred_mode_from_EV, predicted_traj_from_EV, self.EV_dec_prob, self.mpc, self.config)
            elif self.fix_decision > 0: #! for first 5 steps
                self.TV_decision, self.EV_decision = 0, 2
                self.fix_decision -= 1
            elif self.fix_decision < 0: #! for first 5 steps
                self.TV_decision, self.EV_decision = 2, 0
                self.fix_decision += 1

    def _update_environmental_factors(self):
            """Update road friction based on vehicle positions."""
            self.mu = self.perception.get_friction(self.mpc.theta)
            self.mu_tv = self.perception.get_friction(self.mpc_tv.theta)

            is_wet = self.mu != self.config.vehicle.mu
            is_wet_tv = self.mu_tv != self.config.vehicle.mu
            
            self.model.set_wet_road(is_wet, self.mu)
            self.model_TV.set_wet_road(is_wet_tv, self.mu_tv)

            self.mu_list.append(self.mu)
            self.mu_tv_list.append(self.mu_tv)

    def _extract_features(self):
        """
        Based on the current state and track information, extract features for the prediction model.
        """
        N = self.config.mpc.N
        ref_horizon_length = self.config.mpc.pp_ref_horizon_length
        
        # 1. Extract track-based local coordinates (theta, ey) and curvature (kappa) for EV and TV
        theta0, ey0 = self.track.get_theta(*self.state[:2], initial_guess=self.mpc.theta, eval_ey=True)
        kapparefhorizon = array(self.kapparef(theta0 + linspace(0, ref_horizon_length, N + 1))).squeeze()
        
        theta0_tv, ey0_tv = self.track.get_theta(*self.state_tv[:2], initial_guess=self.mpc_tv.theta, eval_ey=True)
        kapparefhorizon_tv = array(self.kapparef_tv(theta0_tv + linspace(0, ref_horizon_length, N + 1))).squeeze()

        # 2. Data integration
        data_ev = np.c_[[self.state], self.mpc.theta, ey0][0]      # [x, y, psi, vx, vy, omega, delta, D, theta, ey]
        data_tv = np.c_[[self.state_tv], self.mpc_tv.theta, ey0_tv][0]

        # 3. Update of sequence data
        if self.simulation_step > 0:
            self.state_list.append(data_ev)
            self.state_tv_list.append(data_tv)
            
        # 4. Full data, curvature info
        data_full = np.concatenate([data_ev, data_tv, kapparefhorizon, kapparefhorizon_tv])
        
        return data_full, kapparefhorizon, kapparefhorizon_tv


    def _handle_prediction_logic(self, is_fallback, is_available):
        if is_fallback and is_available:
            # Get heuristic trajectories from perception module
            self.predicted_traj_from_EV, self.predicted_traj_from_TV = self.perception.get_fallback_prediction(self.mpc, self.mpc_tv)
            
            # Additional fallback updates (modes, uncertainty)
            is_ev_leader = self.mpc.theta > self.mpc_tv.theta
            pred_mode_from_TV, pred_mode_from_EV, _, _ = self._get_prediction_modes(is_ev_leader)
            
            self.predicted_traj_from_EV_mean = None
            self.predicted_traj_from_EV_log_var = None
            self._update_decisions(pred_mode_from_TV, pred_mode_from_EV, self.predicted_traj_from_TV, self.predicted_traj_from_EV)

        elif is_available:
            context = {
                'ev_dec': self.EV_decision, 'ev_prob': self.EV_dec_prob,
                'tv_dec': self.TV_decision, 'tv_prob': self.TV_dec_prob,
                'leader': self.leader, 'sim_step': self.simulation_step
            }
            data = {'ev_kappa': self.EV_kappa, 'tv_kappa': self.TV_kappa}

            is_ev_leader = self.mpc.theta > self.mpc_tv.theta
            self.pred_mode_from_TV, _, _, _ = self._get_prediction_modes(is_ev_leader, add_noise=True)
            _, self.predicted_traj_from_TV = self.perception.get_fallback_prediction(self.mpc, self.mpc_tv)

            # Get prediction from model
            predicted_traj_from_EV, self.pred_mode_from_EV, sampled_traj_from_EV, sampled_traj_from_EV_mean, sampled_traj_from_EV_log_var = self.perception.predict_opponent_trajectory(
                self.state_list[-self.sequence_length:], 
                self.state_tv_list[-self.sequence_length:], 
                data, context
            )

            # Store full prediction for evaluation
            self.predicted_traj_from_EV_full = predicted_traj_from_EV.copy()
            self.sampled_traj_from_EV_full = sampled_traj_from_EV.copy()

            # For MPC, use first N+1 steps
            self.predicted_traj_from_EV_mean = sampled_traj_from_EV_mean[:self.traj_len, :]
            self.predicted_traj_from_EV_log_var = sampled_traj_from_EV_log_var[:self.traj_len, :]
            self.sampled_traj_from_EV = sampled_traj_from_EV[:, :self.traj_len, :]
            self.sampled_traj_from_EV_mean = sampled_traj_from_EV_mean[:self.traj_len, :]
            self.sampled_traj_from_EV_log_var = sampled_traj_from_EV_log_var[:self.traj_len, :]

            predicted_traj_from_EV1 = np.concatenate([predicted_traj_from_EV[:self.traj_len,:], self.mpc.trajectory[:,2:]], axis=1)
            predicted_traj_from_EV2 = np.concatenate([predicted_traj_from_EV[:self.traj_len,:], self.mpc.trajectory[:,2:]], axis=1)
            self.predicted_traj_from_EV = np.array([predicted_traj_from_EV1, predicted_traj_from_EV2], dtype=object)
            
            self._update_decisions(self.pred_mode_from_TV, self.pred_mode_from_EV, self.predicted_traj_from_TV, self.predicted_traj_from_EV)
        
        else:
            self.predicted_traj_from_EV_mean = None
            self.predicted_traj_from_EV_log_var = None

    def _solve_control_step(self, is_fallback):
        if is_fallback:
            self.solver_status_tv = self.mpc_tv.solve(self.state_tv, self.state, self.mpc.theta, self.TV_decision, self.EV_decision, \
                                    obstacle_trajectories=self.mpc.trajectory)
            self.log_var_r_list.append(0)
        else:
            if self.mpc.theta > self.mpc_tv.theta:
                # EV is leader
                predicted_decision_from_TV = np.argmax(self.pred_mode_from_TV) + 2
            else:
                predicted_decision_from_TV = np.argmax(self.pred_mode_from_TV)

            self.solver_status_tv = self.mpc_tv.solve(self.state_tv, self.state, self.mpc.theta, self.TV_decision, predicted_decision_from_TV,\
                                                    obstacle_trajectories=self.predicted_traj_from_TV[np.argmax(self.pred_mode_from_TV)])
        
        if is_fallback:
            self.solver_status = self.mpc.solve(self.state, self.state_tv, self.mpc_tv.theta, self.EV_decision, self.TV_decision, \
                                                obstacle_trajectories=self.mpc_tv.trajectory)
        else:
            if self.mpc.theta > self.mpc_tv.theta:
                # EV is leader
                predicted_decision_from_EV = np.argmax(self.pred_mode_from_EV)
            else:
                predicted_decision_from_EV = np.argmax(self.pred_mode_from_EV) + 2

            log_var_r = np.sqrt(self.sampled_traj_from_EV_log_var[:,0]**2 + self.sampled_traj_from_EV_log_var[:,1]**2)
            log_var_r_x = self.sampled_traj_from_EV_log_var[:,0]
            log_var_r_y = self.sampled_traj_from_EV_log_var[:,1]
            # Regular MPC with uncertainty
            self.solver_status  = self.mpc.solve(self.state, self.state_tv, self.mpc_tv.theta, self.EV_decision, \
                                            predicted_decision_from_EV, obstacle_trajectories=self.predicted_traj_from_EV[np.argmax(self.pred_mode_from_EV)], \
                                            sampled_traj = self.sampled_traj_from_EV, sampled_traj_mean = np.mean(self.sampled_traj_from_EV_mean, axis=1), \
                                            sampled_traj_log_var_x = log_var_r_x, sampled_traj_log_var_y = log_var_r_y)
            self.log_var_r_list.append(log_var_r.mean())

    def _integrate_dynamics(self):
        theta_ev = self.mpc.trajectory[0,-3]
        theta_tv = self.mpc_tv.trajectory[0,-3]

        xypsi_ev = self.state[:3]
        xypsi_tv = self.state_tv[:3]

        self.state_before = self.state.copy()

        self.sol_tv = solve_ivp(
        lambda t, x : self.model.f(*x, *np.clip(self.mpc_tv.control, self.lbdu, self.ubdu),theta_tv, theta_ev,*xypsi_ev),
        (0, self.dt),
        self.state_tv,
        method=self.integrator_type
        )
        self.state_tv = self.sol_tv.y[:, -1]

        self.sol = solve_ivp(
        lambda t, x : self.model.f(*x, *np.clip(self.mpc.control, self.lbdu, self.ubdu),theta_ev,theta_tv,*xypsi_tv),
        (0, self.dt),
        self.state,
        method=self.integrator_type
        )
        self.state = self.sol.y[:, -1]

    def _post_step_updates(self, solver_status):
        if self.simulation_step == 2:
            self.theta_EV_init = self.mpc.theta
        if self.mpc.theta - self.theta_EV_init > self.track.thetaref[-1] * (self.lap+1) and self.simulation_step >= 2:
            self.lap += 1
        self._update_environmental_factors()
        self.leader = self.metrics.update(self.simulation_step,
                                          self.state,
                                          self.state_tv,
                                          self.mpc.theta,
                                          self.mpc_tv.theta,
                                        )

        # If leader changes
        if self.metrics.overtaking_suc[-1] == 1:
            # reinitialize probability 
            self.EV_dec_prob = [0, 0, 1, 0]
            self.TV_dec_prob = [0, 1, 0, 0]
            self.fix_decision = 5
        elif self.metrics.overtook_suc[-1] == 1:
            # reinitialize probability 
            self.EV_dec_prob = [0, 1, 0, 0]
            self.TV_dec_prob = [0, 0, 1, 0]
            self.fix_decision = -5

        # # Store informations
        self.info = self.metrics.get_info(self.lap, solver_status)
        self.info.update({'X': self.state[0],
                'Y': self.state[1],
                'phi': self.state[2],
                'vx': self.state[3],
                'vy': self.state[4],
                'omega': self.state[5],
                'delta': self.state[6],
                'd': self.state[7],
                'theta': self.mpc.theta,
                'ec': self.mpc.ec,
                })
        
        self.data_generate_ev = np.c_[[self.state], self.mpc.theta, self.mpc.ec][0]
        self.data_generate_tv = np.c_[[self.state_tv], self.mpc_tv.theta, self.mpc_tv.ec][0]
        self.data_generate = np.concatenate([self.data_generate_ev, self.data_generate_tv])

        self.kappas = np.concatenate([self.mpc.kapparefhorizon,self.mpc_tv.kapparefhorizon])
        self.data_generate = np.concatenate([self.data_generate, self.kappas])

    def step(self):
        # --------------------------------------------------
        # 1. Initialization
        # --------------------------------------------------
        if self.mpc.theta is not None and self.mpc_tv.theta is not None:
            self.simulation_step += 1
        else:
            self.TV_dec_prob = [0, 0, 0.5, 0.5]
            self.TV_decision = 3
            
            self.EV_dec_prob = [0.5, 0.5, 0, 0]
            self.EV_decision = random.choices([0, 1, 2, 3], self.EV_dec_prob, k=1)[0]
            self.simulation_step = 0

        # --------------------------------------------------
        # 2. Feature Extraction
        # --------------------------------------------------
        data_generate, self.EV_kappa, self.TV_kappa = self._extract_features()

        # --------------------------------------------------
        # 3. Prediction Module
        # --------------------------------------------------
        is_fallback = (self.args.data_gen_mode == 'y' or 
                       len(self.state_list) < self.sequence_length) 
        is_available = (self.mpc.theta !=None and self.mpc_tv.theta !=None)
        self._handle_prediction_logic(is_fallback, is_available)

        # --------------------------------------------------
        # 4. MPC SOLVER
        # --------------------------------------------------
        self._solve_control_step(is_fallback)

        # --------------------------------------------------
        # 5. SIMULATION STEP    
        # --------------------------------------------------
        self._integrate_dynamics()
        self.renderer.render_step(self.EV_decision, self.TV_decision)
        self._post_step_updates(self.solver_status)

        return np.array(self.state, dtype=np.float32), self.info, self.data_generate
    

    def export_video(self, filename=None):
        """Calls the visualizer to export the collected frames."""
        return self.visualizer.export_video(self.screen, filename)

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]

        if self.screen is None:
            self.screen = LivePlot2Players(self.track, self.model, mode=mode)

        if self.clock is None: self.clock = Clock()

        # Update visualization data
        self.screen.update(
            self.state, self.mpc.trajectory, self.state_tv, self.mpc_tv.trajectory,
            self.EV_dec_prob, self.TV_dec_prob, self.EV_decision, self.TV_decision,
            self.predicted_traj_from_EV, self.predicted_traj_from_TV,
            self.mu, self.mu_tv,
            self.predicted_traj_from_EV_mean, self.predicted_traj_from_EV_log_var
        )

        if mode == "human":
            if self.metadata["render_fps"] > 0: self.clock.tick(self.metadata["render_fps"])
            if self.is_X11_forwarding: self.screen.start_event_loop()
        elif mode in {"rgb_array", "single_rgb_array", "raw_data"}:
            return self.screen.get_data()
        return None