import numpy as np
import torch

class PerceptionModule:
    def __init__(self, config, pred_model=None):
        self.config = config
        self.pred_model = pred_model
        self.S_MAX = self.config.mpc.S_MAX
        self.rng = np.random.default_rng(77)

        # Low Friction section 
        self.slippery_intervals = [
            (2, 4, self.config.vehicle.mu_wet3),
            (4, 6, self.config.vehicle.mu_wet2),
            (6, 7, self.config.vehicle.mu_wet3),
        ]

    def get_friction(self, theta):
        """Return friction coefficient based on position (theta)"""
        for start, end, mu_val in self.slippery_intervals:
            if start <= theta % self.S_MAX < end:
                return mu_val
        return self.config.vehicle.mu

    def get_fallback_prediction(self, mpc_ev, mpc_tv):
        """
        Generate heuristic-based predictions when the deep model is unavailable. (Before full sequence data is collected, or data generation mode)
        """
        
        # Generate dummy TV prediction from Ego's previous trajectory + noise
        predicted_traj_from_TV = np.array([mpc_ev.trajectory, mpc_ev.trajectory], dtype=object)
        predicted_traj_from_TV += self.rng.normal(loc=0.0, scale=0.04, size=predicted_traj_from_TV.shape)

        # Generate dummy Ego prediction from TV's previous trajectory + noise
        predicted_traj_from_EV = np.array([mpc_tv.trajectory, mpc_tv.trajectory], dtype=object)
        predicted_traj_from_EV += self.rng.normal(loc=0.0, scale=0.02, size=predicted_traj_from_EV.shape)

        return predicted_traj_from_EV, predicted_traj_from_TV


    def predict_opponent_trajectory(self, state_list, tv_state_list, data, context):
        """
        Use the prediction model to forecast the opponent's future trajectory.
        """
        if self.pred_model is None:
            return None, None, None, None

        pred_traj, pred_mode, sampled_traj, sampled_mean, sampled_log_var = \
            self.pred_model.load_model(
                state_list, data['ev_kappa'], context['ev_dec'], context['ev_prob'],
                tv_state_list, data['tv_kappa'], context['tv_dec'], context['tv_prob'],
                context['leader'], 'EV', context['sim_step']
            )
        
        return pred_traj, pred_mode, sampled_traj, sampled_mean, sampled_log_var

