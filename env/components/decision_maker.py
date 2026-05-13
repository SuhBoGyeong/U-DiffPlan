from numpy import array, zeros, hstack, sin, cos, arctan2, hstack, sqrt, dot, linspace, c_
from scipy import interpolate
import time
import casadi as ca
import random
import numpy as np 
from plot_func import getTrack, get_boundary_future

random.seed(77)


def utility_function(EV_future_traj, TV_future_traj, EV_dec_prob, TV_dec_prob, EV_state, TV_state, config):
    """
    Compute utility for Ego Vehicle (EV) based on
    predicted future trajectories and game strategies.

    Utility components:
        - Collision risk
        - Progress reward
        - Fuel consumption
        - Drafting benefit
        - Strategy interaction reward
    """
    EV_future_traj = EV_future_traj.astype('float64')
    TV_future_traj = TV_future_traj.astype('float64')

    # --------------------------------------------------
    # 1. Distance & Collision Risk
    # --------------------------------------------------

    distances = np.linalg.norm(EV_future_traj[:, :2] - TV_future_traj[:, :2], axis=1)

    collision_th = config.decision_maker.collision_th
    collision_risk = np.sum(distances < collision_th) / len(distances)
    collision_risk_weight = config.decision_maker.collision_risk_weight

    # --------------------------------------------------
    # 2. Progress
    # --------------------------------------------------

    map_file = config.env.map_file
    EV_start_idx, EV_end_idx = get_boundary_future(map_file, EV_future_traj)
    TV_start_idx, TV_end_idx = get_boundary_future(map_file, TV_future_traj)

    def compute_progress(traj):
        return np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
    EV_progress = compute_progress(EV_future_traj)
    TV_progress = compute_progress(TV_future_traj)

    # --------------------------------------------------
    # 3. Fuel Consumption
    # --------------------------------------------------

    def compute_fuel(traj):
        speed = np.linalg.norm(traj[:, 3:5], axis=1)
        drive = traj[:, 7]
        return np.sum(drive * speed)
    EV_fuel = compute_fuel(EV_future_traj)
    TV_fuel = compute_fuel(TV_future_traj)

    # --------------------------------------------------
    # 4. Drafting Ratio
    # --------------------------------------------------

    EV_heading = np.stack([
        np.cos(EV_future_traj[:, 2]),  # cos(psi)
        np.sin(EV_future_traj[:, 2])   # sin(psi)
    ], axis=1)  # shape: (N, 2)

    TV_to_EV = EV_future_traj[:, :2] - TV_future_traj[:, :2]
    TV_to_EV_unit = TV_to_EV / (np.linalg.norm(TV_to_EV, axis=1, keepdims=True) + 1e-6)

    cos_theta = np.sum(EV_heading * TV_to_EV_unit, axis=1) 
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

    drafting_zone = (distances < config.decision_maker.draft_dist) & (angles < np.deg2rad(config.decision_maker.draft_deg))
    drafting_ratio = np.sum(drafting_zone) / len(distances)

    # --------------------------------------------------
    # 5. Leader / Follower Role Assignment
    # --------------------------------------------------

    EV_is_leader = EV_dec_prob >= 2
    
    if EV_is_leader:
        leader_progress, follower_progress = EV_progress, TV_progress
        leader_fuel, follower_fuel = EV_fuel, TV_fuel
        leader_mode, follower_mode = EV_dec_prob, TV_dec_prob
        leader_start, follower_start = EV_start_idx, TV_start_idx
        leader_end, follower_end = EV_end_idx, TV_end_idx
    else:
        leader_progress, follower_progress = TV_progress, EV_progress
        leader_fuel, follower_fuel = TV_fuel, EV_fuel
        leader_mode, follower_mode = TV_dec_prob, EV_dec_prob
        leader_start, follower_start = TV_start_idx, EV_start_idx
        leader_end, follower_end = TV_end_idx, EV_end_idx

    # --------------------------------------------------
    # 6. Fuel Weight Adjustment
    # --------------------------------------------------

    fuel_weight = config.decision_maker.fuel_weight
    leader_fuel_weight = fuel_weight
    follower_fuel_weight = fuel_weight * (1 + 0.5 * drafting_ratio)

    # --------------------------------------------------
    # 7. Overtaking / Blocking Success
    # --------------------------------------------------

    if (leader_start - follower_start) * (leader_end - follower_end) < 0:
        overtaking_suc = config.decision_maker.overtaking_suc_reward
        blocking_suc = 0
    else:
        overtaking_suc = 0
        blocking_suc = config.decision_maker.blocking_suc_reward 

    # --------------------------------------------------
    # 8. Utility Calculation by Strategy Pair
    # --------------------------------------------------
    cfg = config.decision_maker
    blocking_weight = cfg.blocking_weight/max(distances[0], 1e-3) # 0.2/distances[0]  
    overtaking_weight = cfg.overtaking_weight 

    if leader_mode == 2 and follower_mode == 0:
        # (driving, following)
        leader_u = (
            leader_progress * cfg.leader_progress_rate_weight1
            + leader_fuel_weight * leader_fuel
        )
        follower_u = (
            follower_progress * cfg.follower_progress_weight1
            + follower_fuel_weight * follower_fuel
        )

    elif leader_mode == 2 and follower_mode == 1:
        # (driving, overtaking)
        leader_u = (
            leader_progress * cfg.leader_progress_rate_weight1
            - overtaking_suc
            + leader_fuel_weight * leader_fuel
        )
        follower_u = overtaking_weight * (
            follower_progress * cfg.follower_progress_weight1
            + overtaking_suc
            - collision_risk * collision_risk_weight
            + follower_fuel_weight * follower_fuel
        )

    elif leader_mode == 3 and follower_mode == 0:
        # (blocking, following)
        leader_u = blocking_weight * (
            leader_progress * cfg.leader_progress_rate_weight2
            + blocking_suc
            - collision_risk * collision_risk_weight
            + leader_fuel_weight * leader_fuel
        )
        follower_u = (
            follower_progress * cfg.follower_progress_weight2
            + follower_fuel_weight * follower_fuel
        )

    else:
        # (blocking, overtaking)
        # If overtaking succeeds despite blocking, the follower should receive a higher reward.
        # The leader's reward for blocking could be adjusted downward if the block fails.
        leader_u = blocking_weight * (
            leader_progress * cfg.leader_progress_rate_weight2
            + blocking_suc
            - cfg.overtaking_suc_weight * overtaking_suc
            - collision_risk * collision_risk_weight
            + leader_fuel_weight * leader_fuel
        )

        follower_u = overtaking_weight * (
            follower_progress * cfg.follower_progress_weight2
            + overtaking_suc
            - blocking_suc
            - collision_risk * collision_risk_weight
            + follower_fuel_weight * follower_fuel
        )

    EV_u = leader_u if EV_is_leader else follower_u
    
    return EV_u


class DecisionMaker():
    """
    Game-theoretic decision maker for ego vehicle (EV).

    The EV selects a strategy based on:
        - predicted opponent strategies
        - expected utilities
        - current mixed strategy probabilities

    Strategy Index:
        0: Following
        1: Overtaking
        2: Driving (Leader)
        3: Blocking (Leader)
    """
    FOLLOW = 0
    OVERTAKE = 1
    DRIVE = 2
    BLOCK = 3

    def __init__(self):
        super().__init__()

    def _compute_expected_utility(
            self,
            ev_actions,
            tv_actions,
            state,
            state_tv,
            theta_tv,
            pred_mode,
            pred_traj,
            mpc,
            config,
        ):
            """
            Compute expected utility for each EV action given TV predictions.
            """

            utilities = []

            for ev_action in ev_actions:
                expected_u = 0.0

                for i, tv_action in enumerate(tv_actions):

                    ev_future_traj, _ = mpc.solve_for_decision(
                        state,
                        state_tv,
                        theta_tv,
                        ev_action,
                        tv_action,
                        obstacle_trajectories = pred_traj[i],
                    )

                    u = utility_function(
                        ev_future_traj,
                        pred_traj[i],
                        ev_action,
                        tv_action,
                        state,
                        state_tv,
                        config,
                    )

                    expected_u += pred_mode[i] * u

                utilities.append(expected_u)

            return utilities

    def _update_strategy(self, mode_ev, utilities, offset=0):
        """
        Update mixed strategy probabilities using incentive-based update rule.
        """

        total_u = np.sum(np.array(mode_ev) * np.array(utilities))

        incentives = [
            max(0.1, u - total_u)
            for u in utilities
        ]

        new_mode1 = (mode_ev[0] + incentives[0]) / (
            1 + incentives[0] + incentives[1]
        )
        new_mode2 = 1 - new_mode1

        decision = np.argmax([new_mode1, new_mode2]) + offset

        return decision, [new_mode1, new_mode2]

    def solve(self, state, state_tv, theta, theta_tv, pred_mode, pred_traj, mode_ev, mpc, config): 
        """
        Main decision solver.
        Determines EV role (leader/follower) and updates strategy.
        """
        # -------------------------
        # EV is Leader
        # -------------------------
        if theta > theta_tv:

            ev_actions = [self.DRIVE, self.BLOCK]
            tv_actions = [self.FOLLOW, self.OVERTAKE]
            current_mode = [mode_ev[2], mode_ev[3]]

            utilities = self._compute_expected_utility(
                ev_actions,
                tv_actions,
                state,
                state_tv,
                theta_tv,
                pred_mode,
                pred_traj,
                mpc,
                config,
            )

            decision, new_modes = self._update_strategy(
                current_mode,
                utilities,
                offset=2,
            )

            mode_ev = [0, 0, new_modes[0], new_modes[1]]

        # -------------------------
        # EV is Follower
        # -------------------------
        elif theta < theta_tv:

            ev_actions = [self.FOLLOW, self.OVERTAKE]
            tv_actions = [self.DRIVE, self.BLOCK]
            current_mode = [mode_ev[0], mode_ev[1]]

            utilities = self._compute_expected_utility(
                ev_actions,
                tv_actions,
                state,
                state_tv,
                theta_tv,
                pred_mode,
                pred_traj,
                mpc,
                config,
            )

            decision, new_modes = self._update_strategy(
                current_mode,
                utilities,
                offset=0,
            )

            mode_ev = [new_modes[0], new_modes[1], 0, 0]

        else:
            decision = None

        return decision, mode_ev