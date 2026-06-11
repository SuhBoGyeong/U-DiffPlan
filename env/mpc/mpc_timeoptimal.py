"""Time-Optimal MPC controller.

Identical to mpc.py except using acados_ocp_timeoptimal

"""

from numpy import array, zeros, hstack, sin, cos, arctan2, sqrt, dot, linspace, c_
from .acados_ocp_timeoptimal import acados_ocp_timeoptimal
from .acados_solver import acados_solver
from scipy import interpolate
import casadi as ca
import numpy as np
from pathlib import Path


class MPC_TimeOptimal:

    def __init__(self, vehicle_model, track, config, control_init=None):
        if control_init is None:
            self.control = zeros((2,))
        else:
            self.control = control_init

        acados_ocp = acados_ocp_timeoptimal(vehicle_model, track, config)
        self.__solver              = acados_solver(acados_ocp, config, config.acados.export_dir)
        self.__solver_for_decision = acados_solver(acados_ocp, config, config.acados.export_dir_dec)

        self.track = track
        self.N  = config.mpc.N
        self.dt = config.mpc.dt
        self.nu = acados_ocp.model.u.size1()
        self.nx = acados_ocp.model.x.size1()

        self.lbu  = array([config.mpc.delta_min, config.mpc.D_min])
        self.ubu  = array([config.mpc.delta_max, config.mpc.D_max])
        self.lbdu = array([config.mpc.ddelta_min, config.mpc.dD_min])
        self.ubdu = array([config.mpc.ddelta_max, config.mpc.dD_max])

        self.trajectory     = None
        self.curv_trajectory = None
        self.theta          = None
        self.ec             = 0
        self.el             = 0

        self.SIM_STEP        = 0
        self.export_dir      = 'env/mpc/' + config.acados.export_dir
        self.export_dir_dec  = 'env/mpc/' + config.acados.export_dir_dec
        self._ref_horizon_length = config.mpc.pp_ref_horizon_length

        self._kapparef = ca.interpolant("kapparef_s", "bspline", [track.thetaref], track.kapparef)
        self.config         = config
        self.vehicle_model  = vehicle_model

    # ── helpers ──────────────────────────────────────────────

    def _init_sampled_traj_defaults(self, sampled_traj_mean,
                                     sampled_traj_log_var_x, sampled_traj_log_var_y,
                                     use_nonzero_default=True):
        traj_len = self.N + 1
        if sampled_traj_mean is None:
            sampled_traj_mean = np.zeros(traj_len)

        default_val = 0.01 / np.sqrt(2) if use_nonzero_default else 0.0

        if sampled_traj_log_var_x is None:
            sampled_traj_log_var_x = np.ones(traj_len) * default_val
        if sampled_traj_log_var_y is None:
            sampled_traj_log_var_y = np.ones(traj_len) * default_val

        return sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y

    def _compute_initial_state(self, state):
        theta0, ey0 = self.track.get_theta(*state[:2], initial_guess=self.theta, eval_ey=True)
        psiref = arctan2(*self.track(theta0, 1)[::-1])
        epsi   = state[2] - psiref
        epsi   = arctan2(sin(epsi), cos(epsi))
        x0     = hstack([theta0, ey0, epsi, state[3:]])
        return x0, theta0, ey0

    def _compute_following_trajectory(self, theta0, ey0, obstacle_trajectories, epsilon=0.0):
        interpol_theta = hstack([theta0, obstacle_trajectories[:, 8]])
        interpol_ec    = hstack([ey0,    obstacle_trajectories[:, 9]])
        # Remove duplicate theta values to prevent division by zero in interp1d
        _, unique_idx  = np.unique(interpol_theta, return_index=True)
        interpol_theta = interpol_theta[unique_idx]
        interpol_ec    = interpol_ec[unique_idx]
        if len(interpol_theta) < 2:
            return c_[obstacle_trajectories[:, 8], obstacle_trajectories[:, 9]]
        f_interpol     = interpolate.interp1d(interpol_theta, interpol_ec, kind='linear')

        dist_s   = abs(obstacle_trajectories[0][8] - theta0)
        thetaref = linspace(theta0, obstacle_trajectories[-1, 8] - dist_s + epsilon, self.N + 1)

        theta_min = f_interpol.x[0]
        theta_max = f_interpol.x[-1]
        thetaref  = np.clip(thetaref, theta_min, theta_max)

        return c_[obstacle_trajectories[:, 8], f_interpol(thetaref)]

    def _build_solver_params(self, k, theta0, driving_mode, driving_mode_other,
                              obstacle_traj_k, following_traj_k,
                              sampled_traj_mean_k, log_var_x_k, log_var_y_k):

        weights = self.config.mpc.strategy_weights[driving_mode]
        return hstack([
            theta0 + k * self._ref_horizon_length / self.N,
            obstacle_traj_k[:3],
            following_traj_k,
            driving_mode,
            driving_mode_other,
            weights["qtheta"],
            weights["qec"],
            weights["rdelta"],
            weights["rddelta"],
            weights["romega"],
            sampled_traj_mean_k,
            log_var_x_k,
            log_var_y_k,
            self.config.mpc.D_CRIT_weight[driving_mode],
            self.config.mpc.uh_weight[driving_mode]
        ])

    def _convert_to_cartesian(self, trajectory, state):
        XY   = self.track(trajectory[:, 0])
        dXY  = self.track(trajectory[:, 0], 1)
        psiref = arctan2(dXY[:, 1], dXY[:, 0])

        eX = -sin(psiref) * trajectory[:, 1]
        eY =  cos(psiref) * trajectory[:, 1]
        trajectory[:, 0] = XY[:, 0] + eX
        trajectory[:, 1] = XY[:, 1] + eY
        trajectory[:, 2] += psiref

        el  = sqrt(dot(state[:2] - trajectory[0, :2], state[:2] - trajectory[0, :2]))
        eXX = cos(psiref) * el
        eYY = sin(psiref) * el
        trajectory[:, 0] += eXX
        trajectory[:, 1] += eYY

        return trajectory

    # ── core solve ───────────────────────────────────────────

    def _solve_core(self, state, driving_mode, driving_mode_other,
                    obstacle_trajectories, sampled_traj_mean,
                    sampled_traj_log_var_x, sampled_traj_log_var_y,
                    solver, epsilon=0.0):
        x0, theta0, ey0 = self._compute_initial_state(state)

        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        if self.theta is None:
            for k in range(self.N):
                solver.set(k, "x", x0)

        if obstacle_trajectories is None:
            obstacle_trajectories = zeros((self.N + 1, self.nx + 3))

        following_traj = None
        if self.theta is not None and driving_mode in [0, 1]:
            following_traj = self._compute_following_trajectory(
                theta0, ey0, obstacle_trajectories, epsilon
            )

        for k in range(self.N + 1):
            following_traj_k = following_traj[k, :] if following_traj is not None else obstacle_trajectories[k, 8:10]
            params = self._build_solver_params(
                k, theta0, driving_mode, driving_mode_other,
                obstacle_trajectories[k], following_traj_k,
                sampled_traj_mean[k], sampled_traj_log_var_x[k], sampled_traj_log_var_y[k]
            )
            solver.set(k, "p", params)

        kapparefhorizon = array(
            self._kapparef(theta0 + linspace(0, self._ref_horizon_length, self.N + 1))
        ).squeeze()

        status = solver.solve()

        if status != 0:
            solver.set(0, "lbx", x0)
            solver.set(0, "ubx", x0)
            for k in range(self.N):
                solver.set(k, "x", x0)
            for k in range(self.N + 1):
                params = self._build_solver_params(
                    k, theta0, driving_mode, driving_mode_other,
                    obstacle_trajectories[k], obstacle_trajectories[k, 8:10],
                    sampled_traj_mean[k], sampled_traj_log_var_x[k], sampled_traj_log_var_y[k]
                )
                solver.set(k, "p", params)
            solver.reset()
            status = solver.solve()

        trajectory = zeros((self.N + 1, self.nx + 3))
        for k in range(self.N + 1):
            traj_k = solver.get(k, "x")
            trajectory[k, :] = hstack([traj_k, traj_k[:2], kapparefhorizon[k]])

        trajectory = self._convert_to_cartesian(trajectory, state)

        if status == 0:
            theta_next = solver.get(1, "x")[0]
        else:
            theta_next = self.theta if self.theta is not None else theta0

        return status, trajectory, theta_next, ey0, kapparefhorizon

    # ── public API ───────────────────────────────────────────

    def solve(self, state, state_tv, theta_TV, driving_mode, driving_mode_other,
              obstacle_trajectories=None, sampled_traj=None, sampled_traj_mean=None,
              sampled_traj_log_var_x=None, sampled_traj_log_var_y=None):
        sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y = \
            self._init_sampled_traj_defaults(
                sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y,
                use_nonzero_default=True
            )

        x0, _, _ = self._compute_initial_state(state)
        self.__solver_for_decision.set(0, "lbx", x0)
        self.__solver_for_decision.set(0, "ubx", x0)
        if self.theta is None:
            for k in range(self.N):
                self.__solver_for_decision.set(k, "x", x0)

        status, trajectory, theta_next, ey0, kapparefhorizon = self._solve_core(
            state, driving_mode, driving_mode_other,
            obstacle_trajectories, sampled_traj_mean,
            sampled_traj_log_var_x, sampled_traj_log_var_y,
            self.__solver, epsilon=0.0
        )

        self.trajectory      = trajectory
        self.kapparefhorizon = kapparefhorizon
        self.control         = self.__solver.get(0, "u")
        self.ec              = ey0

        if status == 0:
            self.theta = theta_next
            self.ephi  = self.__solver.get(1, "x")[2]

        if self.SIM_STEP % 6 == 0:
            self.__solver.store_iterate(
                self.export_dir + '/mpcc_iterate.json', overwrite=True, verbose=False
            )

        self.SIM_STEP += 1
        return status

    def solve_for_decision(self, state, state_tv, theta_TV, driving_mode, driving_mode_other,
                           obstacle_trajectories=None, sampled_traj=None, sampled_traj_mean=None,
                           sampled_traj_log_var_x=None, sampled_traj_log_var_y=None):
        sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y = \
            self._init_sampled_traj_defaults(
                sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y,
                use_nonzero_default=False
            )

        status, trajectory, theta_next, ey0, _ = self._solve_core(
            state, driving_mode, driving_mode_other,
            obstacle_trajectories, sampled_traj_mean,
            sampled_traj_log_var_x, sampled_traj_log_var_y,
            self.__solver_for_decision, epsilon=1e-6
        )

        self.control = self.__solver_for_decision.get(0, "u")
        self.ec      = ey0

        if status == 0 and self.SIM_STEP % 6 == 0:
            self.__solver_for_decision.store_iterate(
                self.export_dir_dec + '/mpcc_iterate.json', overwrite=True, verbose=False
            )

        self.SIM_STEP += 1
        return trajectory, theta_next

    def reset_theta(self):
        self.theta = None

    def try_load_iterate(self, solver, path):
        if not path.is_file():
            print(f"[INFO] warm-start file not found -> fresh start ({path})")
            solver.reset()
            return False
        try:
            solver.load_iterate(str(path))
            return True
        except Exception as e:
            print(f"[WARN] iterate load failed, resetting solver. Details: {e}")
            solver.reset()
            return False
