"""ACADOS OCP setup for Path-Parametric MPC with uncertainty consideration."""

import casadi as ca
from numpy import zeros, array, append
from casadi import if_else

from .acados_ocp_common import (
    create_state_variables,
    create_control_variables,
    create_xdot_variables,
    create_parameters,
    create_track_interpolants,
    create_dynamics,
    create_obstacle_constraints,
    compute_XY_position,
    extract_cost_params,
    apply_curvature_scaling,
    create_base_cost,
    create_base_cost_terminal,
    setup_state_constraints,
    setup_path_constraints,
    setup_slack_costs,
    D_CRIT, K_DO, S_DO, TRACK_BORDER_SCALE,
)
from acados_template import AcadosOcp


# Configuration specific to uncertainty-aware MPC
UNCERTAINTY_WEIGHT = 0.9          # Weight for uncertainty in constraints
UNCERTAINTY_WEIGHT_LEADER = 0.1   # Weight when EV is leading
SAFE_MARGIN_FOLLOWING = 0.6       # Safe margin for following mode
SAFE_MARGIN_LEADING = 0.1         # Safe margin for leading mode


def acados_ocp_pp(vehicle_model, track, config):
    """
    Create ACADOS OCP for Path-Parametric MPC with uncertainty consideration.

    This version includes:
    - Uncertainty-aware obstacle constraints
    - Driving mode dependent costs (following, overtaking, driving, blocking)
    """
    ocp = AcadosOcp()
    ocp.model.name = "Path_parametric_MPC"

    # -------------------------------------------------------------------------
    # State, Control, and Parameter Setup
    # -------------------------------------------------------------------------
    x, (theta, ec, epsi, vx, vy, omega, delta, D) = create_state_variables()
    ocp.model.x = x
    nx = x.size1()

    u, (ddelta, dD) = create_control_variables()
    ocp.model.u = u
    nu = u.size1()

    xdot = create_xdot_variables()
    ocp.model.xdot = xdot

    ocp.model.z = ca.vertcat([])

    p, thetaref, obstacles, driving_mode = create_parameters(config)
    ocp.model.p = p
    ocp.parameter_values = zeros(p.size1())

    # -------------------------------------------------------------------------
    # Track Interpolants
    # -------------------------------------------------------------------------
    Xref, Yref, psiref, kapparef = create_track_interpolants(track)
    theta_max = track.thetaref[-1]

    # -------------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------------
    states = (theta, ec, epsi, vx, vy, omega, delta, D)
    controls = (ddelta, dD)
    f_expl = create_dynamics(vehicle_model, track, states, controls, obstacles,
                             kapparef, Xref, Yref, psiref)
    ocp.model.f_impl_expr = xdot - f_expl
    ocp.model.f_expl_expr = f_expl

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    ocp.model.con_h_expr = ca.vertcat(ec)

    XY = compute_XY_position(theta, ec, Xref, Yref, psiref, theta_max)
    obstacle_dists, is_front = create_obstacle_constraints(theta, XY, obstacles)
    ocp.model.con_h_expr = ca.vertcat(ocp.model.con_h_expr, *obstacle_dists)

    nh = ocp.model.con_h_expr.size1()
    nsh = nh

    # -------------------------------------------------------------------------
    # Cost Function
    # -------------------------------------------------------------------------
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    cost_params = extract_cost_params(p)
    qtheta = cost_params['qtheta']
    qec = cost_params['qec']
    rdelta = cost_params['rdelta']
    rddelta = cost_params['rddelta']
    romega = cost_params['romega']
    sampled_traj_log_var_x = cost_params['sampled_traj_log_var_x']
    sampled_traj_log_var_y = cost_params['sampled_traj_log_var_y']

    # Curvature-based weight scaling
    rdelta, rddelta, romega = apply_curvature_scaling(rdelta, rddelta, romega, kapparef, theta)

    # Base cost
    ocp.model.cost_expr_ext_cost = create_base_cost(
        thetaref, theta, ec, D, delta, dD, ddelta, omega,
        qtheta, qec, rdelta, rddelta, romega, config
    )
    ocp.model.cost_expr_ext_cost_e = create_base_cost_terminal(
        thetaref, theta, ec, D, delta, omega,
        qtheta, qec, rdelta, romega, config
    )

    # Add driving mode dependent costs
    for obs in obstacles:
        thetaecobs = obs[3:]
        XYobs = obs[:2]

        cost = _compute_obstacle_cost(
            XY, XYobs, thetaecobs, thetaref, theta, vx, vy,
            vehicle_model, driving_mode
        )
        ocp.model.cost_expr_ext_cost += cost
        ocp.model.cost_expr_ext_cost_e += cost

    # -------------------------------------------------------------------------
    # Boundary Constraints
    # -------------------------------------------------------------------------
    track_br, track_bl, nsbx = setup_state_constraints(ocp, track, config, vehicle_model, nu)

    # Uncertainty-aware obstacle constraints
    sigma_TV_d = ca.sqrt(sampled_traj_log_var_x ** 2 + sampled_traj_log_var_y ** 2)
    unc_weight = if_else(is_front, UNCERTAINTY_WEIGHT, UNCERTAINTY_WEIGHT_LEADER)
    margin = ca.fabs(sigma_TV_d) * unc_weight

    lh_obstacles = [margin] * config.mpc.num_obstacles
    uh_obstacles = [1e6] * config.mpc.num_obstacles

    setup_path_constraints(ocp, track, track_br, track_bl, lh_obstacles, uh_obstacles, nsh)

    ns = nsh + nsbx
    setup_slack_costs(ocp, ns)

    return ocp


def _compute_obstacle_cost(XY, XYobs, thetaecobs, thetaref, theta, vx, vy,
                           vehicle_model, driving_mode):
    """
    Compute driving mode dependent obstacle cost.

    Driving modes:
        0: Following   - Track opponent with safe distance
        1: Overtaking  - Aggressive maneuver to pass opponent
        2: Driving     - Leader maintaining position
        3: Blocking    - Leader defending position
    """
    # Cost parameters
    qvx = 0.03
    qvx_overtaking = 0.04
    qfollow = 1
    qblock = 1
    qlead = 1.3
    modified_qtheta = 3
    qdraft = 0.001

    # Draft cost (slipstream effect)
    dist_sq = ca.dot(XY - XYobs, XY - XYobs)
    sigma = ca.exp(S_DO * (K_DO - dist_sq)) / (1 + ca.exp(S_DO * (K_DO - dist_sq)))
    draft_cost = (1 - sigma)

    # Following: track opponent with safe distance
    following_cost = (
        qfollow * (thetaecobs[0] - theta - 2.5 * vehicle_model.L) ** 2 +
        qfollow * (thetaecobs[1]) ** 2 +
        qdraft * draft_cost
    )

    # Blocking: stay on racing line while blocking lateral movement
    blocking_cost = (
        1.0 * modified_qtheta * (thetaref - theta) ** 2 +
        qblock * (thetaecobs[1]) ** 2 / (1 + (thetaecobs[0] - theta) ** 2)
    )

    # Overtaking: prioritize progress and speed
    overtaking_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qvx_overtaking * vx ** 2 +
        1.9 * ca.fmax(0, SAFE_MARGIN_FOLLOWING - (theta - thetaecobs[0])) ** 2
    )

    # Driving: maximize progress when leading
    driving_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qlead * ca.fmax(0, theta - thetaecobs[0] - SAFE_MARGIN_LEADING) ** 2 -
        qvx * vx ** 2
    )

    # Select cost based on driving mode (0=following, 1=overtaking, 2=driving, 3=blocking)
    cost = if_else(
        driving_mode == 0, following_cost,
        if_else(
            driving_mode == 1, overtaking_cost,
            if_else(
                driving_mode == 2, driving_cost, blocking_cost
            )
        )
    )
    return cost
