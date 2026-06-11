"""ACADOS OCP setup for Time-Optimal MPC.

  MPCC: qtheta * (thetaref - theta)^2  
  TOMPC: -qvx * vx                        

following_cost, blocking_cost, overtaking_cost, driving_cost 
uncertainty-aware constraints are all identical with MPCC version
"""

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
    setup_state_constraints,
    setup_path_constraints,
    setup_slack_costs,
    D_CRIT, K_DO, S_DO, TRACK_BORDER_SCALE,
)
from acados_template import AcadosOcp


# acados_ocp_pp.py와 동일한 상수
UNCERTAINTY_WEIGHT        = 0.9
UNCERTAINTY_WEIGHT_LEADER = 0.1
SAFE_MARGIN_FOLLOWING     = 0.6
SAFE_MARGIN_LEADING       = 0.1

# time-optimal용 속도 최대화 가중치
QVX_TO = 1.0


def acados_ocp_timeoptimal(vehicle_model, track, config):
    """
    Time-Optimal MPC OCP.

    """
    ocp = AcadosOcp()
    ocp.model.name = "TimeOptimal_MPC"

    # -------------------------------------------------------------------------
    # State, Control, and Parameter Setup 
    # -------------------------------------------------------------------------
    x, (theta, ec, epsi, vx, vy, omega, delta, D) = create_state_variables()
    ocp.model.x = x

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
    states   = (theta, ec, epsi, vx, vy, omega, delta, D)
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
    nh  = ocp.model.con_h_expr.size1()
    nsh = nh

    # -------------------------------------------------------------------------
    # Cost Function
    # -------------------------------------------------------------------------
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    cost_params = extract_cost_params(p)
    qec      = cost_params['qec']
    rdelta   = cost_params['rdelta']
    rddelta  = cost_params['rddelta']
    romega   = cost_params['romega']
    sampled_traj_log_var_x = cost_params['sampled_traj_log_var_x']
    sampled_traj_log_var_y = cost_params['sampled_traj_log_var_y']

    # Curvature-based weight scaling  
    rdelta, rddelta, romega = apply_curvature_scaling(rdelta, rddelta, romega, kapparef, theta)

    # ── base cost: qtheta*(thetaref-theta)^2 → -QVX_TO*vx ──────────────────
    base_cost = (
        - QVX_TO * vx +             
        qec      * ec ** 2 +
        config.mpc.rD    * D ** 2 +
        rdelta   * delta ** 2 +
        config.mpc.rdD   * dD ** 2 +
        rddelta  * ddelta ** 2 +
        romega   * omega ** 2
    )
    base_cost_terminal = (
        - QVX_TO * vx +
        qec      * ec ** 2 +
        config.mpc.rD    * D ** 2 +
        rdelta   * delta ** 2 +
        romega   * omega ** 2
    )

    ocp.model.cost_expr_ext_cost   = base_cost
    ocp.model.cost_expr_ext_cost_e = base_cost_terminal

    # ── mode-dependent obstacle cost  ──
    for obs in obstacles:
        thetaecobs = obs[3:]
        XYobs      = obs[:2]
        cost = _compute_obstacle_cost(
            XY, XYobs, thetaecobs, thetaref, theta, vx, vy,
            vehicle_model, driving_mode
        )
        ocp.model.cost_expr_ext_cost   += cost
        ocp.model.cost_expr_ext_cost_e += cost

    # -------------------------------------------------------------------------
    # Boundary Constraints  
    # -------------------------------------------------------------------------
    track_br, track_bl, nsbx = setup_state_constraints(ocp, track, config, vehicle_model, nu)

    sigma_TV_d = ca.sqrt(sampled_traj_log_var_x ** 2 + sampled_traj_log_var_y ** 2)
    unc_weight = if_else(is_front, UNCERTAINTY_WEIGHT, UNCERTAINTY_WEIGHT_LEADER)
    margin = ca.fabs(sigma_TV_d) * unc_weight

    lh_obstacles = [margin] * config.mpc.num_obstacles
    uh_obstacles = [1e6]    * config.mpc.num_obstacles
    setup_path_constraints(ocp, track, track_br, track_bl, lh_obstacles, uh_obstacles, nsh)

    ns = nsh + nsbx
    setup_slack_costs(ocp, ns)

    return ocp


def _compute_obstacle_cost(XY, XYobs, thetaecobs, thetaref, theta, vx, vy,
                           vehicle_model, driving_mode):
    qvx             = 0.03
    qvx_overtaking  = 0.04
    qfollow         = 1
    qblock          = 1
    qlead           = 1.3
    modified_qtheta = 3
    qdraft          = 0.001

    dist_sq    = ca.dot(XY - XYobs, XY - XYobs)
    sigma      = ca.exp(S_DO * (K_DO - dist_sq)) / (1 + ca.exp(S_DO * (K_DO - dist_sq)))
    draft_cost = (1 - sigma)

    following_cost = (
        qfollow * (thetaecobs[0] - theta - 2.5 * vehicle_model.L) ** 2 +
        qfollow * (thetaecobs[1]) ** 2 +
        qdraft  * draft_cost
    )
    blocking_cost = (
        1.0 * modified_qtheta * (thetaref - theta) ** 2 +
        qblock * (thetaecobs[1]) ** 2 / (1 + (thetaecobs[0] - theta) ** 2)
    )
    overtaking_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qvx_overtaking  * vx ** 2 +
        1.9 * ca.fmax(0, SAFE_MARGIN_FOLLOWING - (theta - thetaecobs[0])) ** 2
    )
    driving_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qlead * ca.fmax(0, theta - thetaecobs[0] - SAFE_MARGIN_LEADING) ** 2 -
        qvx   * vx ** 2
    )

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
