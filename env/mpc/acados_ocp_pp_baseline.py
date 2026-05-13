"""ACADOS OCP setup for Path-Parametric MPC baseline (no uncertainty)."""

import casadi as ca
from numpy import zeros, array, append

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
    D_CRIT, TRACK_BORDER_SCALE,
)
from acados_template import AcadosOcp


# Baseline configuration: no uncertainty consideration
D_CRIT_NEW = 0.02             # Fixed critical distance for baseline
SAFE_MARGIN_FOLLOWING = 0.6   # Safe margin for following mode (same as EV)
SAFE_MARGIN_LEADING = 0.1     # Safe margin for leading mode (same as EV)


def acados_ocp_pp_baseline(vehicle_model, track, config):
    """
    Create ACADOS OCP for Path-Parametric MPC baseline.

    This baseline version:
    - Does NOT use uncertainty in constraints (uncertainty_weight = 0)
    - Does NOT add driving mode dependent costs
    - Uses fixed D_CRIT_NEW for obstacle distance constraints
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
    obstacle_dists, _ = create_obstacle_constraints(theta, XY, obstacles)
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

    # Curvature-based weight scaling
    rdelta, rddelta, romega = apply_curvature_scaling(rdelta, rddelta, romega, kapparef, theta)

    # Base cost only (no driving mode dependent costs for baseline)
    ocp.model.cost_expr_ext_cost = create_base_cost(
        thetaref, theta, ec, D, delta, dD, ddelta, omega,
        qtheta, qec, rdelta, rddelta, romega, config
    )
    ocp.model.cost_expr_ext_cost_e = create_base_cost_terminal(
        thetaref, theta, ec, D, delta, omega,
        qtheta, qec, rdelta, romega, config
    )

    # NOTE: Baseline does NOT add obstacle_dists_cost to cost function
    # The driving mode dependent costs are computed but not used

    # -------------------------------------------------------------------------
    # Boundary Constraints
    # -------------------------------------------------------------------------
    track_br, track_bl, nsbx = setup_state_constraints(ocp, track, config, vehicle_model, nu)

    # Baseline: fixed distance constraint (no uncertainty)
    lh_obstacles = [D_CRIT_NEW] * config.mpc.num_obstacles
    uh_obstacles = [1e6] * config.mpc.num_obstacles

    setup_path_constraints(ocp, track, track_br, track_bl, lh_obstacles, uh_obstacles, nsh)

    ns = nsh + nsbx
    setup_slack_costs(ocp, ns)

    return ocp
