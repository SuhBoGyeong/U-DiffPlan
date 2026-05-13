"""Common utilities for ACADOS OCP setup."""

import casadi as ca
from numpy import array, zeros, ones, hstack, append
from acados_template import AcadosOcp
from casadi import if_else


# ============================================================================
# Constants
# ============================================================================
D_CRIT = 0.009                # Critical distance for obstacle avoidance
D_CRIT_NEW = 0.02             # New critical distance threshold
TRACK_BORDER_SCALE = 1.7      # Scale factor for track border safety
VEHICLE_W = 0.06              # Vehicle width [m]
VEHICLE_L = 0.12              # Vehicle length [m]
SLACK_COST = 1e3              # Slack variable cost weight

# Cost function constants
K_DO = 1                      # Draft cost sigmoid parameter K
S_DO = 1                      # Draft cost sigmoid parameter S

# Parameter indices (from end of parameter vector)
PARAM_IDX_QTHETA = -10
PARAM_IDX_QEC = -9
PARAM_IDX_RDELTA = -8
PARAM_IDX_RDDELTA = -7
PARAM_IDX_ROMEGA = -6
PARAM_IDX_SAMPLED_MEAN = -5
PARAM_IDX_LOG_VAR_X = -4
PARAM_IDX_LOG_VAR_Y = -3
PARAM_IDX_D_CRIT_WEIGHT = -2
PARAM_IDX_UH_WEIGHT = -1


# ============================================================================
# State and Control Setup
# ============================================================================
def create_state_variables():
    """Create state variables for the vehicle model."""
    theta = ca.MX.sym("theta")
    ec = ca.MX.sym("ec")
    epsi = ca.MX.sym("epsi")
    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    omega = ca.MX.sym("omega")
    delta = ca.MX.sym("delta")
    D = ca.MX.sym("D")
    x = ca.vertcat(theta, ec, epsi, vx, vy, omega, delta, D)
    return x, (theta, ec, epsi, vx, vy, omega, delta, D)


def create_control_variables():
    """Create control variables."""
    ddelta = ca.MX.sym("ddelta")
    dD = ca.MX.sym("dD")
    u = ca.vertcat(ddelta, dD)
    return u, (ddelta, dD)


def create_xdot_variables():
    """Create state derivative variables."""
    thetadot = ca.MX.sym("thetadot")
    ecdot = ca.MX.sym("ecdot")
    epsidot = ca.MX.sym("epsidot")
    vxdot = ca.MX.sym("vxdot")
    vydot = ca.MX.sym("vydot")
    omegadot = ca.MX.sym("omegadot")
    deltadot = ca.MX.sym("deltadot")
    Ddot = ca.MX.sym("Ddot")
    xdot = ca.vertcat(thetadot, ecdot, epsidot, vxdot, vydot, omegadot, deltadot, Ddot)
    return xdot


def create_parameters(config):
    """Create OCP parameters."""
    thetaref = ca.MX.sym("thetaref")

    obstacles = []
    for idx in range(config.mpc.num_obstacles):
        obs = ca.vertcat(
            ca.MX.sym(f"Xobs{idx}"),
            ca.MX.sym(f"Yobs{idx}"),
            ca.MX.sym(f"Psiobs{idx}"),
            ca.MX.sym(f"Thetaobs{idx}"),
            ca.MX.sym(f"Ecobs{idx}")
        )
        obstacles.append(obs)

    driving_mode = ca.MX.sym('driving_mode')
    driving_mode_other = ca.MX.sym('driving_mode_other')

    # Cost function weight parameters
    qtheta = ca.MX.sym('qtheta')
    qec = ca.MX.sym('qec')
    rdelta = ca.MX.sym('rdelta')
    rddelta = ca.MX.sym('rddelta')
    romega = ca.MX.sym('romega')
    sampled_traj_mean = ca.MX.sym('sampled_traj_mean')
    sampled_traj_log_var_x = ca.MX.sym('sampled_traj_log_var_x')
    sampled_traj_log_var_y = ca.MX.sym('sampled_traj_log_var_y')
    D_CRIT_weight = ca.MX.sym('D_CRIT_weight')
    uh_weight = ca.MX.sym('uh_weight')

    p = ca.vertcat(
        thetaref, *obstacles, driving_mode, driving_mode_other,
        qtheta, qec, rdelta, rddelta, romega,
        sampled_traj_mean, sampled_traj_log_var_x, sampled_traj_log_var_y,
        D_CRIT_weight, uh_weight
    )

    return p, thetaref, obstacles, driving_mode


def create_track_interpolants(track):
    """Create track reference interpolants."""
    Xref = ca.interpolant("xref_s", "bspline", [track.thetaref], track.Xref)
    Yref = ca.interpolant("yref_s", "bspline", [track.thetaref], track.Yref)
    psiref = ca.interpolant("psiref_s", "bspline", [track.thetaref], track.psiref)
    kapparef = ca.interpolant("kapparef_s", "bspline", [track.thetaref], track.kapparef)
    return Xref, Yref, psiref, kapparef


# ============================================================================
# Dynamics Setup
# ============================================================================
def create_dynamics(vehicle_model, track, states, controls, obstacles, kapparef, Xref, Yref, psiref):
    """Create vehicle dynamics expressions."""
    theta, ec, epsi, vx, vy, omega, delta, D = states
    ddelta, dD = controls
    theta_max = track.thetaref[-1]

    f_expl = ca.vertcat(
        *vehicle_model.f_pp(
            theta, ec, epsi, vx, vy, omega,
            delta, D, ddelta, dD,
            lambda thetak: kapparef(ca.fmod(thetak, theta_max)),
            lambda thetak: Xref(ca.fmod(thetak, theta_max)),
            lambda thetak: Yref(ca.fmod(thetak, theta_max)),
            lambda thetak: psiref(ca.fmod(thetak, theta_max)),
            obstacles[0][0], obstacles[0][1], obstacles[0][2], obstacles[0][3]
        )
    )
    return f_expl


# ============================================================================
# Constraint Setup
# ============================================================================
def create_obstacle_constraints(theta, XY, obstacles):
    """Create obstacle distance constraints."""
    obstacle_dists = []
    is_front = None

    for obs in obstacles:
        XYobs = obs[:2]
        thetaobs = obs[3]
        long_sep = thetaobs - theta
        is_front = long_sep > 0.01

        h_dist = ca.dot(XY - XYobs, XY - XYobs) - D_CRIT
        obstacle_dists.append(h_dist)

    return obstacle_dists, is_front


def compute_XY_position(theta, ec, Xref, Yref, psiref, theta_max):
    """Compute XY position from curvilinear coordinates."""
    theta_mod = ca.fmod(theta, theta_max)
    XY = ca.vertcat(
        Xref(theta_mod) - ca.sin(psiref(theta_mod)) * ec,
        Yref(theta_mod) + ca.cos(psiref(theta_mod)) * ec
    )
    return XY


# ============================================================================
# Cost Function Setup
# ============================================================================
def extract_cost_params(p):
    """Extract cost parameters from parameter vector."""
    return {
        'qtheta': p[PARAM_IDX_QTHETA],
        'qec': p[PARAM_IDX_QEC],
        'rdelta': p[PARAM_IDX_RDELTA],
        'rddelta': p[PARAM_IDX_RDDELTA],
        'romega': p[PARAM_IDX_ROMEGA],
        'sampled_traj_mean': p[PARAM_IDX_SAMPLED_MEAN],
        'sampled_traj_log_var_x': p[PARAM_IDX_LOG_VAR_X],
        'sampled_traj_log_var_y': p[PARAM_IDX_LOG_VAR_Y],
        'D_CRIT_weight': p[PARAM_IDX_D_CRIT_WEIGHT],
        'uh_weight': p[PARAM_IDX_UH_WEIGHT]
    }


def apply_curvature_scaling(rdelta, rddelta, romega, kapparef, theta):
    """Apply curvature-based weight scaling."""
    kappa_abs = ca.fabs(kapparef(theta))
    gain = 1 + 3 * kappa_abs
    return rdelta / gain, rddelta / gain, romega / gain


def create_base_cost(thetaref, theta, ec, D, delta, dD, ddelta, omega,
                     qtheta, qec, rdelta, rddelta, romega, config):
    """Create base cost expression."""
    cost = (
        qtheta * (thetaref - theta) ** 2 +
        qec * ec ** 2 +
        config.mpc.rD * D ** 2 +
        rdelta * delta ** 2 +
        config.mpc.rdD * dD ** 2 +
        rddelta * ddelta ** 2 +
        romega * omega ** 2
    )
    return cost


def create_base_cost_terminal(thetaref, theta, ec, D, delta, omega,
                               qtheta, qec, rdelta, romega, config):
    """Create terminal cost expression."""
    cost = (
        qtheta * (thetaref - theta) ** 2 +
        qec * ec ** 2 +
        config.mpc.rD * D ** 2 +
        rdelta * delta ** 2 +
        romega * omega ** 2
    )
    return cost


def create_driving_mode_cost(XY, XYobs, thetaecobs, thetaref, theta, vx, vy,
                              vehicle_model, driving_mode, cost_params):
    """Create driving mode dependent cost."""
    qvx = cost_params.get('qvx', 0.03)
    qvx_overtaking = cost_params.get('qvx_overtaking', 0.04)
    qfollow = cost_params.get('qfollow', 1)
    qblock = cost_params.get('qblock', 1)
    qlead = cost_params.get('qlead', 1.3)
    modified_qtheta = cost_params.get('modified_qtheta', 3)
    safe_margin_f = cost_params.get('safe_margin_f', 0.6)
    safe_margin_l = cost_params.get('safe_margin_l', 0.1)
    qdraft = cost_params.get('qdraft', 0.001)

    # Draft cost (slipstream effect)
    sigma = ca.exp(S_DO * (K_DO - ca.dot(XY - XYobs, XY - XYobs))) / \
            (1 + ca.exp(S_DO * (K_DO - ca.dot(XY - XYobs, XY - XYobs))))
    draft_cost = (1 - sigma)

    # Following cost
    following_cost = (
        qfollow * (thetaecobs[0] - theta - 2.5 * vehicle_model.L) ** 2 +
        qfollow * (thetaecobs[1] - thetaecobs[1]) ** 2 +  # ec tracking
        qdraft * draft_cost
    )
    # Fix: use ec from thetaecobs
    following_cost = (
        qfollow * (thetaecobs[0] - theta - 2.5 * vehicle_model.L) ** 2 +
        qfollow * (thetaecobs[1]) ** 2 +
        qdraft * draft_cost
    )

    # Blocking cost
    blocking_cost = (
        1.0 * modified_qtheta * (thetaref - theta) ** 2 +
        qblock * (thetaecobs[1]) ** 2 / (1 + (thetaecobs[0] - theta) ** 2)
    )

    # Overtaking cost
    overtaking_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qvx_overtaking * vx ** 2 +
        1.9 * ca.fmax(0, safe_margin_f - (theta - thetaecobs[0])) ** 2
    )

    # Driving cost
    driving_cost = (
        modified_qtheta * (thetaref - theta) ** 2 -
        qlead * ca.fmax(0, theta - thetaecobs[0] - safe_margin_l) ** 2 -
        qvx * vx ** 2
    )

    # Select cost based on driving mode
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


# ============================================================================
# Boundary Constraints Setup
# ============================================================================
def setup_state_constraints(ocp, track, config, vehicle_model, nu):
    """Setup state and control constraints."""
    track_br = TRACK_BORDER_SCALE * vehicle_model.safe_distance
    track_bl = TRACK_BORDER_SCALE * vehicle_model.safe_distance

    ocp.constraints.x0 = hstack(config.mpc.x0_curv)
    ocp.constraints.lbx = array([
        -track.border_right[0] + track_br,
        config.mpc.vx_min,
        config.mpc.delta_min,
        config.mpc.D_min
    ])
    ocp.constraints.ubx = array([
        track.border_left[0] - track_bl,
        config.mpc.vx_max,
        config.mpc.delta_max,
        config.mpc.D_max
    ])
    ocp.constraints.idxbx = array([1, 3, 6, 7])
    nsbx = ocp.constraints.idxbx.shape[0]

    ocp.constraints.lbu = array([config.mpc.ddelta_min, config.mpc.dD_min])
    ocp.constraints.ubu = array([config.mpc.ddelta_max, config.mpc.dD_max])
    ocp.constraints.idxbu = array(range(nu))

    ocp.constraints.lsbx = zeros([nsbx])
    ocp.constraints.usbx = zeros([nsbx])
    ocp.constraints.idxsbx = array(range(nsbx))

    return track_br, track_bl, nsbx


def setup_path_constraints(ocp, track, track_br, track_bl, lh_obstacles, uh_obstacles, nsh):
    """Setup path constraints."""
    ocp.constraints.lh = array([-track.border_right[0] + track_br])
    ocp.constraints.uh = array([track.border_left[0] - track_bl])

    ocp.constraints.lh = append(ocp.constraints.lh, lh_obstacles)
    ocp.constraints.uh = append(ocp.constraints.uh, uh_obstacles)

    ocp.constraints.lsh = zeros(nsh)
    ocp.constraints.ush = zeros(nsh)
    ocp.constraints.idxsh = array(range(nsh))


def setup_slack_costs(ocp, ns):
    """Setup slack variable costs."""
    ocp.cost.zl = SLACK_COST * ones((ns,))
    ocp.cost.zu = SLACK_COST * ones((ns,))
    ocp.cost.Zl = SLACK_COST * ones((ns,))
    ocp.cost.Zu = SLACK_COST * ones((ns,))
