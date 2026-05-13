#!/usr/bin/env python3
'''
Racing environment configuration file
'''
from .utils import Parameters, dataclass
from dataclasses import field
import numpy as np

@dataclass
class Mode_Parameters(Parameters):

    TV_qtheta = 10
    TV_qobs = 10
    TV_blocking_mode = False

    EV_qtheta = 10
    EV_qfollow = 1
    EV_obstacle_mode = True
    EV_following_mode = False

    EV_x0_thetaecephiV = [2,0,0,0.1] 
    TV_x0_thetaecephiV = [2.3,0,0,0.1]
    

@dataclass
class Environment_Parameters(Parameters):
    integrator_type : str = "RK45" 
    dt : float = 0.05
    sequence_length : int = 60
    sim_method_num_steps : int = 3
    map_file : str = "LMS_Track.txt"
    track_filename = "LMS_Track.txt"
    model_dir = "train"
    sim_result_dir = "simulation"

    is_constant_track_border : bool = True
    render_fps : int = 0  
    is_X11_forwarding : bool = True  
    detach_video_export_process : bool = True

    following_mode = Mode_Parameters.EV_following_mode 
    blocking_mode = False

@dataclass
class Environment_Parameters_TV(Environment_Parameters):
    blocking_mode = Mode_Parameters.TV_blocking_mode
    following_mode = False

from .track import Track
@dataclass
class MPC_Parameters(Parameters):
    qblock = 10
    dt : float = 0.05
    N  : int   = 50
    delta_min : float = -3.5e-1 * 1.5 
    delta_max : float =  3.5e-1 * 1.5
    D_min : float = -1e-1
    D_max : float = 1e0
    ddelta_min : float = -15 # minimum change rate of- steering angle [rad/s]
    ddelta_max : float = 15 # maximum change rate of steering angle [rad/s]
    dD_min : float = -15 # minimum throttle change rate
    dD_max : float = 15 # maximum throttle change rate
    
    vx_min : float = 0.001
    vx_max : float = 1.5 # follower: 1.5  leader 1.3

    qtheta = Mode_Parameters.EV_qtheta #10
    qec = 1
    pp_ref_horizon_length : float = 2.0 # 2.0  # Only used in pp_cost_mode=0
    qfollow = Mode_Parameters.EV_qfollow
    romega: float = 1e-3
    rD : float = 1e-4 # 1e-3
    rdelta : float = 5e-3
    rdD : float = 1e-5 # 1e-4 #1e-3
    rddelta : float = 1e-4 #5e-3
    num_obstacles = 1

    track = Track(Environment_Parameters.track_filename)
    thetaecephiV = Mode_Parameters.EV_x0_thetaecephiV
    x0_curv = np.array([*thetaecephiV, 0.0, 0.0, 0.0, 0.0])
    xyref = track(thetaecephiV[0])
    dp = track(thetaecephiV[0], 1)
    psiref = np.arctan2(dp[1], dp[0])
    eX = -np.sin(psiref) * thetaecephiV[1] 
    eY =  np.cos(psiref) * thetaecephiV[1]
    x  = xyref[0] + eX 
    y  = xyref[1] + eY 
    x0 = np.array([[x,y,psiref+thetaecephiV[2]], [thetaecephiV[3], 0.0, 0.0, 0.0, 0.0]])

    S_MAX = track.thetaref[-1]
    
    strategy_weights = {
        0:  {"qtheta": 13, "qec": 0.8, "rdelta": 2e-3, "rddelta": 5e-5, "romega": 1e-3}, # following
        1: {"qtheta": 16, "qec": 0.8, "rdelta": 1e-3, "rddelta": 1e-5, "romega": 8e-4}, # overtaking
        2:    {"qtheta": 16, "qec": 1.0, "rdelta": 2e-3, "rddelta": 5e-5, "romega": 8e-4}, # driving
        3:   {"qtheta": 13,  "qec": 0.8, "rdelta": 1e-3, "rddelta": 5e-5, "romega": 8e-4} # blocking
    }

    D_CRIT_weight = {
        0: 1,
        1: 1,
        2: 1, 
        3: 1

    }
    uh_weight = {
        0: 1e6, 
        1: 1e6,
        2: 1e12,
        3: 1e12
    }

@dataclass
class MPC_Parameters_TV(MPC_Parameters): 
    qtheta = Mode_Parameters.TV_qtheta
    qec = 1
    qblock = Mode_Parameters.TV_qobs
    track = Track(Environment_Parameters.track_filename)
    thetaecephiV = Mode_Parameters.TV_x0_thetaecephiV
    x0_curv = np.array([*thetaecephiV, 0.0, 0.0, 0.0, 0.0])
    xyref = track(thetaecephiV[0])
    dp = track(thetaecephiV[0], 1)
    psiref = np.arctan2(dp[1], dp[0])
    eX = -np.sin(psiref) * thetaecephiV[1] 
    eY =  np.cos(psiref) * thetaecephiV[1]
    x  = xyref[0] + eX 
    y  = xyref[1] + eY 
    x0 = np.array([[x,y,psiref+thetaecephiV[2]], [thetaecephiV[3], 0.0, 0.0, 0.0, 0.0]])

    D_max : float = 1e0 
    vx_max : float = 1.3 # follower: 1.5  leader 1.3


@dataclass
class Vehicle_Parameters(Parameters):
    m   : float = 0.041
    Iz  : float = 2.78e-5
    lf  : float = 0.029
    lr  : float = 0.033
    Cm1 : float = 0.287
    Cm2 : float = 0.0545
    Cr0 : float = 0.0518
    Cr2 : float = 3.5e-4
    Cr3 : float = 5.0
    Br  : float = 3.3852
    Cr  : float = 1.2691
    Dr  : float = 0.1737
    Bf  : float = 2.579
    Cf  : float = 1.2
    Df  : float = 0.192
    L   : float = 0.12
    W   : float = 0.06
    acados_tire_force_model : str = "pacejka"
    acados_tire_force_eps : float = 0.3
    acados_approx_method : str = "conv"
    acados_approx_frame : str = "tire"

    D_CRIT : float = 0.1242 # vehicle width = 0.08m

    mu = 0.9
    mu_wet3 = 0.8
    mu_wet2 = 0.7 
    mu_wet1 = 0.6

    aero_rho : float = 1.2
    aero_S : float = 0.004
    aero_Cd: float = 1.0

    leader_vx_max : float = 1.3
    follower_vx_max : float = 1.5

    # Sample initial positions:
    Leader_thetaecephiV = [2.9,0,0,0.1] 
    Follower_thetaecephiV = [2.5,0,0,0.1]


@dataclass
class Decision_Maker(Parameters):
    collision_th = 0.1342 # vehicle width = 0.08m
    collision_risk_weight = 3.0
    draft_dist = 0.1
    draft_deg = 30
    fuel_weight = -0.02

    overtaking_suc_reward = 5.0
    blocking_suc_reward = 1.5
    
    overtaking_weight = 1.2
    blocking_weight = 0.2

    leader_progress_rate_weight1 = 3.0
    leader_progress_rate_weight2 = 2.0
    follower_progress_weight1 = 1.2
    follower_progress_weight2 = 1.2

    overtaking_suc_weight = 0.5
    


@dataclass
class Acados_Parameters(Parameters):
    export_dir : str = "acados_generated_follower"
    export_dir_dec: str = "acados_generated_follower_decision"
    export_dir_baseline: str= "acados_generated_follower_baseline"
    export_dir_baseline_dec: str= "acados_generated_follower_baseline_dec"
    
    integrator_type : str = "ERK"
    sim_method_num_stages : int = 4
    qp_solver : str = "PARTIAL_CONDENSING_HPIPM"
    qp_solver_warm_start : int = 1
    qp_solver_iter_max   : int = 50
    hpipm_mode : str = "BALANCE"
    hessian_approx  : str = "GAUSS_NEWTON"
    nlp_solver_type : str = "SQP_RTI"
    print_level : int = 0
    globalization : str = "MERIT_BACKTRACKING"
    
    # if IRK
    newton_tol : float = 1e-4
    newton_iter : int = 200
    num_stages : int = 1
    num_steps : int = 1

    sim_method_num_steps  : int = 3
    nlp_solver_max_iter : int = 100
    nlp_solver_step_length : float = 1.0
    levenberg_marquardt: float = 0.1
    tol : float = 1e-4

@dataclass
class Acados_Parameters_TV(Acados_Parameters):
    export_dir : str = "acados_generated_TV"

@dataclass
class Casadi_Parameters(Parameters):
    dt : float = 0.02
    print_time : int = 1
    print_level : int = 2
    max_iter : int = 200
    tol : float = 1e-8
    dual_inf_tol : float = 1.0
    constr_viol_tol : float = 1e-4
    acceptable_tol : float = 1e-6
    acceptable_constr_viol_tol : float = 1e-2
    linear_solver : str = "ma27"
    hessian_approximation : str = "exact"
    warm_start_init_point : str = "yes"
    sim_method_num_steps  : int = 1


@dataclass
class Config(Parameters):
    env : Environment_Parameters = Environment_Parameters()
    vehicle : Vehicle_Parameters = Vehicle_Parameters()
    mpc : MPC_Parameters = MPC_Parameters()
    acados : Acados_Parameters = Acados_Parameters()
    casadi : Casadi_Parameters = Casadi_Parameters()
    mode : Mode_Parameters = Mode_Parameters()
    decision_maker : Decision_Maker = Decision_Maker()

@dataclass
class Config_TV(Parameters):
    env : Environment_Parameters_TV = Environment_Parameters_TV()
    vehicle : Vehicle_Parameters = Vehicle_Parameters()
    mpc : MPC_Parameters_TV = MPC_Parameters_TV()
    acados : Acados_Parameters_TV = Acados_Parameters_TV()
    casadi : Casadi_Parameters = Casadi_Parameters()
    mode : Mode_Parameters = Mode_Parameters()
