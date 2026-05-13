from operator import ge
from os.path import dirname, realpath
from acados_template import AcadosOcpSolver



def acados_solver(ocp, config, custom_export_dir=None):

    # set QP solver and integration
    ocp.dims.N = config.mpc.N
    ocp.solver_options.tf = config.mpc.N * config.mpc.dt
    ocp.solver_options.qp_solver = config.acados.qp_solver
    ocp.solver_options.qp_solver_iter_max = config.acados.qp_solver_iter_max
    ocp.solver_options.qp_solver_warm_start = config.acados.qp_solver_warm_start
    ocp.solver_options.hpipm_mode = config.acados.hpipm_mode
    ocp.solver_options.nlp_solver_type = config.acados.nlp_solver_type
    ocp.solver_options.hessian_approx  = config.acados.hessian_approx
    ocp.solver_options.integrator_type = config.acados.integrator_type
    ocp.solver_options.globalization = config.acados.globalization
    ocp.solver_options.levenberg_marquardt = config.acados.levenberg_marquardt
    ocp.solver_options.sim_method_num_stages = config.acados.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps  = config.acados.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = config.acados.nlp_solver_max_iter
    ocp.solver_options.nlp_solver_step_length = config.acados.nlp_solver_step_length
    ocp.solver_options.tol = config.acados.tol
    ocp.solver_options.print_level = config.acados.print_level
    ocp.solver_options.newton_tol =     config.acados.newton_tol
    ocp.solver_options.newton_iter =    config.acados.newton_iter
    ocp.solver_options.num_stages =     config.acados.num_stages
    ocp.solver_options.num_steps =      config.acados.num_steps

    # export_dir = dirname(realpath(__file__))+"/"+config.acados.export_dir
    # Use custom export dir if provided, otherwise use default
    export_dir_name = custom_export_dir if custom_export_dir else config.acados.export_dir
    export_dir = dirname(realpath(__file__))+"/"+export_dir_name

    ocp.code_export_directory = export_dir+"/c_generated_code"
    solver = AcadosOcpSolver(ocp, json_file=export_dir+"/acados_ocp.json")

    return solver


def hpipm_solver(ocp, config):

    # set QP solver and integration
    ocp.dims.N = config.mpc.N
    ocp.solver_options.tf = config.mpc.N * config.mpc.dt
    ocp.solver_options.qp_solver = config.hpipm.qp_solver
    ocp.solver_options.qp_solver_iter_max = config.hpipm.qp_solver_iter_max
    ocp.solver_options.qp_solver_warm_start = config.hpipm.qp_solver_warm_start
    ocp.solver_options.hpipm_mode = config.hpipm.hpipm_mode
    ocp.solver_options.nlp_solver_type = config.hpipm.nlp_solver_type
    ocp.solver_options.hessian_approx  = config.hpipm.hessian_approx
    ocp.solver_options.integrator_type = config.hpipm.integrator_type
    ocp.solver_options.globalization = config.hpipm.globalization
    ocp.solver_options.levenberg_marquardt = config.hpipm.levenberg_marquardt
    ocp.solver_options.sim_method_num_stages = config.hpipm.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps  = config.hpipm.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = config.hpipm.nlp_solver_max_iter
    ocp.solver_options.nlp_solver_step_length = config.hpipm.nlp_solver_step_length
    ocp.solver_options.tol = config.hpipm.tol
    ocp.solver_options.print_level = config.hpipm.print_level

    export_dir = dirname(realpath(__file__))+"/"+config.hpipm.export_dir
    ocp.code_export_directory = export_dir+"/c_generated_code"
    solver = AcadosOcpSolver(ocp, json_file=export_dir+"/acados_ocp.json")

    return solver