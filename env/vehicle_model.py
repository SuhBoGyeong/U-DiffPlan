import casadi as ca
import numpy as np
from scipy.special import erf

def smooth_abs_conv(x, eps):
    return x * ca.erf(x / ca.sqrt(2.0) / eps) + 2.0 * eps * ca.exp(-0.5 * (x / eps) ** 2) / ca.sqrt(2.0 * np.pi)

def smooth_abs_sqrt(x, eps):
    return ca.sqrt(x**2 + eps**2)

class VehicleModel:

    def __init__(self, config):

        self.m = config.vehicle.m
        self.Iz = config.vehicle.Iz
        self.lf = config.vehicle.lf
        self.lr = config.vehicle.lr

        self.Br = config.vehicle.Br
        self.Cr = config.vehicle.Cr
        self.Dr = config.vehicle.Dr
        self.Bf = config.vehicle.Bf
        self.Cf = config.vehicle.Cf
        self.Df = config.vehicle.Df

        self.W = config.vehicle.W
        self.L = config.vehicle.L

        self.Cm1 = config.vehicle.Cm1
        self.Cm2 = config.vehicle.Cm2
        self.Cr0=config.vehicle.Cr0
        self.Cr2=config.vehicle.Cr2
        self.Cr3=config.vehicle.Cr3

        self.mu = config.vehicle.mu

        # aero dynamics 
        self.aero_rho = config.vehicle.aero_rho
        self.aero_S = config.vehicle.aero_S
        self.aero_Cd = config.vehicle.aero_Cd
            
        self.acados_tire_force_model = config.vehicle.acados_tire_force_model
        self.acados_tire_force_eps   = config.vehicle.acados_tire_force_eps
        self.acados_approx_method    = config.vehicle.acados_approx_method
        self.acados_approx_frame     = config.vehicle.acados_approx_frame

        # dynamic slip condition
        self.wet_road = False #intialize 
        self.mu_wet = None

        # tire wear 
        self.wear_level = 0.0
        self.alpha_wear = 0.2
        self.delta_wear =0.005

    def set_wet_road(self, is_wet, mu_wet):
        '''dynamically update wet road condition'''
        self.wet_road = is_wet 
        self.mu_wet = mu_wet

    def get_current_mu(self):
        '''Return the current friction coefficient based on wet/dry condition.'''
        return self.mu_wet if self.wet_road else self.mu 

    def get_dragforce(self, vx,theta,thetaobs, X, Y, Psi, Xobs, Yobs, Psiobs):

        vmax = 1.5
        kc = 0.805; kx = 0.003; ky = 0.0825

        dist = ca.sqrt(((Xobs-X)**2 + (Yobs-Y)**2))

        # draft force 
        condition = thetaobs > theta
        beta_cd_then = ca.fmin(kc + kx * dist * ca.cos(ca.fabs(Psiobs - Psi)) + ky * dist * ca.sin(ca.fabs(Psiobs - Psi)), 1)

        alpha_cd_then = 1 + (vx / vmax) * (beta_cd_then - 1)
        
        alpha_cd_else = 1
        alpha_cd = ca.if_else(condition, alpha_cd_then, alpha_cd_else)


        Fd = 1/2*vx**2*self.aero_rho*self.aero_Cd*self.aero_S * alpha_cd
        return Fd, alpha_cd

    def get_tire_slip_angle(self, vx, vy, omega, delta):
        alphaf = -ca.atan2(self.lf * omega + vy, vx) + delta
        alphar =  ca.atan2(self.lr * omega - vy, vx)
        if self.wet_road:
            slip_perturbation = 0.05 * ca.sin(omega * 2 * np.pi)
            alphaf += slip_perturbation 
            alphar += slip_perturbation

        return alphaf, alphar

    def get_exact_tire_force(self, vx, vy, omega, delta, D):

        current_mu = self.get_current_mu()

        alphaf, alphar = self.get_tire_slip_angle(ca.fabs(vx), vy, omega, delta)
        Ffy = current_mu * self.Df * ca.sin(self.Cf * ca.atan(self.Bf * alphaf))
        Fry = current_mu * self.Dr * ca.sin(self.Cr * ca.atan(self.Br * alphar))

        Frx = (self.Cm1 - self.Cm2 * ca.fabs(vx)) * D - self.Cr0 * ca.tanh(self.Cr3 * vx) - self.Cr2 * vx**2 * ca.sign(vx)
        return Ffy, Fry, Frx

    def get_pacejka_tire_force_R(self, vx, vy, omega, delta, D):
        cdelta = ca.cos(delta)
        sdelta = ca.sin(delta)

        vxf = -cdelta * vx - sdelta * (self.lf * omega + vy)
        vyf =  sdelta * vx - cdelta * (self.lf * omega + vy)
        vyr = self.lr * omega - vy

        wear_factor = 1 - self.alpha_wear * self.wear_level 
        wear_factor_B = 1 + 0.5 * self.alpha_wear * self.wear_level 
        current_mu = self.get_current_mu() * wear_factor

        Df_worn = self.Df * wear_factor  # Reduce max lateral force
        Dr_worn = self.Dr * wear_factor
        Bf_worn = self.Bf * wear_factor_B  # Increase slip sensitivity
        Br_worn = self.Br * wear_factor_B

        if self.acados_approx_method=="conv":
            alphaf = ca.atan2(vyf, smooth_abs_conv(vxf, self.acados_tire_force_eps))
            alphar = ca.atan2(vyr, smooth_abs_conv(vx, self.acados_tire_force_eps))
        elif self.acados_approx_method=="sqrt":
            alphaf = ca.atan2(vyf, smooth_abs_sqrt(vxf, self.acados_tire_force_eps))
            alphar = ca.atan2(vyr, smooth_abs_sqrt(vx, self.acados_tire_force_eps))
        elif self.acados_approx_method=="max_abs":
            alphaf = ca.atan2(vyf, ca.fmax(ca.fabs(vxf), self.acados_tire_force_eps))
            alphar = ca.atan2(vyr, ca.fmax(ca.fabs(vx), self.acados_tire_force_eps))
        else:
            alphaf = ca.atan2(vyf, ca.fmax(vxf, self.acados_tire_force_eps))
            alphar = ca.atan2(vyr, ca.fmax(vx, self.acados_tire_force_eps))

        Ffy = Df_worn * ca.sin(self.Cf * ca.atan(Bf_worn * alphaf))
        Fry = Dr_worn * ca.sin(self.Cr * ca.atan(Br_worn * alphar))

        # Adjust rolling resistance based on current friction coefficient
        Cr0_eff = self.Cr0 * (self.mu/current_mu) * (1 + 0.2 * self.wear_level)  # Adjust rolling resistance coefficient based on wear
        Frx = (self.Cm1 - self.Cm2 * vx) * D - Cr0_eff * ca.tanh(self.Cr3 * vx) - self.Cr2 * vx**2 
        return Ffy, Fry, Frx

    def get_linear_tire_force_R(self, vx, vy, omega, delta, D):

        cdelta = ca.cos(delta)
        sdelta = ca.sin(delta)

        vxf = -cdelta * vx - sdelta * (self.lf * omega + vy)
        vyf =  sdelta * vx - cdelta * (self.lf * omega + vy)
        vyr = self.lr * omega - vy

        if self.acados_approx_method=="conv":
            alphaf = vyf / smooth_abs_conv(vxf, self.acados_tire_force_eps)
            alphar = vyr / smooth_abs_conv(vx, self.acados_tire_force_eps)
        elif self.acados_approx_method=="sqrt":
            alphaf = vyf / smooth_abs_sqrt(vxf, self.acados_tire_force_eps)
            alphar = vyr / smooth_abs_sqrt(vx, self.acados_tire_force_eps)
        elif self.acados_approx_method=="max_abs":
            alphaf = vyf / ca.fmax(ca.fabs(vxf), self.acados_tire_force_eps)
            alphar = vyr / ca.fmax(ca.fabs(vx), self.acados_tire_force_eps)
        else:
            alphaf = vyf / ca.fmax(vxf, self.acados_tire_force_eps)
            alphar = vyr / ca.fmax(vx, self.acados_tire_force_eps)

        Ffy = 0.5942016 * alphaf
        Fry = 0.7462425 * alphar
        Frx = (self.Cm1 - self.Cm2 * vx) * D - self.Cr0 * ca.tanh(self.Cr3 * vx) - self.Cr2 * vx**2
        return Ffy, Fry, Frx


    def f(self, X, Y, psi, vx, vy, omega, delta, D, ddelta, dD, theta= None, thetaobs= None, Xobs=None,Yobs=None,Psiobs=None):
        '''
        Contunuons-time dynamics model.
        Arguments
        ---------
        X: X-axis position of the Vehicle on the global coordinates.
        Y: Y-axis position of the Vehicle on the global coordinates.
        psi: Heading angle of the Vehicle on the global coordinates.
        vx: Longitudinal velocity of the vehicle.
        vy: Lateral velocity of the vehicle.
        omega: Turning rate of the vehicle.
        delta: Steering angle.
        D: Duty-cycle of the PWM signal of the DC motor.
        ddelta: Steering angle change rate.
        dD: Duty-cycle change rate.
        '''

        Fd,_ = self.get_dragforce(vx,theta,thetaobs, X, Y, psi, Xobs, Yobs, Psiobs)

        cos_delta = ca.cos(delta)
        sin_delta = ca.sin(delta)

        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)

        dx = vx * cpsi - vy * spsi
        dy = vx * spsi + vy * cpsi
        dpsi =  omega

        Ffy, Fry, Frx = self.get_exact_tire_force(vx, vy, omega, delta, D)

        Frx -= Fd 
        dvx = (Frx - Ffy * sin_delta) / self.m + vy * omega
        dvy = (Fry + Ffy * cos_delta) / self.m - vx * omega
        
        domega = (Ffy * self.lf * cos_delta - Fry * self.lr) / self.Iz
    
        return dx, dy, dpsi, dvx, dvy, domega, ddelta, dD

    def f_pp(self, theta, ec, epsi, vx, vy, omega, delta, D, ddelta, dD, kapparef,Xref=None,Yref=None,Psiref=None, Xobs=None, Yobs=None, Psiobs=None,thetaobs=None):
        '''
        Contunuons-time dynamics model for Path parametric MPC.
        Arguments
        ---------
        theta: Length parameter of the reference path.
        ey: Lateral position error of the Vehicle to the reference path.
        epsi: Heading angle error of the Vehicle to the reference path.
        vx: Longitudinal velocity of the vehicle.
        vy: Lateral velocity of the vehicle.
        omega: Turning rate of the vehicle.
        delta: Steering angle.
        D: Duty-cycle of the PWM signal of the DC motor.
        ddelta: Steering angle change rate.
        dD: Duty-cycle change rate.
        kapparef: Cubic spline interpolation of the reference path curvature (callable).
        '''

        X = Xref(theta) - ca.sin(Psiref(theta))*ec
        Y = Yref(theta) + ca.cos(Psiref(theta))* ec
        Psi = Psiref(theta) + epsi
        Fd,_ = self.get_dragforce(vx,theta,thetaobs, X, Y, Psi, Xobs, Yobs, Psiobs)

        if self.acados_tire_force_model=="linear":
            if self.acados_approx_frame=="tire":
                Ffy, Fry, Frx = self.get_linear_tire_force(vx, vy, omega, delta, D)
            else:
                Ffy, Fry, Frx = self.get_linear_tire_force_R(vx, vy, omega, delta, D)
        else:
            if self.acados_approx_frame=="tire":
                Ffy, Fry, Frx = self.get_pacejka_tire_force_R(vx, vy, omega, delta, D)
            else:
                Ffy, Fry, Frx = self.get_pacejka_tire_force(vx, vy, omega, delta, D)

        cos_epsi = ca.cos(epsi)
        sin_epsi = ca.sin(epsi)
        cos_delta = ca.cos(delta)
        kappa = kapparef(theta)

        Frx -= Fd

        dtheta = (vx * cos_epsi - vy * sin_epsi) / (1 - kappa * ec)
        dec    = vx * sin_epsi + vy * cos_epsi
        depsi  = omega - kappa * dtheta
        dvx    = (Frx - Ffy * ca.sin(delta)) / self.m + vy * omega
        dvy    = (Fry + Ffy * cos_delta) / self.m - vx * omega
        domega = (Ffy * self.lf * cos_delta - Fry * self.lr) / self.Iz

        
        return dtheta, dec, depsi, dvx, dvy, domega, ddelta, dD

    def footprint(self, x=0.0, y=0.0, psi=0.0):
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        vertices = np.array([
            [cpsi, -spsi],
            [spsi,  cpsi]
        ]) @ np.array([
            [self.L, self.L, -self.L, -self.L],
            [self.W, -self.W, -self.W, self.W],
        ]) / 2 + np.array([[x], [y]])
        return vertices[0, :], vertices[1, :]

    @property
    def safe_distance(self):
        return (self.W/2) 

