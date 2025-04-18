import numpy as np
import hummingbirdParam as P


## Theta = Pitch
## Psi = Yaw
## Phi = Roll

class ctrlPID:
    def __init__(self):
        # tuning parameters
        tr_pitch = 0.6   # rise time for pitch (theta)
        zeta_pitch = .707 # damping ratio for pitch (theta)
        tr_yaw = 2 # rise time for yaw (psi)
        zeta_yaw = .707  # damping ratio for yaw (psi)
        M = 10  # bandwidth separation
        tr_roll = tr_yaw / M  # rise time for roll (phi)
        zeta_roll = 1 # damping ratio for roll (phi)

        # gain calculation
        wn_pitch = 2.2 / tr_pitch  # natural frequency for pitch
        wn_yaw = 2.2 / tr_yaw  # natural frequency for yaw
        wn_roll = 2.2 / tr_roll  # natural frequency for roll
        self.kp_pitch = wn_pitch**2 / P.b_theta  
        self.kd_pitch = 2 * zeta_pitch * wn_pitch / P.b_theta  
        self.kp_roll = P.J1x * wn_roll**2
        self.kd_roll = P.J1x * 2 * zeta_roll * wn_roll
        self.kp_yaw = wn_yaw**2 / P.b_psi  
        self.kd_yaw = 2 * zeta_yaw * wn_yaw / P.b_psi
        self.ki_pitch = 1  # Integrator gain for pitch
        self.ki_roll = 0.01  # Integrator gain for roll
        
        # print gains to terminal
        print('kp_pitch: ', self.kp_pitch)
        print('kd_pitch: ', self.kd_pitch) 
        print('kp_roll: ', self.kp_roll)
        print('kd_roll: ', self.kd_roll) 
        print('kp_yaw: ', self.kp_yaw)
        print('kd_yaw: ', self.kd_yaw) 
        
        # sample rate of the controller
        self.Ts = P.Ts

        # dirty derivative parameters
        sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)

        # delayed variables
        self.phi_d1 = 0.0
        self.phi_dot = 0.0
        self.theta_d1 = 0.0
        self.theta_dot = 0.0
        self.psi_d1 = 0.0
        self.psi_dot = 0.0
        self.error_theta_d1 = 0.0  # pitch error delayed by 1
        self.error_psi_d1 = 0.0  # yaw error delayed by 1

        # Integrators
        self.integrator_theta = 0.0
        self.integrator_phi = 0.0

    def update(self, r, y, disturbance_force):
        theta_ref = r[0][0]
        psi_ref = r[1][0]
        phi = y[0][0]
        theta = y[1][0]
        psi = y[2][0]

        # Equilibrum force to keep the vehicle flying
        force_equilibrium = P.g * (P.m1 * P.ell1 + P.m2 * P.ell2) \
                            * np.cos(theta) / P.ellT
        
        # compute errors
        error_theta = theta_ref - theta
        error_psi = psi_ref - psi

        # Update integrators
        self.integrator_theta += (self.Ts / 2) * (error_theta + self.error_theta_d1)
        self.integrator_phi += (self.Ts / 2) * (error_psi + self.error_psi_d1)

        # Update differentiators
        self.phi_dot = self.beta * self.phi_dot \
                       + (1 - self.beta) * ((phi - self.phi_d1) / self.Ts)
        self.theta_dot = self.beta * self.theta_dot \
                       + (1 - self.beta) * ((theta - self.theta_d1) / self.Ts)
        self.psi_dot = self.beta * self.psi_dot \
                       + (1 - self.beta) * ((psi - self.psi_d1) / self.Ts)

        # Pitch (theta) control
        force_unsat = force_equilibrium \
                      + self.kp_pitch * error_theta \
                      - self.kd_pitch * self.theta_dot \
                      + self.ki_pitch * self.integrator_theta
        force = saturate(force_unsat, -P.force_max, P.force_max) + disturbance_force

        # Outer loop yaw (psi) control
        phi_ref_unsat = self.kp_yaw * error_psi \
                        - self.kd_yaw * self.psi_dot
        phi_ref = saturate(phi_ref_unsat, -np.pi/4, np.pi/4) # limit yaw torque to +- pi/4

        # Inner loop roll (phi) control
        error_phi = phi_ref - phi
        torque_unsat = self.kp_roll * error_phi \
                       - self.kd_roll * self.phi_dot \
                       + self.ki_roll * self.integrator_phi
        torque = saturate(torque_unsat, -P.torque_max, P.torque_max) # limit total torque

        # Convert force and torque to pwm signals
        pwm = np.array([[force + torque / P.d],               # u_left
                      [force - torque / P.d]]) / (2 * P.km)   # r_right          
        pwm = saturate(pwm, 0, 1)
 
        # update all delayed variables
        self.phi_d1 = phi
        self.theta_d1 = theta
        self.psi_d1 = psi
        self.error_theta_d1 = error_theta
        self.error_psi_d1 = error_psi
        # return pwm plus reference signals
        return pwm, np.array([[phi_ref], [theta_ref], [psi_ref]])



def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u




