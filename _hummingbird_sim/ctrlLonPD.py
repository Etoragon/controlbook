import numpy as np
import hummingbirdParam as P

class ctrlLonPD:
    def __init__(self):
        # Tuning parameters
        tr_pitch = 0.1  # Desired rise time
        self.theta_d2 = 0.0
        self.theta_dot_d2 = 0.0
        self.theta_dot_d3 = 0.0
        zeta_pitch = 0.707  # Damping ratio
        
        # Compute system parameter
        b_theta = P.ellT / (P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)

        
        # Compute natural frequency
        wn_pitch = 2.2 / tr_pitch  # wn = 2.2 / tr
        
        # Compute PD gains
        self.kp_pitch = (wn_pitch**2) / b_theta / 2
        self.kd_pitch = -1 #* (2 * zeta_pitch * wn_pitch) / b_theta
        
        # Print computed gains
        print('kp_pitch:', self.kp_pitch)
        print('kd_pitch:', self.kd_pitch)
        
        # Sample rate
        self.Ts = P.Ts
        
        # Dirty derivative parameters
        sigma = 0.05  # Cutoff frequency for dirty derivative
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)
        
        self.theta_d1 = 0.
        self.theta_dot = 0.

    def update(self, r: np.ndarray, y: np.ndarray):
        theta_ref = r[0][0]  # Desired theta\
        theta = y[1][0] # Measured theta

        theta_d1 = theta
        # y_{k-1} is the current reading from sensors
        theta_dot_d1 = (theta_d1 - self.theta_d2) / self.Ts
        # numerical derivative using most recent samples
        theta_dot = 3 * theta_dot_d1 - 3 * self.theta_dot_d2 + self.theta_dot_d3
        # quadratic predictor
        # update all delayed variables: Note the order!!!
        self.theta_d2 = theta_d1
        self.theta_dot_d3 = self.theta_dot_d2
        self.theta_dot_d2 = theta_dot_d1

        # Compute error
        error_theta = theta_ref - theta

        # Gravity Compensation
        F_fl = (P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT * np.cos(theta)

        # PD Control Law with Gravity Compensation
        force_unsat = self.kp_pitch * error_theta + self.kd_pitch * theta_dot + F_fl

        # Apply saturation
        force = saturate(force_unsat, -P.force_max, P.force_max)

        # Convert force to PWM signals
        pwm = np.array([[force], [force]]) / (2 * P.km)
        pwm = saturate(pwm, 0, 1)

        # Update delayed variables
        self.theta_d1 = theta

        return pwm, np.array([[0.], [theta_ref], [0.]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float):
        return max(min(u, up_limit), low_limit)
    else:
        return np.clip(u, low_limit, up_limit)
