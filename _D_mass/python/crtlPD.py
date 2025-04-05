import numpy as np
import massParam as P


class ctrlPD:

    def __init__(self):
        #  tuning parameters
        tr = 2.0
        zeta = 0.7
        
        # compute PD gains
        # open loop char polynomial and poles
        a1 = P.b/P.m
        a0 = P.k/P.m
        wn = 2.2/tr
        alpha1 = 2.0 * zeta * wn
        alpha0 = wn**2
        self.kp = P.m * (alpha0 - a0)
        self.kd = P.m * (alpha1 - a1)
        self.ki = 1
        self.derivativeCalculator = DirtyDerivative(Ts=P.Ts)
        self.integrator = 0.0
        self.error_dot = 0.0
        self.error_d1 = 0.0
        self.z_d1 = 0.0
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, z_r, state):
        z = state[0][0]
        zdot = self.derivativeCalculator.compute(z)
        print("zdot: " + str(zdot))
        self.integrator = self.integrator + (P.Ts / 2) * ((z_r - z) + self.error_d1)
        print("self.integrator: " + str(self.integrator))
        tau_tilde = self.kp * (z_r - z) - zdot * self.kd + self.ki * self.integrator
        tau = saturate(tau_tilde, P.F_max)

        if self.ki != 0.0:
            self.integrator = self.integrator + P.Ts / self.ki * (tau - tau_tilde)
        self.error_d1 = z_r - z
        self.z_d1 = z
        return tau

class DirtyDerivative:
    def __init__(self, Ts, sigma=0.05):
        self.sigma = sigma
        self.Ts = Ts
        self.beta = (2.0 * self.sigma - self.Ts) / (2.0 * self.sigma + self.Ts)
        self.prev_input = 0.0
        self.derivative = 0.0

    def compute(self, current_input):
        self.derivative = self.beta * self.derivative + (1 - self.beta) * ((current_input - self.prev_input) / self.Ts)
        self.prev_input = current_input
        return self.derivative

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

