import numpy as np
import rodMassParam as P


class ctrlPD:
    def __init__(self):
        self.kp = 3.0005
        self.kd = 0.094425
        self.derivativeCalculator = DirtyDerivative(Ts=P.Ts)
        self.error_dot = 0.0
        self.error_d1 = 0.0
        self.z_d1 = 0.0
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, theta_r, state):
        theta = state
        thetadot = self.derivativeCalculator.compute(theta)
        print("zdot: " + str(thetadot))

        tau_tilde = self.kp * (theta_r - theta) - thetadot * self.kd
        tau = saturate(tau_tilde, P.tau_max)

        self.error_d1 = theta_r - theta
        self.z_d1 = theta
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

