import numpy as np
import massParam as P

class ctrlPDClass:
    def __init__(self):
        """
        Initializes the PD controller with given proportional (kp) and derivative (kd) gains.
        """
        self.kp = 10.5
        self.kd = 12
        print('kp:', self.kp)
        print('kd:', self.kd)
    
    def update(self, z_r, x):
        """
        Compute the PD control input.
        :param z_r: Desired position (reference input)
        :param x: State vector [position; velocity]
        :return: Control force u
        """
        z = x[0][0]  # Position
        zdot = x[1][0]  # Velocity
        
        # Compute the PD control force
        u_tilde = self.kp * (z_r - z) - self.kd * zdot
        
        # Apply force saturation
        u = self.saturate(u_tilde, P.F_max)
        
        return u
    
    def saturate(self, u, limit):
        """
        Saturate the control input to stay within limits.
        """
        if abs(u) > limit:
            u = limit * np.sign(u)
        return u
