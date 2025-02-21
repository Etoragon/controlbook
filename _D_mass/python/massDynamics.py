import numpy as np
import massParam as params

class massDynamics:
    def __init__(self, sample_rate):
        # Initial state conditions
        y0 = 2.0
        ydot0 = 1.0
        self.state = np.array([
            [y0],  # initial condition for y
            [ydot0]  # initial condition for ydot
        ])
        self.Ts = sample_rate  # sample rate of system
        self.limit = 1.0  # input saturation limit
        
        # System parameters
        self.a0 = 3.0
        self.a1 = 2.0
        self.b0 = 4.0
        
        # Modify the system parameters by random value
        alpha = 0.2  # Uncertainty parameter
        self.a1 *= (1. + alpha * (2. * np.random.rand() - 1.))
        self.a0 *= (1. + alpha * (2. * np.random.rand() - 1.))
        self.b0 *= (1. + alpha * (2. * np.random.rand() - 1.))
    
    def f(self, state, u):
        # For system xdot = f(x,u), return f(x,u)
        y = state.item(0)
        ydot = state.item(1)
        
        # The equations of motion
        yddot = -params.b * ydot - 5 * y + 1 * u
        
        # Build xdot and return
        xdot = np.array([[ydot], [yddot]])
        return xdot
    
    def h(self):
        # Returns the measured output y = h(x)
        return self.state.item(0)
    
    def update(self, u):
        # This is the external method that takes the input u(t)
        # and returns the output y(t)
        # u = self.saturate(u, self.limit)  # Saturate the input
        self.rk4_step(u)  # Propagate the state by one time step
        return self.h()  # Compute the output at the current state
    
    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
    
    def saturate(self, u, limit):
        return np.clip(u, -limit, limit)
