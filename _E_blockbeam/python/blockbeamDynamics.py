import numpy as np
import blockbeamParam as params
import math as Math

class blockbeamDynamics:
    def __init__(self, sample_rate):
        # Initial state conditions
        y0 = 0.5
        ydot0 = 0.01
        theta0 = 0.0
        thetadot0 = 0.0
        self.first = True
        self.state = np.array([
            [y0],  # initial condition for y
            [theta0],  # initial condition for ydot
            [ydot0],  # initial condition for y
            [thetadot0]  # initial condition for ydot
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
        theta = state.item(1)
        ydot = state.item(2)
        thetadot = state.item(3)
        
        # The equations of motion
        yddot = y * thetadot ** 2 - params.g * np.sin(theta)
        print("sup" + str(theta))
        print("sup" + str(yddot))
        #thetaddotNumerator = u * params.length * Math.cos(theta) - 2 * params.m1 * y * ydot * thetadot - params.m1 * params.g * y * Math.cos(theta)- params.m2 * params.g * params.length * Math.cos(theta) / 2
        thetaddotNumerator = -(u * params.length - params.m1 * params.g * y - params.m2 * params.g * params.length / 2.0) * np.cos(theta) - 2 * params.m1 * y * ydot * thetadot
        thetaddotDenominator = (params.m2 * params.length) / 3.0 + params.m1 * y **2
        thetaddot = thetaddotNumerator / thetaddotDenominator
        

        if (self.first):
            #print([y, ydot, yddot, theta, thetadot, thetaddotNumerator, thetaddotDenominator, thetaddot])
            print([y, ydot, yddot, theta, thetadot, thetaddot])
            self.first = False

        
        # Build xdot and return
        xdot = np.array([[ydot], [thetadot], [yddot], [thetaddot]])
        return xdot
    
    def h(self):
        # Returns the measured output y = h(x)
        return [self.state.item(0), self.state.item(2)]
    
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
