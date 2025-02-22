import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter

# instantiate reference input classes
phi_ref = SignalGenerator(amplitude=1.5, frequency=0.05)
theta_ref = SignalGenerator(amplitude=1.0, frequency=0.3)
psi_ref = SignalGenerator(amplitude=0.5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    phi = phi_ref.sin(t)
    theta = theta_ref.sin(t)
    psi = psi_ref.sin(t)
    # update animation
    state = np.array([[phi], [theta], [psi], [0.0], [0.0], [0.0]])
    ref = np.array([[0], [0], [0]])
    force = 1000000
    torque = 200000 # Ethan Patten (eppatten) Couldn't figure out what these do yet
    # convert force and torque to pwm values
    pwm = P.mixing @ np.array([[force], [torque]]) / P.km
    animation.update(t, state)
    dataPlot.update(t, state, pwm)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()

