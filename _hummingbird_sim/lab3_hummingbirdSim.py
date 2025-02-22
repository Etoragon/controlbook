import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics

# instantiate reference input classes
phi_ref = SignalGenerator(amplitude=1.5, frequency=0.05)
theta_ref = SignalGenerator(amplitude=1.0, frequency=0.3)
psi_ref = SignalGenerator(amplitude=0.5, frequency=.5)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()
mass = HummingbirdDynamics()


t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    u = np.array([[0.5], [0.5]])

    state = mass.update(u)

    animation.update(t, state)
    dataPlot.update(t, state, u)

    t += P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()

