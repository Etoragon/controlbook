import matplotlib.pyplot as plt
import massParam as P
from signalGenerator import signalGenerator
from massAnimation import massAnimation
from dataPlotter import dataPlotter
from massDynamics import massDynamics
import time

# instantiate pendulum, controller, and reference classes
mass = massDynamics(0.1)
force = signalGenerator(amplitude=1, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()

t = P.t_start  # time starts at t_start
time.sleep(3)
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        u = 0
        if ((force.sin(t) > 0.85)):
            u = 20
        y = mass.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(mass.state)
    dataPlot.update(t, mass.state, u)
    plt.pause(0.001)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
