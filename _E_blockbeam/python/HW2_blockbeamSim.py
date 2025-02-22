import matplotlib.pyplot as plt
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
import blockbeamParam as params
import math as Math

# instantiate pendulum, controller, and reference classes
mass = blockbeamDynamics(0.1)
force = 0 # signalGenerator(amplitude=1, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()

firstHere = 20

t = P.t_start  # time starts at t_start
y = 0.5
state = [0, 0]
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        u = (params.m1 * params.g / params.length) * y + (params.m2 * params.g / (2))
        if (firstHere > 0):
            print(u)
            firstHere = firstHere - 1
        state = mass.update(u)  # Propagate the dynamics
        y = state[0]
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(mass.state)
    dataPlot.update(t, mass.state, u)
    plt.pause(0.001)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
