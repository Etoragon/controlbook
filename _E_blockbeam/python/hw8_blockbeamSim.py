import matplotlib.pyplot as plt
import blockbeamParam as P
from blockbeamDynamics import blockbeamDynamics
from ctrlPD import ctrlPD
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
# instantiate blockbeam, controller, and reference classes
blockbeam = blockbeamDynamics(alpha=0.2)
controller = ctrlPD()

#HW asked for reference input frequency of 0.01, but this is more interesting
reference = signalGenerator(amplitude=0.15, frequency=0.05)
disturbance = signalGenerator(amplitude=0.0, frequency=0.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()
t = P.t_start  # time starts at t_start
y = blockbeam.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = blockbeam.state
        u = controller.update(r, x)  # update controller
        y = blockbeam.update(u + d)  # propagate system
        t += P.Ts  # advance time by Ts
        
    # update animation and data plots
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, r)
    plt.pause(0.1)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
