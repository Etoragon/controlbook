import matplotlib.pyplot as plt
import numpy as np
import rodMassParam as P
from signalGenerator import signalGenerator
from rodMassAnimation import rodMassAnimation
from dataPlotter import dataPlotter
from rodMassDynamics import rodMassDynamics
from ctrlPD import ctrlPD

# instantiate system, controller, and reference classes
rodMass = rodMassDynamics()
controller = ctrlPD()
reference = signalGenerator(amplitude=20*np.pi/180.0, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = rodMassAnimation()

t = P.t_start
y = rodMass.h()
while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = reference.square(t)
        d = 0.5
        n = 0.0  #noise.random(t)
        u = controller.update(r, y + n)
        y = rodMass.update(u + d)
        t += P.Ts
    # update animation and data plots
    animation.update(rodMass.state)
    dataPlot.update(t, r, rodMass.state, u)
    plt.pause(0.01)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
