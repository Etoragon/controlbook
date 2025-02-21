import matplotlib.pyplot as plt
import numpy as np
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter

# instantiate reference input classes
reference = signalGenerator(amplitude=0.3, frequency=0.1, y_offset=0.5)
reference2 = signalGenerator(amplitude=0.1, frequency=0.1,)

 # instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()
t = P.t_start # time starts at t_start

while t < P.t_end: # main simulation loop
    # set variables
    z = reference.sin(t)
    theta = reference2.sin(t)
    F = 0

    
    
    # update animation
    state = np.array([[z], [theta]]) #state is made of theta, and theta_dot

    animation.update(state)
    dataPlot.update(t, state, F)
    # advance time by t_plot
    t = t + P.t_plot
    plt.pause(0.001) # allow time for animation to draw

 # Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()