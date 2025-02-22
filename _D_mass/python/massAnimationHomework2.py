import matplotlib.pyplot as plt
import numpy as np
import massParam as P
from signalGenerator import signalGenerator
from massAnimation import massAnimation
from dataPlotter import dataPlotter
import time

# instantiate reference input classes
reference = signalGenerator(amplitude=1, frequency=0.1)

 # instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()
t = P.t_start # time starts at t_start
time.sleep(3)
while t < P.t_end: # main simulation loop
    # set variables
    z = reference.sin(t)
    b = P.b
    k = P.k
    m = P.m

    F = m * ( -1 * z) + k * ( z ) + b * ( z )
    
    # update animation
    state = np.array([[z]]) #state is made of theta, and theta_dot

    animation.update(state)
    dataPlot.update(t, state, F)
    # advance time by t_plot
    t = t + P.t_plot
    plt.pause(0.001) # allow time for animation to draw

 # Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()